#!/usr/bin/env python

import sys,os
from scipy import signal,cluster
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition, makeColors
from main import convolve

def smooth(x,k):
    f = signal.blackmanharris(k)#nu.ones(k)/nu.float(k)
    return nu.convolve(x,f,'same')

def makeHZ(img):
    hl = nu.zeros(img.shape[0])
    nzz = [nu.nonzero(img[i,:])[0] for i in range(0,img.shape[0])]
    for i in range(1,img.shape[0]-1):
        # sum nonzero pixels of three consecutive rows (to accommodate for angle))
        nz = nu.array(list(set(nu.hstack((nzz[i-1],nzz[i],nzz[i+1])))))
        nz = nu.sort(nz)
        # dz: the longest horizontal white space within the 3-pixel band
        dz = nu.diff(nz)
        dz[dz == 1] = 0
        # nz are the places where whitespace occurs
        nz = nu.nonzero(dz)[0]
        if len(nz) < 2:
            hl[i] = 0
        else:
            hl[i] = nu.max(nu.diff(nz))**2
        #nz = nu.sort(nz)
    hl = hl-nu.median(hl)
    hl[hl<0] = 0
    return hl
    #nu.savetxt('/tmp/hl.txt',hl)

def findSystemLRNew(v):
    vp = v.copy()
    # margin is proportion of pagewidth from sides where dont look for system boundaries 
    vpmed = nu.median(vp)
    vpmin,vpmax = nu.min(vp),nu.max(vp)
    #print(vpmed)
    #print(vpmin,vpmax)
    N = vp.shape[0]
    margin = .05
    # TODO: catch situations where there are no lows in left or right halves
    lowidx = vp<(vpmin+(vpmed-vpmin)*margin)
    low = nu.sort(nu.nonzero(lowidx)[0])
    nu.savetxt('/tmp/nn.txt',nu.column_stack((v,lowidx.astype(nu.int))))
    lidx = low<N/2
    leftInvalid = False
    rightInvalid = False
    if any(lidx):
        vp[:int(nu.median(low[lidx]))] = vpmin
    else:
        # no whitespace column left of system
        # probably due to black scanning artifacts on the left
        leftInvalid = True
    ridx = nu.logical_not(lidx)
    if any(ridx):
        vp[int(nu.median(low[ridx])):] = vpmin
    else:
        # no whitespace column left of system
        # probably due to black scanning artifacts on the left
        rightInvalid = True

    vpmax = nu.max(vp)
    vp = normalize(vp)
    vpmed = nu.median(vp)
    vp = vp-vpmed/2.0
    
    if True:
        K = N/100.
        wsys = signal.gaussian(K,K/5.0)
        wsys = nu.hstack((wsys,-wsys))
        nu.savetxt('/tmp/wsys.txt',wsys)
        wconv = nu.convolve(vp,wsys,'same')
        #sidx = nu.argsort(wconv)
        #k = int((N-wconv.shape[0])/2)
        left = nu.argmax(wconv[:int(N/2)])+K/2.0
        right = N/2+nu.argmin(wconv[int(N/2):])-K/2.0
        #k+sidx[0]-K/2.0
        #right = nu.argmin(wconv)+K/2.0
        #wcon = nu.zeros(vp.shape[0])
        #wcon[k:k+wconv.shape[0]] = wconv
        nu.savetxt('/tmp/n.txt',nu.column_stack((v,vp,wconv)))
        print('lr',int(left),int(right))
        return -1 if leftInvalid else int(left), -1 if rightInvalid else int(right)
    else:
        K = N/60.
        wsys = signal.gaussian(K,K/5.0)
        wsys = nu.hstack((wsys,-wsys))
        nu.savetxt('/tmp/wsys.txt',wsys)
        wconv = nu.convolve(vp,wsys,'valid')
        #sidx = nu.argsort(wconv)
        k = int((N-wconv.shape[0])/2)
        left = k+nu.argmax(wconv[:int(N/2)])+K/2.0
        right = k+N/2+nu.argmin(wconv[int(N/2):])-K/2.0
        #k+sidx[0]-K/2.0
        #right = nu.argmin(wconv)+K/2.0
        wcon = nu.zeros(vp.shape[0])
        wcon[k:k+wconv.shape[0]] = wconv
        nu.savetxt('/tmp/n.txt',nu.column_stack((v,vp,wcon)))
        print('lr',int(left),int(right))
        return int(left),int(right)

def addPrior(pdf,proportion,prior=None):
    if prior == None:
        prior = nu.ones(pdf.shape[0])
    assert prior.shape[0] == pdf.shape[0]
    pdfsum = nu.sum(pdf)
    priorsum = nu.sum(prior)
    pdfFactor = (1-proportion)*pdf/pdfsum
    priorFactor = proportion*prior/priorsum
    return pdfFactor+priorFactor

def getIdxOfMaxima(v,sort=False):
    d2v = nu.diff(nu.sign(nu.diff(v)))
    idx = nu.nonzero(d2v == -2)[0]+1
    if sort:
        return idx[nu.argsort(v[idx])[::-1]]
    else:
        return idx

def findVertSystemLimits(centers,bottomCurve,topCurve):
    N = centers.shape[0]
    assert N > 0
    i = 0
    start = 0
    end = centers[i]
    limits = [[nu.argmax(topCurve[start:end])+start]]
    while i < N-1:
        i +=1
        start = end
        end = centers[i]
        limits[-1].append(nu.argmax(bottomCurve[start:end])+start)
        limits.append([nu.argmax(topCurve[start:end])+start])
    start = end
    limits[-1].append(nu.argmax(bottomCurve[start:])+start)
    return limits


def findBars(img,fn):
    # smooth to get rid of barline peaks
    # typical nr of staffs per page
    N = img.shape[0]
    # S: window size
    S = N/100 
    # K: nr of systems
    K = 4
    # impact of prior for different system pdfs to be multiplied
    prior = .5

    bg = nu.median(img)+10
    img[img < bg] = 0
    hp = makeHZ(img)
    vp = nu.sum(img,0)

    vp = vp-.5*nu.median(vp)
    print('white',bg)
    sl,sr = findSystemLRNew(vp)
    # TODO make sure the selection doesn't give index errors in pathetic cases:
    borderW = 20
    horzpdf = nu.log(addPrior(hp,proportion=prior))
    hp = horzpdf
    #hp = horzpdf*leftpdf*rightpdf
    out = horzpdf
    if sl >= 0: 
        leftpdf = nu.sum(img[:,sl-borderW:sl+borderW],1)
        leftpdf = nu.log(addPrior(leftpdf,proportion=prior))
        hp *= leftpdf
        out = nu.column_stack((out,leftpdf))
    else:
        hp = -hp
        print('no left pdf')
    if sr >= 0: 
        rightpdf = nu.sum(img[:,sr-borderW:sr+borderW],1)
        rightpdf = nu.log(addPrior(rightpdf,proportion=prior))
        hp *= rightpdf
        out = nu.column_stack((out,rightpdf))
    else:
        hp = -hp
        print('no right pdf')
    hp = hp-nu.mean(hp)
    out = nu.column_stack((out,hp))
    
    nu.savetxt('/tmp/hp.txt',out)
    # smooth curve to find systems
    k = N/nu.float(K)
    w = signal.gaussian(k,k/8.)/k
    c = nu.convolve(smooth(hp,S),w,'same')
    np = hp-nu.median(hp)
    systemCenters = getIdxOfMaxima(c)
    avgSystemDist = nu.median(nu.diff(nu.sort(systemCenters)))

    #wsys = signal.gaussian(avgSystemDist/5,avgSystemDist/5)/k
    # TESTING: REMOVED SPURIOUS k
    wsys = signal.gaussian(avgSystemDist/5.,avgSystemDist/5.)
    wsys = nu.hstack((wsys,-wsys))
    cwtop = nu.convolve(hp,wsys,'same')
    cwbot = nu.convolve(hp,-wsys,'same')
    limits = findVertSystemLimits(systemCenters,cwbot,cwtop)
    print('centers',systemCenters)
    print('limits',limits)

    # save image
    im_r = 255-img
    im_g = 255-img
    im_b = 255-img
    for i,(top,bot) in enumerate(limits[:]):
        # findBarsInSystem(img,limits[system][0],limits[system][1])
        left,right = findSysBlob(img,top,bot)
        margin = 5
        top -= margin
        left -= margin
        bot += margin
        right += margin
        print('finding stafflines in system',i)
        #staff_x_angle = findStaffLinesInSystem(img,top,bot,left,right)
        agents = findStaffLinesInSystem(img,top,bot,left,right)
        staff_x_angle = [(a.point[0]+top,a.angle) for a in agents]
        print(staff_x_angle)

        #print('finding bars in system',i)
        #barys = findBarsInSystem(img,top,bot,left,right)
        print(i,left,right,top,bot)
        im_g[top:bot,left] = 0
        im_r[top:bot,left] = 255
        im_g[top:bot,right] = 0
        im_r[top:bot,right] = 255
        im_g[top,left:right] = 0
        im_r[top,left:right] = 255
        im_g[bot,left:right] = 0
        im_r[bot,left:right] = 255

        print('left right',left,right)
        colors = makeColors(len(agents))
        for j,a in enumerate(agents):
            #v1 = int(nu.random.randint(255))
            #v2 = int(nu.random.randint(255))
            #v3 = int(nu.random.randint(255))
            for points in a.points:
                im_g[points[0]+top-1:points[0]+top+1,points[1]] = colors[j][0]
                im_r[points[0]+top-1:points[0]+top+1,points[1]] = colors[j][1]
                im_b[points[0]+top-1:points[0]+top+1,points[1]] = colors[j][2]
        continue 

        for s,angle in staff_x_angle:
            #im_g[s,left-10:right+10] = 0
            #im_r[s,left-10:right+10] = 255
            print(s,angle)
            horz = nu.arange(left-10,right+10)
            vert = (s-(horz-s)*nu.sin(2*nu.pi*angle))
            hv = nu.column_stack((horz,vert)).astype(nu.int)
            print(vert)
            for h,v in hv:
                im_g[v,h] = 255
                im_b[v,h] = 255
                im_r[v,h] = 0
            #im_r[vert,left-10:right+10] = 255
            #im_g[vert,left-10:right+10] = 0
            #im_r[vert,left-10:right+10] = 255
            #im_g[s,] = 0
        #for b in barys:
        #    im_g[top-50:bot+50,b] = 0
        #    im_r[top-50:bot+50,b] = 255

    #im_b = im_g
    writeImageData(os.path.join('/tmp',os.path.basename(fn).replace('.tif','.png')),img.shape,im_r,im_g,im_b)

def ravel_multi_index(pairs,D):
    pass
    
def findSysBlob(img,top,bot):
    # idea:
    # parameters: BGthreshold, maxBGWidth
    bgSurfacePercentage = 30 # assumed % of surface that counts as background
    bgThreshold = nu.percentile(img[top:bot,:].ravel(),bgSurfacePercentage)
    N,M = img.shape
    maxBgWidth = .01*N # gaps in system cannot exceed 1/100 of page width
    cmeans = nu.mean(img[top:bot,:],0)
    #nu.savetxt('/tmp/lc.txt',nu.column_stack((cmeans,cmeans > bgThreshold)))
    d = nu.diff((cmeans > bgThreshold).astype(nu.int))
    assert len(d) > 0
    gaps = nu.nonzero(d)[0]
    dgaps = nu.diff(gaps)
    dgaps = nu.append(dgaps,0)
    #n = nu.zeros((len(gaps),3),nu.int)
    #n[:,0] = gaps
    #n[:,1] = d[gaps]
    #n[:,2] = dgaps
    #print(n)
    startidx = d[gaps] == 1
    blobstarts = nu.nonzero(startidx)
    blobstartidx = nu.argmax(dgaps[blobstarts])
    begin = gaps[blobstarts][blobstartidx]
    end = begin+dgaps[blobstarts][blobstartidx]
    return begin, end


class Agent(object):
    targetAngleDeg = 0
    minScore = 0
    def __init__(self,xy):
        self.point = xy
        self.points = [xy]
        self.angle = self.targetAngleDeg
        self.score = 0
        self.adopted = True
        self.age = 0
        self.nadopted = 0
    def __str__(self):
        return 'Agent: point: {0}; angle: {1}; score: {2} age: {3}'.format(self.point,self.angle,self.score,self.age)
    def tick(self):
        self.age += 1
        if self.adopted:
            self.score += 1
        else:
            self.score -= 1
        self.adopted = False
        return not self.died()
    def getScore(self):
        return score
    def died(self):
        return self.score < Agent.minScore

    def getAngle(self,xy):
        return nu.arctan2(*nu.abs(self.point-xy))/(2*nu.pi)
        
    #def getBestAngle(self,xy0,xy1=None):
        
    def award(self,xy0,xy1=None):
        self.adopted = True
        self.nadopted += 1
        if False:
            return True

        bestAngle = nu.abs(self.bid(xy0,xy1))
        self.angle = ((self.nadopted-1)*self.angle+bestAngle)/self.nadopted
        if xy1 == None:
            newpoint = xy0
        else:
            #newpoint = (xy0+xy1)/2.
            if xy0[0] == xy1[0]:
                newpoint = nu.array((xy0[0],self.point[1]+(xy0[0]-self.point[0])*nu.cos(2*nu.pi*bestAngle)))
            if xy0[1] == xy1[1]:
                newpoint = nu.array((self.point[0]+(xy0[1]-self.point[1])*nu.sin(2*nu.pi*bestAngle),xy0[1]))
                if xy0[0] < xy1[0]:
                    smallest = xy0
                    largest = xy1
                else:
                    smallest = xy1
                    largest = xy0
                if newpoint[0] < smallest[0]:
                    newpoint = smallest
                if newpoint[0] > largest[0]:
                    newpoint = largest
                print('np',bestAngle,self.point,xy0,xy1,newpoint)
            else:
                print('Error, dont know what to do')
        self.point = ((self.nadopted-1)*self.point+newpoint)/self.nadopted
        self.points.append(newpoint)

    def evaluateAngle(self,a):
        return a-nu.abs(self.angle%360.)
      
    def bid(self,xy0,xy1=None):
        a0 = self.getAngle(xy0)
        e0 = self.evaluateAngle(a0)
        if xy1 is None:
            return e0
        a1 = self.getAngle(xy1)
        e1 = self.evaluateAngle(a1)
        if a0 <= a1:
            if e0 <= 0 and e1 >= 0:
                return 0
            else:
                if nu.abs(e0) < nu.abs(e1):
                    return e0
                else:
                    return e1
        else:
            if e1 <= 0 and e0 >= 0:
                return 0
            else:
                if nu.abs(e0) < nu.abs(e1):
                    return e0
                else:
                    return e1
                
class BarLineAgent(Agent):
    #minAngleDeg = 87
    #maxAngleDeg = 93
    targetAngleDeg = 90
    minScore = -10
    
class StaffLineAgent(Agent):
    #minAngleDeg = -1
    #maxAngleDeg = 1
    targetAngleDeg = 0
    minScore = -5
        
def getCrossings(v,agents,AgentType,vert=None,horz=None):
    mindist = 1
    data = nu.nonzero(v)[0]
    if len(data) > 1:
        candidates = [x if len(x)==1 else (x[0],x[-1]) for x in nu.split(data,nu.nonzero(nu.diff(data)>1)[0]+1)]
    elif len(data) == 1:
        candidates = [data]
    else:
        return agents
    #print(vert)
    #print('protocandidates',candidates)
    if vert is not None:
        candidates = [[nu.array([vert,horz]) for horz in horzz] for horzz in candidates]
    elif horz is not None:
        candidates = [[nu.array([vert,horz]) for vert in vertz] for vertz in candidates]
    else:
        print('error, need to specify vert or horz')
    #print('candidates',len(candidates))
    #print('candidates',candidates)
    #print('agents',len(agents))
    unadopted = []
    maxDeviation = .5/360.
    for i,c in enumerate(candidates):
        if len(agents) == 0:
            unadopted.append(i)
        else:
            bids = [nu.abs(a.bid(*c)) for a in agents]
            bestBidder = nu.argmin(bids)
            #print('bids',bids)
            if bids[bestBidder] < maxDeviation:
                #print('c',bids[bestBidder],maxDeviation)
                agents[bestBidder].award(*c)
            else:
                unadopted.append(i)
    for i in unadopted:
        agents.append(AgentType(nu.mean(nu.array(candidates[i]),0)))
        #for j in candidates[i]:
            #print('unadopted',j)
        
    return [a for a in agents if a.tick()]


def findStaffLinesInSystem(img,top,bot,left,right):
    maxStaffLines = 15
    lookatProportion = .2
    vsums = nu.sum(img[top:bot,left:right],0)
    #vsums = vsums[nu.nonzero(vsums)[0]]
    colorder = nu.argsort(vsums)
    colorder = colorder[nu.nonzero(vsums[colorder])]
    #print(vsums[colorder[:10]])
    #print('first nonz:',nu.where(vsums[colorder]>0))

    #center = int((bot+top)/2)
    agents = []
    N = right-left
    for c in colorder[:int(lookatProportion*N)]:
    #for c in colorder:
        #print('column',c+left)
        agents = getCrossings(img[top:bot,left+c],agents,StaffLineAgent,horz=left+c)
        #print('agents',len(agents))
    #sys.exit()
    print(N/2)
    agents = agents[:maxStaffLines]
    agents.sort(key=lambda x: x.age*x.score)
    agents.reverse()
    for a in agents:
        print('{1} {0}'.format(a,a.point[0]+top))
    k = nu.argmin(nu.diff([a.score for a in agents]))
    print(k+1,'stafflines found')
    for a in agents[:k+1]:
        print('{1} {0}'.format(a,a.point[0]+top))
    print('nr of agents',len(agents))
    return agents[:10]
    #return [(a.point[0]+top,a.angle) for a in agents[:10]]

def findBarsInSystem(img,top,bot,left,right):
    maxBarLines = 20
    lookatProportion = .3
    hsums = nu.sum(img[top:bot,left:right],1)
    roworder = nu.argsort(hsums)
    #center = int((bot+top)/2)
    agents = []
    N = bot-top
    for r in roworder[:int(lookatProportion*N)]:
        agents = getCrossings(img[top+r,left:right],agents,BarLineAgent,vert=top+r)
    print(N/2)
    agents = agents[:maxBarLines]
    agents.sort(key=lambda x: x.score)
    agents.reverse()
    for a in agents:
        print('{1} {0}'.format(a,a.point[1]+left))
    k = nu.argmin(nu.diff([a.score for a in agents]))
    print(k+1,'barlines found')
    for a in agents[:k+1]:
        print('{1} {0}'.format(a,a.point[1]+left))
    return [a.point[1]+left for a in agents[:k+1]]

if __name__ == '__main__':
    fn = sys.argv[1]
    print('Loading image...'),
    sys.stdout.flush()
    try:
        img = 255-getPattern(fn,False,False)
    except IOError as e: 
        print('problem')
        raise e
        sys.exit()#pass
    print('Done')
    findBars(img,fn)
    #nu.savetxt('/tmp/p.txt',nu.sum(img,1))
    sys.exit()    

    N,M = img.shape
    K = N/3
    w = signal.gaussian(K,K/4.0).reshape((1,-1))
    nu.savetxt('/tmp/w.txt',w[0,:])
    img1 = convolve(img,w)
    img1[img1 < 0 ] = 0
    img1 = 1/(1+nu.exp(-100*(normalize(img1)-.5)))
    
    nu.savetxt('/tmp/c.txt',img1)
    findBarsTry(img1,'/tmp/p1.txt')

#problematic:
"""
chopin:
7 5 1
18 1 1
35 2 7

"""
