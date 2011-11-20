#!/usr/bin/env python

import sys,os,pickle
from scipy import signal,cluster,spatial
from scipy.stats import distributions
import numpy as nu
from imageUtil import writeImageData, getPattern, findValleys, smooth, normalize
from utilities import argpartition, partition, makeColors
from agent import Agent, AgentPainter, makeAgentClass


def assignToAgents(v,agents,AgentType,M,vert=None,horz=None,fixAgents=False):
    data = nu.nonzero(v)[0]
    if len(data) > 1:
        candidates = [tuple(x) if len(x)==1 else (x[0],x[-1]) for x in 
                      nu.split(data,nu.nonzero(nu.diff(data)>1)[0]+1)]
    elif len(data) == 1:
        candidates = [tuple(data)]
    else:
        return agents
    #print('candidates',candidates)
    if vert is not None:
        candidates = [[nu.array([vert,horz]) for horz in horzz] for horzz in candidates]
    elif horz is not None:
        candidates = [[nu.array([vert,horz]) for vert in vertz] for vertz in candidates]
    else:
        print('error, need to specify vert or horz')

    unadopted = []
    bids = None
    newagents =[]
    if len(agents) == 0:
        unadopted.extend(range(len(candidates)))
    else:
        #print('agents, candidates',len(agents),len(candidates))
        bids = nu.zeros((len(candidates),len(agents)))
        for i,c in enumerate(candidates):
            bids[i,:] = nu.array([nu.abs(a.bid(*c)) for a in agents])
            
        sortedBets = nu.argsort(bids,1)
        cidx = nu.argsort(sortedBets[:,0])
        adopters = set([])
        for i in cidx:
            bestBidder = sortedBets[i,0]
            bestBet = bids[i,bestBidder]
            if bestBet <= agents[bestBidder].maxError and not bestBidder in adopters:
                agents[bestBidder].award(*candidates[i])
                adopters.add(bestBidder)
                newagents.append(agents[bestBidder])
            else:
                unadopted.append(i)
        newagents.extend([agents[x] for x in set(range(len(agents))).difference(adopters)])
    if not fixAgents:
        for i in unadopted:
            if len(candidates[i]) >1 and (candidates[i][-1][0]-candidates[i][0][0]) <= M/50.:
                # only add an agent if we are on a small section
                newagent = AgentType(nu.mean(nu.array(candidates[i]),0))
                newagents.append(newagent)
    
    return [a for a in newagents if a.tick(fixAgents)]

def selectColumns(vsums,bins):
    N = len(vsums)
    nzidx = nu.nonzero(vsums)[0]
    binSize = int(nu.floor(len(nzidx)/float(bins)))
    #print(len(nzidx),binSize,bins)
    idxm = nzidx[:bins*binSize].reshape((bins,binSize))
    for i in range(bins):
        idxm[i,:] = idxm[i,nu.argsort(vsums[idxm[i,:]])]

    columns = idxm.T.ravel()
    colBins = nu.array(range(bins)*binSize)

    columns = nu.append(columns,nzidx[bins*binSize:])
    # incorrect, but mostly irrelevant:
    colBins = nu.append(colBins,nu.zeros(len(nzidx)-bins*binSize))
    assert len(columns) == len(colBins)
    return columns,colBins

def mergeAgents(agents):
    if len(agents) < 3:
        return agents
    newagents = []
    N = len(agents)
    pdist = []
    for i in range(N-1):
        for j in range(i+1,N):
            if agents[i].points.shape[0] < 2 or agents[j].points.shape[0] < 2:
                pdist.append(agents[i].maxError+1)
            else:
                #cAngle = (nu.arctan2(*(agents[i].mean-agents[j].mean))%nu.pi)/nu.pi
                cAngle = ((nu.arctan2(*(agents[i].mean-agents[j].mean))/nu.pi)+1)%1
                # fast check: are means in positions likely for merge?
                if True: #((cAngle-agents[i].targetAngle+.5)%1-.5) < agents[i].maxAngleDev:
                #if nu.abs(cAngle-agents[i].targetAngle) < agents[i].maxAngleDev:
                    # yes, do further check
                    pdist.append(agents[i].mergeable(agents[j]))
                else:
                    # no, exclude
                    pdist.append(agents[i].maxError+1)
    pdist = nu.array(pdist)
    l = cluster.hierarchy.complete(pdist)
    c = cluster.hierarchy.fcluster(l,agents[0].maxError,criterion='distance')
    idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
    for v in idict.values():
        if len(v) == 1:
            newagents.append(agents[v[0]])
        else:
            a = agents[v[0]]
            for i in v[1:]:
                a.merge(agents[i])
            newagents.append(a)
    return newagents

def sortAgents(agents,k=10):
    agents.sort(key=lambda x: -x.score)
    N = len(agents)
    scores = nu.array([a.score for a in agents])
    if nu.min(scores) == nu.max(scores):
        return agents

    meanScorePerSystem = [nu.mean(scores[i:i+k]) for i in range(0,N,k)]
    dms = nu.diff(meanScorePerSystem)
    nsystems = nu.argmin(dms)+1
    print('estimating ',nsystems,'groups,',k*nsystems,'stafflines')
    na = agents[:k*nsystems]
    print('keeping',len(na),'agents')
    return na

def getOffset(v1,v2,dx,maxAngle):
    #kmax = min(70,int(1.5*nu.ceil(dx*nu.tan(maxAngle*nu.pi))))
    kmax = 50
    N = len(v1)
    dotproducts = []
    rng = []
    for i in range(-kmax,kmax+1):
        b = min(max(0,i),N)
        e = max(0,min(N,N-i))
        if e > 0:
            dotproducts.append(nu.dot(v1[:e],v2[b:]))
            rng.append(i)
    ndp = normalize(nu.array(dotproducts))
    return ndp, -rng[nu.argmax(ndp)]

def getter(f):
    "Stores the value for later use"
    def _getter(self):
        if not hasattr(self,'valuedict'):
            self.valuedict = {}
        if not self.valuedict.has_key(f):
            self.valuedict[f] = f(self)
        return self.valuedict[f]
    return _getter

def assessStaffLineAgents(iagents,M,nPerStaff):
    agents = sortAgents(iagents,nPerStaff)
    for a in agents:
        print(a)
    meansAngles = nu.array([(a.mean[0],a.mean[1],a.getAngle()) for a in agents])
    x = meansAngles[:,0]+(M/2-meansAngles[:,1])*nu.tan(meansAngles[:,2]*nu.pi)
    xs = nu.sort(x)
    dxs = nu.diff(xs)
    l0 = nu.median(dxs)
    checkIdx = nu.ones(len(agents)-1,nu.bool)
    checkIdx[nu.arange(nPerStaff,len(agents),nPerStaff)-1] = False
    #thr = l0/20.
    thr = l0/10.
    result =nu.std(dxs[checkIdx]) < thr
    return result, agents

class VerticalSegment(object):
    def __init__(self,scoreImage,top,bottom,colGroups=11,
                 maxAngle=2/180.,nAngleBins=300):
        self.scrImage = scoreImage
        self.top = top
        self.bottom = bottom
        self.maxAngle = maxAngle
        self.nAngleBins = nAngleBins
        self.colGroups = colGroups
        self.nPerStaff = 5
        self.containsStaff = True

    def getVHSums(self):
        vsum = len(nu.nonzero(self.getVSums())[0])
        hsum = len(nu.nonzero(self.getHSums())[0])
        return vsum, hsum

    def flagNonStaff(self):
        self.containsStaff = False

    def hasStaff(self):
        return self.containsStaff
        
    @getter
    def getStaffLines(self):
        agents = []
        defAngle = self.getAngle()
        cols = selectColumns(self.getVSums(),self.colGroups)[0]
        StaffAgent = makeAgentClass(targetAngle=defAngle,
                                    maxAngleDev=2/180.,
                                    maxError=3,
                                    minScore=-2,
                                    offset=self.top)
        draw = False
        f0 = os.path.splitext(self.scrImage.fn)[0]
        print('default angle for this staff',defAngle)
        stop = False
        finalStage = False
        nFinalRuns = 10
        for i,c in enumerate(cols):
            if nFinalRuns == 0:
                break
            agentsnew = assignToAgents(self.getImgSegment()[:,c],agents,StaffAgent,
                                       self.scrImage.getWidth(),horz=c,fixAgents=finalStage)
            if len(agentsnew) > 1:
                agentsnew = mergeAgents(agentsnew)
            
            agents = agentsnew
            agents.sort(key=lambda x: -x.score)

            if len(agents) > 5 and i > 50 and not finalStage:
                finalStage,selection = assessStaffLineAgents(agents,self.scrImage.getWidth(),
                                                             self.nPerStaff)
                if finalStage:
                    agents = selection
            
            if finalStage:
                nFinalRuns -= 1

            if draw:
                self.scrImage.ap.reset()
                self.scrImage.ap.paintVLine(c)
                for a in agents:
                    if not self.scrImage.ap.isRegistered(a):
                        self.scrImage.ap.register(a)
                    self.scrImage.ap.drawAgentGood(a,-3000,3000)
                self.scrImage.ap.writeImage(f0+'-{0:04d}-c{1}'.format(i,c)+'.png')
        agents.sort(key=lambda x: x.getMiddle(self.scrImage.getWidth()))
        return [agents[k*self.nPerStaff:(k+1)*self.nPerStaff] 
                for k in range(len(agents)/self.nPerStaff)]

    def getImgSegment(self):
        return self.scrImage.getImg()[self.top:self.bottom,:]

    def getHSums(self):
        return self.scrImage.getHSums()[self.top:self.bottom]

    @getter
    def getVSums(self):
        return nu.sum(self.getImgSegment(),0)

    @getter
    def getAngleHistogram(self):
        #self.vSums = nu.sum(self.scrImage.getImg()[self.top:self.bottom,:],0)
        hparts = 3
        cols = selectColumns(self.getVSums(),hparts)[0]
        angles = []
        nColsToProcess = int(2*len(cols)/10)
        for j in range(nColsToProcess):
            dx = cols[j]-cols[j+1]
            dps,dy = getOffset(self.scrImage.getImg()[self.top:self.bottom,cols[j]],
                               self.scrImage.getImg()[self.top:self.bottom,cols[j+1]],
                               nu.abs(dx),self.maxAngle)
            if nu.min(dps) < 0.5: # min is only > 0 when dps has zero range
                angles.append((nu.arctan2(dy,dx)/nu.pi+.5)%1-.5)
        histrange = (-self.maxAngle,self.maxAngle)
        #bins,lims = nu.histogram(angles,bins=self.nbins,range=histrange)
        return nu.histogram(angles,bins=self.nAngleBins,range=histrange)[0]

    @getter
    def getAngle(self):
        """estimate the angle by taking the argmax of the angle histogram,
        weighted by the global angle histogram of the page.
        """
        i = nu.argmax(self.getAngleHistogram()*self.scrImage.getWeights())
        #nu.savetxt('/tmp/h{0}'.format(i),self.getAngleHistogram()*weights)
        return (float(self.maxAngle)/self.nAngleBins)-\
            self.maxAngle+i*2.0*self.maxAngle/self.nAngleBins

def identifyNonStaffSegments(vertSegments,N,M):
    """Identify vertical segments that are unlikely to contain staffs
    """
    vhsums = nu.array([vs.getVHSums() for vs in vertSegments],nu.int)
    meds = nu.median(vhsums,0)
    ref = nu.array((N/vhsums.shape[0],M))
    nvhsums = vhsums-meds
    nvhsums[nvhsums>0] = 0
    dref = nu.sum(ref**2)**.5
    ndists = (nu.sum(nvhsums**2,1)**.5)/dref
    nonStaff = nu.nonzero(nu.logical_not(ndists < .2))[0]
    #nvhsums = nu.column_stack((nvhsums,vhsums[:,-1]))
    #nvhsums = nu.vstack((nvhsums,nu.append(-ref,-1)))
    #nu.savetxt('/tmp/ns.txt',nvhsums,fmt='%d')
    print('of {0} segments, items {1} were identified as non-staff'.format(len(vertSegments),nonStaff))
    return nonStaff

class Staff(object):
    def __init__(self,scoreImage,staffLineAgents,top,bottom):
        self.scrImage = scoreImage
        self.staffLineAgents = staffLineAgents
        self.top = top
        self.bottom = bottom
        self.staffLineAgents.sort(key=lambda x: x.getMiddle(self.scrImage.getWidth()))
    def __str__(self):
        return 'Staff {0}; nAgents: {1}; avggap: {2}'\
            .format(self.__hash__(),len(self.staffLineAgents),self.getStaffLineDistance())
    def draw(self):
        for agent in self.staffLineAgents:
            self.scrImage.ap.register(agent)
            self.scrImage.ap.drawAgentGood(agent,-self.scrImage.getWidth(),self.scrImage.getWidth())

    def getAngle(self):
        print('staff angles',[(a.getAngle()+.5)%1-.5 for a in self.staffLineAgents])
        return nu.mean([(a.getAngle()+.5)%1-.5 for a in self.staffLineAgents])

    @getter
    def getTopBottom(self):
        lTop = nu.array((0,0))
        lBot = nu.array((self.scrImage.getHeight()-1,0))
        rTop = nu.array((0,self.scrImage.getWidth()-1))
        rBot = nu.array((self.scrImage.getHeight()-1,self.scrImage.getWidth()-1))
        x0offset = self.staffLineAgents[0].offset
        x1offset = self.staffLineAgents[-1].offset
        xx = nu.sort([self.staffLineAgents[0].getIntersection(lTop,lBot)[0]+x0offset,
                      self.staffLineAgents[0].getIntersection(rTop,rBot)[0]+x0offset,
                      self.staffLineAgents[-1].getIntersection(lTop,lBot)[0]+x1offset,
                      self.staffLineAgents[-1].getIntersection(rTop,rBot)[0]+x1offset])
        return xx[0],xx[-1]

    @getter
    def getStaffLineDistance(self):
        return nu.mean(nu.diff([a.getMiddle(self.scrImage.getWidth()) for a in self.staffLineAgents]))

class System(object):
    def __init__(self,scoreImage,staffs):
        self.scrImage = scoreImage
        self.staffs = staffs
        
    def getTop(self):
        return self.staffs[0].top
    def getBottom(self):
        return self.staffs[1].bottom
        
    def getLowerLeft(self):
        #meanLower = self.staffs[1].staffLineAgents[-1].getDrawMean()
        #return nu.array((meanLower[0]-meanLower[1]*nu.tan(nu.pi*self.getStaffAngle()),0))
        return self.getLowerLeftRight()[0]

    def getUpperLeft(self):
        #meanLower = self.staffs[1].staffLineAgents[-1].getDrawMean()
        #return nu.array((meanLower[0]-meanLower[1]*nu.tan(nu.pi*self.getStaffAngle()),0))
        return self.getUpperLeftRight()[0]

    @getter
    def getLowerLeftRight(self):
        #meanLower = self.staffs[1].staffLineAgents[-1].getDrawMean()
        #return nu.array((meanLower[0]-meanLower[1]*nu.tan(nu.pi*self.getStaffAngle()),0))

        # self.Qg, the point in the middle of the lower border of the system 
        # in the coordinate system of the original picture (global)
        # this is the point around which the rotation is done
        self.Qg = nu.array((self.getBottom(),int(self.scrImage.getWidth()/2.)))
        bot = self.Qg
        dyl = -bot[1]
        dyr = self.scrImage.getWidth()-bot[1]
        botLeft = bot[0]+dyl*nu.tan(nu.pi*self.getStaffAngle())
        botRight = bot[0]+dyr*nu.tan(nu.pi*self.getStaffAngle())
        correction = min(0,nu.floor(self.scrImage.getHeight()-max(botLeft,botRight)-1))
        print(self.scrImage.getHeight(),botLeft,botRight,correction,self.getStaffAngle())
        print(self.getBottom(),botLeft+correction-1,botRight+correction)
        return nu.array((botLeft+correction,0)),nu.array((botRight+correction,self.scrImage.getWidth()))
 
    @getter
    def getUpperLeftRight(self):
        top = nu.array((self.getTop(),int(self.scrImage.getWidth()/2.)))
        dyl = -top[1]
        dyr = self.scrImage.getWidth()-top[1]
        topLeft = top[0]+dyl*nu.tan(nu.pi*self.getStaffAngle())
        topRight = top[0]+dyr*nu.tan(nu.pi*self.getStaffAngle())
        correction = min(0,nu.floor(min(topLeft,topRight))-1)
        #min(0,self.scrImage.getHeight()-max(botLeft,botRight))
        return nu.array((topLeft-correction,0)),nu.array((topRight-correction,self.scrImage.getWidth()))

    @getter
    def getUpperLeftOld(self):
        #meanLower = self.staffs[0].staffLineAgents[0].getDrawMean()
        #return nu.array((meanLower[0]-meanLower[1]*nu.tan(nu.pi*self.getStaffAngle()),0))
        top = nu.array((self.getTop(),int(self.scrImage.getWidth()/2.)))
        return nu.array((top[0]-top[1]*nu.tan(nu.pi*self.getStaffAngle()),0))

    def backTransform(self,d):
        print('Qg,Ql')
        print(self.Qg)
        print(self.Ql)
        dl = (d-self.Ql).reshape((-1,2))
        r = nu.sum(dl**2,1)**.5
        ph = nu.arctan2(dl[:,0],dl[:,1])-self.getStaffAngle()*nu.pi
        return (nu.column_stack((r*nu.sin(ph),r*nu.cos(ph)))+self.Qg).reshape(d.shape)
        
    @getter
    def getCorrectedImgSegment(self):
        #systemTop = self.staffs[0].getTopBottom()[0]
        #systemBot = self.staffs[1].getTopBottom()[1]
        sysHeight = (self.getLowerLeft()-self.getUpperLeft())[0]
        w1 = sysHeight*nu.tan(nu.pi*self.getStaffAngle())
        w2 = self.scrImage.getWidth()/nu.cos(nu.pi*self.getStaffAngle())
        if w1 < 0:
            # start late
            rng = nu.arange(int(nu.ceil(-w1)),int(nu.floor(w2)))
        else:
            # stop early
            rng = nu.arange(0,int(nu.floor(w2-w1)))
        hrng = nu.arange(sysHeight)
        col = nu.column_stack((-hrng*nu.cos(nu.pi*self.getStaffAngle()),
                                hrng*nu.sin(nu.pi*self.getStaffAngle())))
        row = nu.column_stack((rng*nu.sin(nu.pi*self.getStaffAngle()),
                               rng*nu.cos(nu.pi*self.getStaffAngle())))+self.getLowerLeft()
        z = nu.zeros((col.shape[0],row.shape[0]),nu.uint8)
        # the point in the middle of the lower border of the system 
        # in the coordinate system of the corrected image segment (local)
        # TODO get horizontal coordinate of Ql
        self.Ql = nu.array((sysHeight,.5*self.scrImage.getWidth()/nu.cos(nu.pi*self.getStaffAngle())-w1))
        
        for i,r in enumerate(row):
            rc = r+col
            xf = nu.floor(rc[:,0]).astype(nu.int)
            xc = nu.ceil(rc[:,0]).astype(nu.int)
            yf = nu.floor(rc[:,1]).astype(nu.int)
            yc = nu.ceil(rc[:,1]).astype(nu.int)
            wxc = rc[:,0]%1
            wxf = 1-wxc
            wyc = rc[:,1]%1
            wyf = 1-wyc
            
            z[::-1,i] = \
                (((wxf+wyf)*self.scrImage.getImg().flat[self.scrImage.getWidth()*xf+yf] + \
                (wxf+wyc)*self.scrImage.getImg().flat[self.scrImage.getWidth()*xf+yc] + \
                (wxc+wyf)*self.scrImage.getImg().flat[self.scrImage.getWidth()*xc+yf] + \
                (wxc+wyc)*self.scrImage.getImg().flat[self.scrImage.getWidth()*xc+yc])/4.0).astype(nu.uint8)
            #v = nu.round(rc).astype(nu.int)
            #z[::-1,i] = self.scrImage.getImg().flat[self.scrImage.getWidth()*v[:,0]+v[:,1]]
        #myap = AgentPainter(z)
        #myap.writeImage('tst')
        return z

    def draw(self):
        for staff in self.staffs:
            staff.draw()

    def getStaffAngle(self):
        return nu.mean([s.getAngle() for s in self.staffs])

    #def getVSums(self):
    #    self.scrImage.getVSums()[self.staffs[0].top]

    @getter
    def getHSums(self):
        return nu.sum(self.getCorrectedImgSegment(),1)

    def getBarLines(self):
        agents = []
        defBarAngle = .5 #(self.getStaffAngle()+.5)%1
        print('default staff angle for this system',self.getStaffAngle())
        print('default bar angle for this system',defBarAngle)
        assert defBarAngle >= 0
        BarAgent = makeAgentClass(targetAngle=defBarAngle,
                                  maxAngleDev=2/180.,
                                  maxError=.5,
                                  minScore=-5,
                                  offset=0)
        print('default angle for this system',defBarAngle)
        #cols = selectColumns(self.getVSums(),self.colGroups)[0]
        systemTop = self.staffs[0].getTopBottom()[0]
        systemBot = self.staffs[1].getTopBottom()[1]
        rows = selectColumns(self.getHSums(),3)[0] # sounds funny, change name of function
        finalStage = False
        k = 0
        for i,r in enumerate(rows[:int(.3*len(rows))]):
            agentsnew = assignToAgents(self.getCorrectedImgSegment()[r,:],agents,BarAgent,
                                       self.getCorrectedImgSegment().shape[1],vert=r,fixAgents=finalStage)
            agents = agentsnew
            print('row',i)
            if len(agents) > 1:
                k = sortBarAgents(agents)
            agents = agents[:20]

        for i,a in enumerate(agents):
            a.points = self.backTransform(a.points)
            a.mean = self.backTransform(a.mean)
            a.targetAngle -= self.getStaffAngle()
            print('{0} {1}'.format(i,a))
        return agents[:k]

def sortBarAgents(agents):
    agents.sort(key=lambda x: -x.score)
    scores = nu.append(nu.array([x.score for x in agents if x.score > 1]),0)
    hyp = 0
    if len(scores) > 1:
        hyp = nu.argmin(nu.diff(scores))
        print(scores)
        print('guessing:',hyp+1)
    return hyp+1

class ScoreImage(object):
    def __init__(self,fn):
        self.fn = fn
        self.typicalNrOfSystemPerPage = 6
        self.maxAngle = 1.5/180.
        self.nAnglebins = 600
        self.colGroups = 11
        self.bgThreshold = 20
        self.ap = AgentPainter(self.getImg())

    @getter
    def getImg(self):
        print('Loading image...'),
        sys.stdout.flush()
        try:
            img = 255-getPattern(self.fn,False,False)
        except IOError as e: 
            print('problem')
            raise e
        print('Done')
        img[img< self.bgThreshold] = 0
        return img

    def getWidth(self):
        return self.getImg().shape[1]
    def getHeight(self):
        return self.getImg().shape[0]

    @getter
    def getStaffs(self):
        # staffs get selected if their avg staffline distance (ASD) is
        # larger than thresholdPropOfMax times the largest ASD over all staffs
        thresholdPropOfMax = .75
        staffs = []
        for i,vs in enumerate(self.getStaffSegments()):
            print('Processing staff segment {0}'.format(i))
            staffs.extend([Staff(self,s,vs.top,vs.bottom) for s in vs.getStaffLines()])
        for staff in staffs:
            print(staff)
        slDists = nu.array([staff.getStaffLineDistance() for staff in staffs])
        print('avg staff line distance per staff:')
        print(slDists)
        maxDist = nu.max(slDists)
        print('original nr of staffs',len(staffs))
        staffs = list(nu.array(staffs)[slDists >= thresholdPropOfMax*maxDist])
        print('new nr of staffs',len(staffs))
        if len(staffs)%2 != 0:
            print('WARNING: detected unequal number of staffs!')
            print('TODO: retry to find an equal number of staffs')
        return staffs

    @getter
    def getSystems(self):
        staffs = self.getStaffs()
        assert len(staffs)%2 == 0
        return [System(self,(staffs[i],staffs[i+1])) for i in range(0,len(staffs),2)]

    def drawImage(self):
        # draw segment boundaries
        for i,vs in enumerate(self.getVSegments()):
            self.ap.paintHLine(vs.bottom,step=2)

        for vs in self.getNonStaffSegments():
            for j in range(vs.top,vs.bottom,4):
                self.ap.paintHLine(j,alpha=0.9,step=4)
                
        sysSegs = []
        for i,system in enumerate(self.getSystems()):
            if True: #i == 4:
                sys.stdout.write('drawing system {0}\n'.format(i))
                sys.stdout.flush()
                system.draw()
                sysSegs.append(system.getCorrectedImgSegment())
                #x = nu.array([[50,50],[100,30],[0,0]])
                #xr = system.backTransform(x)
                #print(xr)
                #nu.savetxt('/tmp/x.txt',x)
                #nu.savetxt('/tmp/xr.txt',xr)
                #barAgents = system.getBarLines()
                #for a in barAgents:
                #    self.ap.register(a)
                #    self.ap.drawAgentGood(a,-500,500)
        self.ap.writeImage(self.fn)
        return True
        shapes = nu.array([ss.shape for ss in sysSegs])
        ssH = nu.sum(shapes[:,0])
        ssW = nu.max(shapes[:,1])
        ssImg = nu.zeros((ssH,ssW),nu.uint8)
        x0 = 0
        for ss in sysSegs:
            h,w = ss.shape
            horzOffset = 0#int(nu.floor((ssW-w)/2.))
            ssImg[x0:x0+h,horzOffset:horzOffset+w] = ss
            x0 += h
        #ap1 = AgentPainter(ssImg)
        #ap1.writeImage(self.fn.replace('.png','-corr.png'))

    def drawImageSelection(self,selection):
        # draw segment boundaries
        for i,vs in enumerate(self.getVSegments()):
            self.ap.paintHLine(vs.bottom,step=2)

        for vs in self.getNonStaffSegments():
            for j in range(vs.top,vs.bottom,4):
                self.ap.paintHLine(j,alpha=0.9,step=4)

        ss = self.getStaffSegments()
        
        for i in selection:
            staffs = [Staff(self,x) for x in ss[i].getStaffLines()]
            system = System(self,staffs)
            system.draw()
            system.getBarLines()
            print(system)
        self.ap.writeImage(self.fn)

    @getter
    def getHSums(self):
        return nu.sum(self.getImg(),1)

    @getter
    def getVSegments(self):
        K = int(self.getHeight()/(2*self.typicalNrOfSystemPerPage))+1
        sh = smooth(self.getHSums(),K)
        #nu.savetxt('/tmp/vh1.txt',nu.column_stack((self.getHSums(),sh)))
        segBoundaries = nu.append(0,nu.append(findValleys(sh),self.getHeight()))
        vsegments = []
        for i in range(len(segBoundaries)-1):
            vsegments.append(VerticalSegment(self,segBoundaries[i],segBoundaries[i+1],
                                             colGroups = self.colGroups,
                                             maxAngle = self.maxAngle,
                                             nAngleBins = self.nAnglebins))
        nonStaff = identifyNonStaffSegments(vsegments,self.getHeight(),self.getWidth())
        for i in nonStaff:
            vsegments[i].flagNonStaff()
        return vsegments

    def getNonStaffSegments(self):
        return [vs for vs in self.getVSegments() if not vs.hasStaff()]

    def getStaffSegments(self):
        return [vs for vs in self.getVSegments() if vs.hasStaff()]
        #d = partition(lambda x: x.hasStaff(),self.getVSegments())
        #return d[True], d[False]

    @getter    
    def getWeights(self):
        globalAngleHist = smooth(nu.sum(nu.array([s.getAngleHistogram() for s 
                                                  in self.getVSegments()]),0),50)
        angles = nu.linspace(-self.maxAngle,self.maxAngle,self.nAnglebins+1)[:-1] +\
            (float(self.maxAngle)/self.nAnglebins)

        amax = angles[nu.argmax(globalAngleHist)]
        return distributions.norm(amax,.5/180.0).pdf(angles)
            
if __name__ == '__main__':
    fn = sys.argv[1]
    si = ScoreImage(fn)
    si.drawImage()
    sys.exit()

    simgfn = os.path.join('/tmp/',os.path.splitext(os.path.basename(fn))[0]+'.scrimg')
    if os.access(simgfn,os.R_OK):
        with open(simgfn,'r') as f:
            si = pickle.load(f)
    else:
        si = ScoreImage(fn)
        with open(simgfn,'w') as f:
            pickle.dump(si,f)
    staffs = si.getSystems()
    print(staffs)

"""
evaluate bar agents:
* how much overshoot above and below upper and lower stafflines resp?
* interruptions within staffs?
  * interruptions at end? -> probably curved staff, don't penalize
* how much interruption?
"""
