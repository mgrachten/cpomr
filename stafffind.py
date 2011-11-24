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
        if False: #vert == 326:
            for j,a in enumerate(agents):
                print('{0} {1}'.format(j,a))
            print('candidates',candidates)
            print(sortedBets)
            print(bids[-1,sortedBets[-1,:]])
            sys.exit()
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
    
    #return [a for a in newagents if a.tick(fixAgents)]
    r = partition(lambda x: x.tick(fixAgents),newagents)
    return r.get(True,[]),r.get(False,[])

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
    died = []
    for v in idict.values():
        if len(v) == 1:
            newagents.append(agents[v[0]])
        else:
            a = agents[v[0]]
            for i in v[1:]:
                a.merge(agents[i])
            newagents.append(a)
            died.extend(v[1:])
    return newagents,died

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
            agentsnew,died = assignToAgents(self.getImgSegment()[:,c],agents,StaffAgent,
                                            self.scrImage.getWidth(),horz=c,fixAgents=finalStage)
            if len(agentsnew) > 1:
                agentsnew,d = mergeAgents(agentsnew)
                
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
        #print('staff angles',[(a.getAngle()+.5)%1-.5 for a in self.staffLineAgents])
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
        self.barPoints = []
        # self.Qg, the point in the middle of the lower border of the system 
        # in the coordinate system of the original picture (global)
        # this is the point around which the rotation is done
        #self.Qg = nu.array((self.getBottom()-1,int(self.scrImage.getWidth()/2.)))
        
    def getTop(self):
        return self.staffs[0].top
    def getBottom(self):
        return self.staffs[1].bottom

    def addBarPoint(self,xy):
        self.barPoints.append(xy)

    def getLowerLeft(self):
        return self.getSystemPoints()[2]

    def getUpperLeft(self):
        return self.getSystemPoints()[0]

    def getLowerMid(self):
        return self.getSystemPoints()[4]

    def getLowerMidLocal(self):
        return nu.array((self.getSystemHeight()-1,int((self.getSystemWidth()-1)/2)))

    @getter
    def getSystemPoints(self):
        # returns topleft, topright, botleft, botright, and lower hmid
        # of tilted rectangle, such that all above coordinates fall inside the img
        hMid = int(self.scrImage.getWidth()/2.)
        top = self.staffs[0].top
        bot = self.staffs[1].bottom

        dyl = -hMid
        dyr = self.scrImage.getWidth()-hMid

        botLeft = bot+dyl*nu.tan(nu.pi*self.getStaffAngle())
        botRight = bot+dyr*nu.tan(nu.pi*self.getStaffAngle())
        topLeft = top+dyl*nu.tan(nu.pi*self.getStaffAngle())
        topRight = top+dyr*nu.tan(nu.pi*self.getStaffAngle())

        botCorrection = min(0,nu.floor(self.scrImage.getHeight()-max(botLeft,botRight)-1))
        topCorrection = min(0,nu.floor(min(topLeft,topRight))-1)
        print('tcor bcor',topCorrection,botCorrection)
        r = (nu.array((topLeft-topCorrection,0)),
             nu.array((topRight-topCorrection,self.scrImage.getWidth())),
             nu.array((botLeft+botCorrection,0)),
             nu.array((botRight+botCorrection,self.scrImage.getWidth())),
             nu.array((bot+botCorrection,hMid)))
        print('tl,tr,bl,br,bm')
        for p in r:
            print(p)
        return r

    def draw(self):
        for staff in self.staffs:
            staff.draw()
        self.drawBarPoints()

    def drawBarPoints(self):
        lower = int(self.getLowerLeft()[0])
        upper = int(self.getUpperLeft()[0])
        r = lower-upper
        c = nu.array((255,0,0))
        for p in self.barPoints:
            self.scrImage.ap.paintRav(nu.column_stack((nu.arange(upper,lower),(nu.zeros(r)+p[0]).astype(nu.int))),
                                      c)
        
    def getStaffAngle(self):
        return nu.mean([s.getAngle() for s in self.staffs])

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
        systemTopL = self.getRotator().rotate(self.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        systemBotL = self.getRotator().rotate(self.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        bins = 9
        rows = [p for p in selectColumns(self.getHSums(),bins)[0] if systemTopL <= p <= systemBotL] # sounds funny, change name of function
        
        finalStage = False
        k = 0
        ap = AgentPainter(self.getCorrectedImgSegment())
        draw = False
        for i,r in enumerate(rows[:int(.3*len(rows))]):
            agentsnew,died = assignToAgents(self.getCorrectedImgSegment()[r,:],agents,BarAgent,
                                            self.getCorrectedImgSegment().shape[1],vert=r,fixAgents=finalStage)
            if len(agents) > 2:
                agentsnew,d = mergeAgents(agents)
                died.extend(d)
            agents = agentsnew
            print('row',i)
            if len(agents) > 1:
                k = sortBarAgents(agents)
            if draw:
                ap.reset()
                ap.paintHLine(r)
                for a in died:
                    ap.drawAgentGood(a,-300,300)
                    ap.unregister(a)
                for j,a in enumerate(agents):
                    print('{0} {1}'.format(j,a))
                    ap.register(a)
                    ap.drawAgentGood(a,-300,300)
                f0,ext = os.path.splitext(fn)
                print(f0,ext)
                ap.writeImage(f0+'-{0:04d}-r{1}'.format(i,r)+'.png')
        #for i,a in enumerate(agents):
        #    a.points = self.backTransform(a.points)
        #    a.mean = self.backTransform(a.mean)
        #    a.targetAngle -= self.getStaffAngle()
        #    print('{0} {1}'.format(i,a))
        agents = agents[:k]
        draw = True
        if draw:
            ap.reset()
            for a in agents:
                print(a)
                ap.register(a)
                ap.drawAgentGood(a,-300,300)
            f0,ext = os.path.splitext(fn)
            print(f0,ext)
            ap.writeImage(f0+'-final.png')
        for a in agents:
            self.assesBarLine(a)
        return agents

    def assesBarLine(self,agent):
        system0Top = self.getRotator().rotate(self.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system0Bot = self.getRotator().rotate(self.staffs[0].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        system1Top = self.getRotator().rotate(self.staffs[1].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.getRotator().rotate(self.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        # staff 0:
        horz = agent.mean[1]
        l0 = self.getCorrectedImgSegment()[system0Top:system0Bot,horz]
        l1 = self.getCorrectedImgSegment()[system1Top:system1Bot,horz]
        print(agent)
        print(l0)
        print(l1)
        print(l0.shape[0],l1.shape[0],nu.sum(l0)+nu.sum(l1),255*(l0.shape[0]+l1.shape[0]))
        pass
        
    def getSystemWidth(self):
        # this gets cut off from the width, to fit in the page rotated
        cutOff = nu.abs(self.getSystemHeight()*nu.tan(nu.pi*self.getStaffAngle()))
        systemWidth = self.scrImage.getWidth()/nu.cos(nu.pi*self.getStaffAngle()) - 2*cutOff
        systemWidth = int((nu.floor(systemWidth/2.0)-1)*2+1)
        return systemWidth

    def getSystemHeight(self):
        return self.getLowerLeft()[0]-self.getUpperLeft()[0]
        
    @getter
    def getRotator(self):
        return Rotator(self.getStaffAngle(),self.getLowerMid(),self.getLowerMidLocal())

    @getter
    def getCorrectedImgSegment(self):
        halfSystemWidth = int((self.getSystemWidth()-1)/2)
        r = Rotator(self.getStaffAngle(),self.getLowerMid(),self.getLowerMidLocal())
        xx,yy = nu.mgrid[0:self.getSystemHeight(),-halfSystemWidth:halfSystemWidth]
        yy += self.getLowerMidLocal()[1]
        xxr,yyr = r.derotate(xx,yy)

        xf = nu.floor(xxr).astype(nu.int)
        xc = nu.ceil(xxr).astype(nu.int)
        yf = nu.floor(yyr).astype(nu.int)
        yc = nu.ceil(yyr).astype(nu.int)
        wxc = xxr%1
        wxf = 1-wxc
        wyc = yyr%1
        wyf = 1-wyc

        #cimg = self.scrImage.getImg()[nu.round(xxr).astype(nu.int),nu.round(yyr).astype(nu.int)]
        cimg = (((wxf+wyf)*self.scrImage.getImg()[xf,yf] + \
                (wxf+wyc)*self.scrImage.getImg()[xf,yc] + \
                (wxc+wyf)*self.scrImage.getImg()[xc,yf] + \
                (wxc+wyc)*self.scrImage.getImg()[xc,yc])/4.0).astype(nu.uint8)
        
        ap = AgentPainter(cimg)
        ag = self.staffs[0].staffLineAgents[3]
        x,y = r.rotate(ag.points[:,0]+ag.offset,ag.points[:,1])
        ag.points = nu.column_stack((x-ag.offset,y))
        ap.register(ag)
        ap.drawAgentGood(ag)
        ap.writeImage('tst.png')
        return cimg

class Rotator(object):
    def __init__(self,theta,og,ol):
        self.og = og
        self.ol = ol
        self.theta = theta

    def rotate(self,x,y=None):
        if y == None:
            return nu.column_stack(self._rotate(x[:,0],x[:,1]))
        else:
            return self._rotate(x,y)

    def derotate(self,x,y=None):
        if y == None:
            return nu.column_stack(self._derotate(x[:,0],x[:,1]))
        else:
            return self._derotate(x,y)

    def _rotate(self,xx,yy):
        xxr = nu.cos(self.theta*nu.pi)*(xx-self.og[0])-nu.sin(self.theta*nu.pi)*(yy-self.og[1])
        yyr = nu.sin(self.theta*nu.pi)*(xx-self.og[0])+nu.cos(self.theta*nu.pi)*(yy-self.og[1])
        return xxr+self.ol[0],yyr+self.ol[1]

    def _derotate(self,xxr,yyr):
        xx = nu.cos(-self.theta*nu.pi)*(xxr-self.ol[0])-nu.sin(-self.theta*nu.pi)*(yyr-self.ol[1])
        yy = nu.sin(-self.theta*nu.pi)*(xxr-self.ol[0])+nu.cos(-self.theta*nu.pi)*(yyr-self.ol[1])
        return xx+self.og[0],yy+self.og[1]


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
            if i == 2:
                sys.stdout.write('drawing system {0}\n'.format(i))
                sys.stdout.flush()
                system.draw()
                #system.getcorrectedImgSegmentNew()
                sysSegs.append(system.getCorrectedImgSegment())
                #x = nu.array([[50,50],[100,30],[0,0]])
                #sH,sW = system.getCorrectedImgSegment().shape
                #x1 = system.backTransform(nu.column_stack((nu.arange(sH),nu.zeros(sH))))
                #x2 = system.backTransform(nu.column_stack((nu.arange(sH),sW-1+nu.zeros(sH))))
                #x3 = system.backTransform(nu.column_stack((nu.zeros(sW),nu.arange(sW))))
                #x4 = system.backTransform(nu.column_stack((nu.zeros(sW)+sH-1,nu.arange(sW))))
                #self.ap.paintRav(x1,nu.array([255,0,0]))
                #self.ap.paintRav(x2,nu.array([255,0,0]))
                #self.ap.paintRav(x3,nu.array([255,0,0]))
                #self.ap.paintRav(x4,nu.array([255,0,0]))
                #print(xr)
                #nu.savetxt('/tmp/x.txt',x)
                #nu.savetxt('/tmp/xr.txt',xr)
                barAgents = system.getBarLines()
                #for a in barAgents:
                #    self.ap.register(a)
                #    self.ap.drawAgentGood(a,-500,500)
        self.ap.writeImage(self.fn)
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
        ap1 = AgentPainter(ssImg)
        ap1.writeImage(self.fn.replace('.png','-corr.png'))

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
