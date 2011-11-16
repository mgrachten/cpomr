#!/usr/bin/env python

import sys,os,pickle
from scipy import signal,cluster,spatial
from scipy.stats import distributions
import numpy as nu
from imageUtil import writeImageData, getPattern, findValleys, smooth, normalize
from utilities import argpartition, partition, makeColors
from main import convolve
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
    print('candidates',candidates)
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
        print('agents, candidates',len(agents),len(candidates))

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
                track = False #2807 < agents[i].mean[0] < 2812 and 2807 < agents[j].mean[0] < 2812
                if track:
                    print('merge')
                    print(agents[i])
                    print(agents[j])
                    print('cangle',cAngle)
                    print(((cAngle-agents[i].targetAngle+.5)%1-.5),agents[i].maxAngleDev)
                    print(((cAngle-agents[i].targetAngle+.5)%1-.5),agents[i].maxAngleDev)
                if True: #((cAngle-agents[i].targetAngle+.5)%1-.5) < agents[i].maxAngleDev:
                #if nu.abs(cAngle-agents[i].targetAngle) < agents[i].maxAngleDev:
                    # yes, do further check
                    if track:
                        print('passed angle check, continuing')
                        print('max error',agents[i].maxError)
                    pdist.append(agents[i].mergeable(agents[j],track))
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
                track = False#2807 < a.mean[0] < 2812 and 2807 < agents[i].mean[0] < 2812
                a.merge(agents[i],track)
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

def assessLines(agents,N,M,show=False):
    # check if 4 nn are equidistant for each line
    print(len(agents))
    meansAngles = nu.array([(a.mean[0],a.mean[1],((a.angle-a.targetAngle+.5)%1-.5)*nu.pi) for a in agents])
    x = meansAngles[:,0]+(M/2-meansAngles[:,1])*nu.tan(meansAngles[:,2]*nu.pi)
    xs = nu.sort(x)
    dxs = nu.diff(xs)
    l0 = nu.median(dxs)
    checkIdx = nu.ones(len(agents)-1,nu.bool)
    checkIdx[nu.arange(5,len(agents),5)-1] = False
    #thr = l0/20.
    thr = l0/10.
    return nu.std(dxs[checkIdx]) < thr

def findStaffLines(img,fn):
    N,M = img.shape
    
    bins = 9
    vsums = nu.sum(img,0)

    columns,colBins = selectColumns(vsums,bins)

    splitParts = 5
    try:
        K = int(N/splitParts)
    except ZeroDivisionError:
        K = 1
    agents = []
    seenCols = set([])
    for k in range(splitParts):
        t = max(0,k*K)
        b = min(N,(k+1)*K)
        pagents,doneCols = findStaffLinesInPart(img,t,b,StaffLineAgent,bins)
        seenCols = set.union(seenCols,doneCols)
        for a in pagents:
            a.mean[0] += t
            a.points[:,0] += t
        agents.extend(pagents)
    #agents = mergeAgents(agents)    

    draw = True
    #draw = False
    if draw:
        ap = AgentPainter(img)
        for a in agents:
            ap.register(a)

    #taprev= []

    for i,c in enumerate(columns):
        if False: #c in seenCols:
            continue
        print('column',i)
        agentsnew = getCrossings(img[:,c],agents,StaffLineAgent,M,horz=c)

        if len(agentsnew)> 1:
            agentsnew = mergeAgents(agentsnew)

        if i > 50 and len(agentsnew)> 10:
            agentsnew = sortAgents(agentsnew[:])
            print('agents')
            for j,a in enumerate(sorted(sorted(agentsnew,key=lambda x: x.mean[0]),key=lambda x: -x.score)):
                print(j),
                print(a)
            r = assessLines(agentsnew,N,M,i==27)
            #r = True
            if r:
                if draw:
                    ap.reset()
                agents = agentsnew[:]
                break
        else:
            print('agents')
            for j,a in enumerate(sorted(sorted(agentsnew,key=lambda x: x.mean[0]),key=lambda x: -x.score)):
                print(j),
                print(a)

        if draw:
            sagentsnew = set(agentsnew)
            setagents = set(agents)
            born = sagentsnew.difference(setagents)
            died = setagents.difference(sagentsnew)
            ap.reset()
            for a in born:
                ap.register(a)
            for a in died:
                ap.unregister(a)
            for a in agentsnew:
                ap.drawAgentGood(a,-3000,3000)
            print('drew agents',len(agentsnew))
            ap.paintVLine(c)
            f0,ext = os.path.splitext(fn)
            print(f0,ext)
            #ap.writeImage(fn.replace('.png','-{0:04d}-c{1}.png'.format(i,c)))
            ap.writeImage(f0+'-{0:04d}-c{1}'.format(i,c)+'.png')
        
        # DON'T DELETE
        agents = agentsnew[:]

    agents = finalizeAgents(agents,img,bins,fn,ap if draw else None)
    if not draw:
        aa = [agents[k*10:(k+1)*10] for k in range(len(agents)/10)]
        for x,ab in enumerate(aa):
            print('staff',x)
            for a in ab:
                print(a.mean)
        return aa

    j = 0
    for a in agents:
        if a.points.shape[0] > 1:
            print('{0} {1}'.format(j,a))
            j += 1
            ap.register(a)
            ap.drawAgentGood(a)
    ap.writeImage(fn)
    return [agents[k*10:(k+1)*10] for k in range(len(agents)/10)]


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

# def setter(f):
#     "Stores the value for later use"
#     def _setter(self,*args,**kwargs):
#         if not hasattr(self,'valuedict'):
#             self.valuedict = {}
#         self.valuedict[f] = f(self,*args,**kwargs)
#     return _setter

#def getBestSubset(agents,k=5):
#    k
    
def assessStaffLineAgents(iagents,M):
    agents = sortAgents(iagents,5)
    for a in agents:
        print(a)
    meansAngles = nu.array([(a.mean[0],a.mean[1],a.getAngle()) for a in agents])
    x = meansAngles[:,0]+(M/2-meansAngles[:,1])*nu.tan(meansAngles[:,2]*nu.pi)
    xs = nu.sort(x)
    dxs = nu.diff(xs)
    l0 = nu.median(dxs)
    checkIdx = nu.ones(len(agents)-1,nu.bool)
    checkIdx[nu.arange(5,len(agents),5)-1] = False
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


    def getStats(self):
        vsum = len(nu.nonzero(self.getVSums())[0])
        hsum = len(nu.nonzero(self.getHSums())[0])
        height = (self.bottom-self.top)
        print('vsum',vsum,
              'hsum',hsum,
              'h',height)
        return vsum, hsum, height

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
                finalStage,selection = assessStaffLineAgents(agents,self.scrImage.getWidth())
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
        return agents

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

def selectMajority(stats,N,M):
    meds = nu.median(stats[:,:-1],0)
    #ref = nu.max(stats[:,:-1],0)-nu.min(stats[:,:-1],0)
    ref = nu.array((N/stats.shape[0],M))
    nstats = stats[:,:-1]-meds
    nstats[nstats>0] = 0
    dref = nu.sum(ref**2)**.5
    ndists = (nu.sum(nstats**2,1)**.5)/dref
    snstats = nu.column_stack((nstats,stats[:,-1]))
    snstats = nu.vstack((snstats,nu.append(-ref,-1)))
    nu.savetxt('/tmp/ns.txt',snstats,fmt='%d')
    return nu.nonzero(nu.logical_not(ndists < .2))[0]

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

    def drawImage(self):
        stats = []
        for i,vs in enumerate(self.getVSegments()):
            self.ap.paintHLine(vs.bottom,step=2)
            stats.append(list(vs.getStats())+[i])
        stats = nu.array(stats,nu.int)
        print(stats)
        stats = stats[:,(0,1,3)]
        nu.savetxt('/tmp/s.txt',stats,fmt='%d')

        exclude = selectMajority(stats,self.getHeight(),self.getWidth())
        groups = []
        for i,vs in enumerate(self.getVSegments()):
            if i in exclude:
                for j in range(vs.top,vs.bottom,4):
                    self.ap.paintHLine(j,alpha=0.9,step=4)
            else:
                groups.append(vs.getStaffLines())

        for i,g in enumerate(groups):
            print('group',i)
            for a in g:
                self.ap.register(a)
                print(a)
                self.ap.drawAgentGood(a,-2600,2600)
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
        return vsegments

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

    #img[img>= bgThreshold] = 255
    #findStaffLines(img[1500:,:],fn)

    si = ScoreImage(fn)
    si.drawImage()

    sys.exit()
    groups = []
    for vs in si.getVSegments():
        #print(180*vs.getAngle())
        groups.append(vs.getStaffLines())
    for i,g in enumerate(groups):
        print('group',i)
        for a in g:
            print(a)

    #print(si.getHSums())
    #print(si.getHSums())
    #print(si.getHSums())
    #si.estimateSegmentAngles()
    #si.getVSegments()[0].getStaffLines()
    sys.exit()

    agentfn = os.path.join('/tmp/',os.path.splitext(os.path.basename(fn))[0]+'.agents')
    agents = findStaffLines(img,fn)
    with open(agentfn,'w') as f:
        pickle.dump(agents,f)
