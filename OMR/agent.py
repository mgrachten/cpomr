#!/usr/bin/env python

import sys,os,logging
from scipy import signal,cluster,spatial
import numpy as nu
from utilities import argpartition, partition


def assignToAgents(v,agents,agentConfig,M,vert=None,horz=None,fixAgents=False,maxWidth=nu.inf):
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
        log = logging.getLogger(__name__)
        log.critical('Need to specify vert or horz')
    unadopted = []
    bids = None
    newagents =[]
    if len(agents) == 0:
        unadopted.extend(range(len(candidates)))
    else:
        bids = nu.zeros((len(candidates),len(agents)))
        for i,c in enumerate(candidates):
            bids[i,:] = nu.array([nu.abs(a.bid(*c)) for a in agents])
        sortedBets = nu.argsort(bids,1)
        cidx = nu.argsort(sortedBets[:,0])
        adopters = set([])
        for i in cidx:
            bestBidder = sortedBets[i,0]
            bestBet = bids[i,bestBidder]
            bidderHas = bestBidder in adopters
            if bestBet <= agents[bestBidder].maxError and not bidderHas:
                agents[bestBidder].award(*candidates[i])
                adopters.add(bestBidder)
                newagents.append(agents[bestBidder])
            else:
                unadopted.append(i)
        newagents.extend([agents[x] for x in set(range(len(agents))).difference(adopters)])
    if not fixAgents:
        for i in unadopted:
            if len(candidates[i]) == 1 or (len(candidates[i]) >1 and (candidates[i][-1][0]-candidates[i][0][0]) <= M/50.):
                # only add an agent if we are on a small section
                newagent = Agent(agentConfig,nu.mean(nu.array(candidates[i]),0))
                newagents.append(newagent)
    
    r = partition(lambda x: x.tick(fixAgents),newagents)
    return r.get(True,[]),r.get(False,[])

def mergeAgents(agents):
    if len(agents) < 3:
        return agents,[]
    newagents = []
    N = len(agents)
    pdist = []
    for i in range(N-1):
        for j in range(i+1,N):
            if agents[i].points.shape[0] < 2 or agents[j].points.shape[0] < 2:
                pdist.append(agents[i].maxError+1)
            else:
                pdist.append(agents[i].mergeable(agents[j]))
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

def tls(X):
    """total least squares for 2 dimensions
    """
    u,s,v = nu.linalg.svd(X)
    return (nu.arctan2(-v[1,1],v[1,0])/nu.pi+1)%1.0

def getError(x,a):
    return nu.sum(nu.dot(x,nu.array([nu.cos(a*nu.pi),-nu.sin(a*nu.pi)]).T)**2)**.5


class AgentConfig(object):
    def __init__(self,targetAngle=0,maxAngleDev=0,maxError=0,minScore=0,offset=0,yoffset=0):
        self.targetAngle = (targetAngle+1.0)%1
        self.maxError = maxError
        self.maxAngleDev = maxAngleDev
        self.minScore = minScore
        self.offset = offset
        self.yoffset = yoffset
        self.aoffset = nu.array((offset,yoffset))

class Agent(object):

    def __init__(self,agentConfig,xy0,xy1=None):
        self._setConfig(agentConfig)
        self.lineWidth = []
        if xy1 != None:
            xy = (xy0+xy1)/2.0
            self._addLineWidth(1+nu.sum((xy0-xy1)**2)**.5)
        else:
            xy = xy0
            self._addLineWidth(1.0)
        self.points = nu.array(xy).reshape((1,2))
        self.mean = xy
        self.angleDev = 0
        self.error = 0
        self.score = 0
        self.adopted = True
        self.age = 0
        self.id = str(self.__hash__())
        self.offspring = 0

    def _setConfig(self,agentConfig):
        self.targetAngle = agentConfig.targetAngle
        self.maxError = agentConfig.maxError
        self.maxAngleDev = agentConfig.maxAngleDev
        self.minScore = agentConfig.minScore
        self.offset = agentConfig.offset
        self.yoffset = agentConfig.yoffset
        self.aoffset = agentConfig.aoffset
          
    def __str__(self):
        return 'Agent: {id}; angle: {angle:0.4f} ({ta:0.4f}+{ad:0.4f}); error: {err:0.3f} age: {age}; npts: {pts}; score: {score}; mean: {mean}'\
            .format(id=self.id,err=self.error,angle=self.angle,ta=self.targetAngle,ad=self.angleDev,
                    age=self.age,pts=self.points.shape[0],score=self.score,mean=self.getDrawMean())
    
    def getLineWidth(self):
        return self.lw

    def getLineWidthStd(self):
        return self.lwstd

    def _addLineWidth(self,w):
        self.lineWidth.append(w)
        self.lw = nu.median(self.lineWidth)
        self.lwstd = nu.std(self.lineWidth)

    @property
    def angle(self):
        return self.targetAngle + self.angleDev

    def getDrawPoints(self):
        return self.points+self.aoffset
    def getDrawMean(self):
        return self.mean+self.aoffset

    def getMiddle(self,M):
        "get Vertical position of agent at the horizontal center of the page of width M" 
        x = self.mean[0]+(M/2.0-self.mean[1])*nu.tan(self.angle*nu.pi)
        return x

    def mergeable(self,other):
        if other.age > 1 and self.age > 1:
            e0 = getError(other.getDrawPoints()-self.getDrawMean(),self.angle)/float(other.points.shape[0])
            e1 = getError(self.getDrawPoints()-other.getDrawMean(),other.angle)/float(self.points.shape[0])
            return (e0+e1)/2.0
        else:
            return self.maxError+1

    def mergeOld(self,other):
        self.points = nu.array(tuple(set([tuple(y) for y in nu.vstack((self.points,other.points))])))
        self.lineWidth = self.lineWidth+other.lineWidth
        self.mean = nu.mean(self.points,0)
        self.angleDev = ((tls(self.points-self.mean)-self.targetAngle)+.5)%1-.5
        self.error = getError(self.points-self.mean,self.angle)/self.points.shape[0]
        self.age = max(self.age,other.age)
        self.score = self.score+other.score

    def merge(self,other):
        
        self.points = nu.array(tuple(set([tuple(y) for y in 
                                          nu.vstack((self.points,other.getDrawPoints()-self.aoffset))])))
        self.lineWidth = self.lineWidth+other.lineWidth
        self.mean = nu.mean(self.points,0)
        self.angleDev = ((tls(self.points-self.mean)-self.targetAngle)+.5)%1-.5
        self.error = getError(self.points-self.mean,self.angle)/self.points.shape[0]
        self.age = max(self.age,other.age)
        self.score = self.score+other.score
        
            
    def tick(self,immortal=False):
        self.offspring = 0
        self.age += 1
        if self.adopted:
            self.score += 1
        else:
            self.score -= 1
        self.adopted = False
        if immortal:
            return True
        else:
            return not self.died()
    
    def died(self):
        angleOK = nu.abs(self.angleDev) <= self.maxAngleDev
        errorOK = self.error <= self.maxError
        successRateOK = self.score >= self.minScore
        r = not all((angleOK,errorOK,successRateOK))
        #if r:
        #    print('Died: {0}; angleOK: {1}; errorOK: {2}, scoreOK: {3}'.format(self,angleOK,errorOK,successRateOK))
        return r
  
    def getIntersection(self,xy0,xy1):
        # NB: x and y coordinates are in reverse order
        # because of image conventions: x = vertical, y = horizontal
        dy,dx = nu.array(xy0-xy1,nu.float)
        # slope of new line
        slope = dy/dx
        ytilde,xtilde = xy0-self.mean
        # offset of new line
        b = (ytilde-slope*xtilde)
        # special case 1: parallel lines
        if slope == nu.tan(nu.pi*self.angle):
            log = logging.getLogger(__name__)
            log.info('Parallel lines detected')
            return None

        # special case 2: vertical line
        if nu.isinf(slope):
            # x constant
            x = xy0[1]
            y = nu.tan(nu.pi*self.angle)*(x - self.mean[1])+self.mean[0]
            return nu.array((y,x))
        # special case 3: line undefined
        if nu.isnan(slope):
            # undefined slope, constant y
            return None

        x = b/(nu.tan(nu.pi*self.angle)-slope)
        r =  nu.array((slope*x+b,x))+self.mean
        return r

    def _getAngleDistance(self,a):
        return (a-self.targetAngle+.5)%1-.5

    def _getAngle(self,xy):
        return ((nu.arctan2(*((xy-self.mean)))/nu.pi)+1)%1

    def _getClosestAngle(self,a):
        return nu.sort([(a+1)%1,self.angle-self.maxAngleDev,
                        self.angle+self.maxAngleDev])[1]

    def preparePointAdd(self,xy0,xy1=None):
        if xy1 != None:
            error0 = nu.dot(xy0-self.mean,nu.array([nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]))
            error1 = nu.dot(xy1-self.mean,nu.array([nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]))
            lw = 1+nu.sum((xy0-xy1)**2)**.5
            acceptableWidth = lw <= self.getLineWidth() + max(1,self.getLineWidthStd())
            if nu.sign(error0) != nu.sign(error1) and not acceptableWidth:
                xy = self.getIntersection(xy0,xy1)
            else:
                if acceptableWidth: #lw <= self.getLineWidth() + max(1,self.getLineWidthStd()):
                    # it's most probably a pure segment of the line, store the mean
                    xy = (xy0+xy1)/2.0
                else:
                    # too thick to be a line, store the point that has smallest error
                    xy = (xy0,xy1)[nu.argmin(nu.abs([error0,error1]))]
        else:
            lw = 1.0
            xy = xy0
        points = nu.vstack((self.points,xy))
        mean = nu.mean(points,0)
        tlsr = tls(points-mean)
        angleDev = ((tlsr-self.targetAngle)+.5)%1-.5
        error = getError(points-mean,angleDev+self.targetAngle)/points.shape[0]
        return error,angleDev,mean,lw,points

    def bid(self,xy0,xy1=None):
        # distance of xy0 to the current line (defined by self.angle)
        if self.points.shape[0] == 1:
            # we have no empirical angle yet
            # find the optimal angle (within maxAngleDev), and give the error respective to that
            if xy1 == None:
                xyp1 = xy0
            else:
                xyp1 = xy1
            aa0 = self._getAngleDistance(self._getAngle(xy0))
            aa1 = self._getAngleDistance(self._getAngle(xyp1))
            angle = nu.sort([self.targetAngle+aa0,self.targetAngle+aa1,self.angle])[1]
        else:
            angle = self.angle
        if nu.abs(self._getAngleDistance(angle)) > self.maxAngleDev:
            return self.maxError+1
        # print('adjusting angle:',self.angle,'to',angle)
        anglePen = nu.abs(self.angle-angle)
        error0 = nu.dot((xy0-self.mean),nu.array([nu.cos(angle*nu.pi),-nu.sin(angle*nu.pi)]))
        if xy1 == None:
            return nu.abs(error0)+anglePen
        error1 = nu.dot(xy1-self.mean,nu.array([nu.cos(angle*nu.pi),-nu.sin(angle*nu.pi)]))
        if nu.sign(error0) != nu.sign(error1):
            return 0.0+anglePen
        else:
            return min(nu.abs(error0),nu.abs(error1))+anglePen

    def award(self,xy0,xy1=None):
        self.adopted = True
        
        self.error,self.angleDev,self.mean,lw,self.points = self.preparePointAdd(xy0,xy1=xy1)
        self._addLineWidth(lw)


if __name__ == '__main__':
    pass
