#!/usr/bin/env python

import sys
import numpy as nu
from utilities import getter

def sortStaffLineAgents(agents,k=10):
    agents.sort(key=lambda x: -x.score)
    scores = nu.array([a.score for a in agents]+[0]*5)
    N = len(scores)
    if nu.min(scores) == nu.max(scores):
        return agents

    meanScorePerSystem = [nu.mean(scores[i:i+k]) for i in range(0,N,k)]
    print(meanScorePerSystem)
    dms = nu.diff(meanScorePerSystem)
    nsystems = nu.argmin(dms)+1
    print('estimating ',nsystems,'groups,',k*nsystems,'stafflines')
    na = agents[:k*nsystems]
    print('keeping',len(na),'agents')
    return na

def assessStaffLineAgents(iagents,M,nPerStaff):
    agents = sortStaffLineAgents(iagents,nPerStaff)
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

class Staff(object):
    def __init__(self,scoreImage,staffLineAgents,top,bottom):
        self.scrImage = scoreImage
        self.staffLineAgents = staffLineAgents
        self.top = top
        self.bottom = bottom
        self.staffLineAgents.sort(key=lambda x: x.getMiddle(self.scrImage.getWidth()))
    def __str__(self):
        return 'Staff {0}; nAgents: {1}; avggap: {2}: top: {3}; bot: {4}'\
            .format(self.__hash__(),len(self.staffLineAgents),self.getStaffLineDistance(),
                    self.top,self.bottom)
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
    def getStaffLineDistances(self):
        return nu.diff([a.getMiddle(self.scrImage.getWidth()) for a in self.staffLineAgents])

    @getter
    def getStaffLineDistance(self):
        return nu.mean(self.getStaffLineDistances())

    @getter
    def getStaffLineDistanceStd(self):
        return nu.std(self.getStaffLineDistances())

if __name__ == '__main__':
    pass
