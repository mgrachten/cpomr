#!/usr/bin/env python

import sys,os
from scipy import signal,cluster,spatial
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition, makeColors
from main import convolve

from agenttest import Agent

class AgentPainter(object):
    def __init__(self,img):
        self.img = nu.array((255-img,255-img,255-img))
        self.maxAgents = 100
        self.colors = makeColors(self.maxAgents)
        self.paintSlots = nu.zeros(self.maxAgents,nu.bool)
        self.agents = {}

    def writeImage(self,fn):
        #print(nu.min(img),nu.max(img))
        self.img = self.img.astype(nu.uint8)
        fn = os.path.join('/tmp',os.path.splitext(os.path.basename(fn))[0]+'.png')
        print(fn)
        writeImageData(fn,self.img.shape[1:],self.img[0,:,:],self.img[1,:,:],self.img[2,:,:])

    def register(self,agent):
        available = nu.where(self.paintSlots==0)[0]
        if len(available) < 1:
            print('no paint slots available')
            return False
        self.agents[agent] = available[0]
        self.paintSlots[available[0]] = True
        #self.paintStart(agent.point,self.colors[self.agents[agent]])

    def unregister(self,agent):
        if self.agents.has_key(agent):
            self.paintSlots[self.agents[agent]] = False
            del self.agents[agent]
        else:
            sys.stderr.write('Warning, unknown agent')
        
    def drawAgent(self,agent):
        if self.agents.has_key(agent):
            c = self.colors[self.agents[agent]]
            c1 = nu.maximum(0,c-50)
            #self.paintStart(agent.point,c)
            #print(agent.getTrajectory())
            for p in agent.getTrajectory():
                self.paint(p,c)
            for p in agent.points:
                #for p in pp:
                self.paint(p,c1)
                self.paint((p[0]-1,p[1]),c1)
                self.paint((p[0]+1,p[1]),c1)
        else:
            sys.stderr.write('Warning, unknown agent')

    def paint(self,coord,color):
        #print('point',coord,img.shape)
        self.img[:,int(coord[0]),int(coord[1])] = color


        
def getCrossings(v,agents,AgentType,ap,vert=None,horz=None):
    mindist = 1
    data = nu.nonzero(v)[0]
    if len(data) > 1:
        candidates = [tuple(x) if len(x)==1 else (x[0],x[-1]) for x in nu.split(data,nu.nonzero(nu.diff(data)>1)[0]+1)]
    elif len(data) == 1:
        candidates = [tuple(data)]
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
    bids = None
    if len(agents) == 0:
        unadopted.extend(range(len(candidates)))
    else:
        print('agents, candidates',len(agents),len(candidates))
        bids = nu.zeros((len(candidates),len(agents)))
        for i,c in enumerate(candidates):
            bids[i,:] = [nu.abs(a.bid(*c)) for a in agents]

        cidx = nu.argsort(nu.min(bids,1))
        #print(bids)
        #print(cidx)
        adopters = set([])
        for i in cidx:
            #print(bids[i,:])
            bestBidder = nu.argmin(bids[i,:])
            bestBet = bids[i,bestBidder]
            if bestBet <= agents[bestBidder].maxError and not bestBidder in adopters:
                agents[bestBidder].award(*candidates[i])
                adopters.add(bestBidder)
            else:
                unadopted.append(i)
    for i in unadopted:
        newagent = AgentType(nu.mean(nu.array(candidates[i]),0))
        #ap.register(newagent)
        agents.append(newagent)
        #for j in candidates[i]:
            #print('unadopted',j)
    
    return [a for a in agents if a.tick()]

class StaffLineAgent(Agent):
    targetAngle = 0 # in degrees
    maxError = 5 # mean perpendicular distance of points to line (in pixels)
    maxAngleDev = 2 # in degrees
    minScore = -7

def findStaffLines(img,fn):
    N,M = img.shape
    lookatProportion = .05
    vsums = nu.sum(img,0)
    #vsums = vsums[nu.nonzero(vsums)[0]]
    colorder = nu.argsort(vsums)
    colorder = colorder[nu.nonzero(vsums[colorder])]

    #print(vsums[colorder[:10]])
    #print('first nonz:',nu.where(vsums[colorder]>0))

    agents = []
    
    ap = AgentPainter(img)

    for c in colorder[:int(lookatProportion*N)]:
    #for c in colorder[:100]:
        print('column',c)
        agents = getCrossings(img[:,c],agents,StaffLineAgent,ap,horz=c)
        #print('agents',len(agents))
        #agents = agents[:maxStaffLines]
        #agents.sort(key=lambda x: x.age*x.score)
        agents.sort(key=lambda x: x.evaluate())
        agents.reverse()
        #for i,a in enumerate(agents[:50]):
        #    print('{0} {1}'.format(i,a))
    
    k = len(agents)-1
    #k = 30
    #print(agents[k])
    j = 0
    print('columns processed',lookatProportion*N)
    for a in agents[0:k+1]:
        if a.points.shape[0] > 1 and a.age > .8*lookatProportion*N:
        #if a.points.shape[0] > 1 and a.age > 60:
            print('{0} {1}'.format(j,a))
            j += 1
            ap.register(a)
            ap.drawAgent(a)
    ap.writeImage(fn)

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
    bgThreshold = 100
    img[img< bgThreshold] = 0
    #img[img> 0] = 255
    #findStaffLines(img[:1000,:],fn)
    findStaffLines(img,fn)
    #findStaffLines(img[:100,:],fn)
    #nu.savetxt('/tmp/p.txt',nu.sum(img,1))
