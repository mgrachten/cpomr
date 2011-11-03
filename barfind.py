#!/usr/bin/env python

import sys,os
from scipy import signal,cluster,spatial
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition, makeColors
from main import convolve
from agent import Agent


class AgentPainter(object):
    def __init__(self,img):
        self.img = nu.array((255-img,255-img,255-img))
        self.imgOrig = nu.array((255-img,255-img,255-img))
        self.maxAgents = 300
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
            sys.stderr.write('Warning, unknown agent\n')
        
    def reset(self):
        self.img = self.imgOrig.copy()

    def drawAgent(self,agent):
        if self.agents.has_key(agent):
            c = self.colors[self.agents[agent]]
            c1 = nu.minimum(255,c+50)
            c2 = nu.maximum(0,c-100)
            #self.paintStart(agent.point,c)
            #print(agent.getTrajectory())
            #if agent.points.shape[0] > 1:
            #    for p in agent.getTrajectory():
            #        self.paint(p,c,.3)
            #a.mean-nu.arange(0,self.img.shape[1])
            N = self.img.shape[2]
            xbegin = agent.mean[0]+(-agent.mean[1])*nu.tan(((agent.angle-agent.targetAngle+.5)%1-.5)*nu.pi)
            xend = agent.mean[0]+(N-agent.mean[1])*nu.tan(((agent.angle-agent.targetAngle+.5)%1-.5)*nu.pi)
            
            dx = nu.abs(xend-xbegin)
            dy = N
            delta = max(dx,dy)+2
            z = nu.round(nu.column_stack((nu.linspace(xbegin,xend,delta),
                                          nu.linspace(0,N-1,delta)))).astype(nu.int)
            for p in z:
                try:
                    alpha = min(1,max(.2,.5+float(agent.score)/agent.age))
                    self.paint(p,c2,alpha)
                except IndexError:
                    print(p,img.shape[1:])
                    
            #print(img.shape)
            #print('draw',agent.angle,agent.mean)
            #print(z)
            #print(agent.points)
            self.paintRect(agent.points[0][0],agent.points[0][0],
                           agent.points[0][1],agent.points[0][1],c)
            for p in agent.points:
                #for p in pp:
                self.paint(p,c1)
                if p[0]-1 >= 0:
                    self.paint((p[0]-1,p[1]),c1)
                if p[0]+1 < self.img.shape[0]:
                    self.paint((p[0]+1,p[1]),c1)
        else:
            sys.stderr.write('Warning, unknown agent')

    def paint(self,coord,color,alpha=1):
        #print('point',coord,img.shape)
        self.img[:,int(coord[0]),int(coord[1])] = (1-alpha)*self.img[:,int(coord[0]),int(coord[1])]+alpha*color

    def paintVLine(self,y,alpha=.5):
        self.img[:,:,y] = (1-alpha)*self.img[:,:,y]+alpha*0

    def paintRect(self,xmin,xmax,ymin,ymax,color,alpha=.5):
        rectSize = 10
        N,M = self.img.shape[1:]
        t = int(max(0,xmin-nu.floor(rectSize/2.)))
        b = int(min(N-1,xmax+nu.floor(rectSize/2.)))
        l = int(max(0,ymin-nu.floor(rectSize/2.)))
        r = int(min(M-1,ymax+nu.floor(rectSize/2.)))
        #self.img[:,:,int(ymin)] = (1-alpha)*self.img[:,:,int(ymin)]+alpha*0
        #self.img[:,int(xmin),:] = (1-alpha)*self.img[:,int(xmin),:]+alpha*0
        for i,c in enumerate(color):
            self.img[i,t:b,l] = c
            self.img[i,t:b,r] = c
            self.img[i,t,l:r] = c
            self.img[i,b,l:r+1] = c

def getCrossings(v,oldagents,AgentType,ap,N,vert=None,horz=None):
    agents = oldagents[:]
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
        if len(candidates[i]) > 1 and (candidates[i][-1][0]-candidates[i][0][0]) <= N/1000.:
            # only add an agent if we are on a small section
            newagent = AgentType(nu.mean(nu.array(candidates[i]),0))
            agents.append(newagent)
    
    return [a for a in agents if a.tick()]

class StaffLineAgent(Agent):
    targetAngle = 0 # in degrees
    maxError = 10 # mean perpendicular distance of points to line (in pixels)
    maxAngleDev = 2 # in degrees
    minScore = -5

def selectColumns(vsums,lookatProportion):
    N = len(vsums)
    bins = 5
    nzidx = nu.nonzero(vsums)[0]
    binSize = int(nu.floor(len(nzidx)/bins))
    idxm = nzidx[:bins*binSize].reshape((bins,binSize))
    for i in range(bins):
        idxm[i,:] = idxm[i,nu.argsort(vsums[idxm[i,:]])]
    columns = idxm.T.ravel()
    columns = nu.append(columns,nzidx[bins*binSize:])
    return columns[:int(N*lookatProportion)]

def mergeAgents(agents):
    newagents = []
    N = len(agents)
    pdist = []
    for i in range(N-1):
        for j in range(i+1,N):
            if agents[i].points.shape[0] < 2 or agents[j].points.shape[0] < 2:
                pdist.append(agents[i].maxError+1)
            else:
                cAngle = (nu.arctan2(*(agents[i].mean-agents[j].mean))%nu.pi)/nu.pi
                # fast check: are means in positions likely for merge?
                if ((cAngle-agents[i].targetAngle+.5)%1-.5) < agents[i].maxAngleDev/360.:
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


def sortAgents(agents):
    #agents = [a for a in agents if not all((a.error == 0,a.angle==a.targetAngle,a.points.shape[0]<10))]
    N = len(agents)
    #agents = nu.array(agentsl)
    scores = nu.array([a.score for a in agents])
    scoreIdx = nu.argsort(scores)[::-1]
    scores = scores[scoreIdx]
    #angles = nu.array([(a.angle-a.targetAngle+.5)%1-.5 for a in agents])[scoreIdx]
    k = 10
    meanScorePerSystem = [nu.mean(scores[i:i+k]) for i in range(0,N,k)]
    #meanAnglePerSystem = [nu.mean(angles[i:i+k]) for i in range(0,N,k)]
    dms = nu.diff(meanScorePerSystem)
    nsystems = nu.argmin(dms)+1
    print('estimating ',nsystems,'systems,',k*nsystems,'stafflines')
    na = agents[:k*nsystems]
    print('keeping',len(na),'agents')
    return na


def findPredominantNNdist(agents,K=10):
    # NOT FINISHED
    assert len(agents) > 10
    xx = nu.array([a.mean[0] for a in agents[:K]])+1
    for i in range(K):
        dists = nu.abs(xx-xx[i])
        nn = xx[nu.argsort(dists)[:10]]
        assert nn[0] > 0
        print('agent',i,nu.abs((nn/nn[0]+.5)%1-.5))

def findStaffLines(img,fn):
    N,M = img.shape
    lookatProportion = .04
    vsums = nu.sum(img,0)
    columns = selectColumns(vsums,lookatProportion)
    agents = []
    ap = AgentPainter(img)
    for i,c in enumerate(columns):
        #print('column',c)
        agentsnew = getCrossings(img[:,c],agents,StaffLineAgent,ap,N,horz=c)
        #agentsnew.sort(key=lambda x: -x.score)
        if len(agents)> 1:
            agentsnew = mergeAgents(agentsnew)

        draw = False
        if draw:
            born = set(agentsnew).difference(set(agents))
            died = set(agents).difference(set(agentsnew))
            ap.reset()
            for a in born:
                ap.register(a)
            for a in died:
                ap.unregister(a)
            for a in agents:
                ap.drawAgent(a)
            ap.paintVLine(c)
            ap.writeImage(fn.replace('.png','-{0:04d}-c{1}.png'.format(i,c)))
        
        # don't delete!
        agents = agentsnew

    #agents.sort(key=lambda x: -x.score)
    #nu.savetxt('/tmp/a.txt',nu.array([a.toVector() for a in agents]))
    #for i,a in enumerate(agents):
    #    print('{0} {1}'.format(i,a))
    #    nu.savetxt('/tmp/a{0:02d}.txt'.format(i),a.getScorehist())
    print('columns processed',int(lookatProportion*N))
    agents = sortAgents(agents)
    for a in agents:
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
    findStaffLines(img,fn)
