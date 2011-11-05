#!/usr/bin/env python

import sys,os
from scipy import signal,cluster,spatial
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition, makeColors
from main import convolve
from agent import Agent, AgentPainter


def getCrossings(v,oldagents,AgentType,ap,M,vert=None,horz=None):
    agents = oldagents[:]
    mindist = 1
    data = nu.nonzero(v)[0]
    if len(data) > 1:
        candidates = [tuple(x) if len(x)==1 else (x[0],x[-1]) for x in nu.split(data,nu.nonzero(nu.diff(data)>1)[0]+1)]
    elif len(data) == 1:
        candidates = [tuple(data)]
    else:
        return agents

    #print('protocandidates',candidates)
    if vert is not None:
        candidates = [[nu.array([vert,horz]) for horz in horzz] for horzz in candidates]
    elif horz is not None:
        candidates = [[nu.array([vert,horz]) for vert in vertz] for vertz in candidates]
    else:
        print('error, need to specify vert or horz')

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
            bestBidder = nu.argmin(bids[i,:])
            bestBet = bids[i,bestBidder]
            if bestBet <= agents[bestBidder].maxError and not bestBidder in adopters:
                agents[bestBidder].award(*candidates[i])
                adopters.add(bestBidder)
            else:
                unadopted.append(i)
    for i in unadopted:
        if len(candidates[i]) >1 and (candidates[i][-1][0]-candidates[i][0][0]) <= M/722.:
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
    bins = 9
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
    agents.sort(key=lambda x: -x.score)
    N = len(agents)
    scores = nu.array([a.score for a in agents])
    scoreIdx = nu.argsort(-scores)
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

def fitLines(agents,N,M):
    print(N,M,M/2)
    meansAngles = nu.array([(a.mean[0],a.mean[1],((a.angle-a.targetAngle+.5)%1-.5)*nu.pi) for a in agents])
    x = meansAngles[:,0]+(M/2-meansAngles[:,1])*nu.tan(meansAngles[:,2]*nu.pi)
    x = nu.sort(x)
    #print(nu.column_stack((meansAngles,x)))
    nu.savetxt('/tmp/x.txt',x)
    nidx = nu.array([nu.argsort(nu.abs(x-xi))[1:] for xi in x])
    nndist = nu.array([(xi-x[nu.argsort(nu.abs(x-xi))])[1:] for xi in x])
    n0dist = nndist[:,1]
    d0 = nu.median(nu.abs(n0dist))
    nu.savetxt('/tmp/n0.txt',n0dist)
    hits = []
    for j,(ii,bm) in enumerate([(i,nu.abs(nndist[i,:]) < (nu.abs(n0dist[i])*2.5)) for i in range(len(n0dist))]):
        l = nndist[ii,bm]
        print(x[j],n0dist[j],l)
        #print([(k/n0dist[j]+.5)%1-.5 for k in l])
        y = nu.array([((nu.round(k/n0dist[j]))/1,(k/n0dist[j]+.5)%1-.5) for k in l])
        y = y[nu.argsort(nu.abs(y[:,1]))]
        print(y)
        error = 0
        missing = 0
        partners = []
        u = 0
        for m in (-2,-1,1,2):
            z = nu.where(y[:,0] == m)[0]
            if len(z) == 0:
                missing += 1
            else:
                partners.append(nidx[j,bm][u])
                u +=1
                error += nu.abs(y[z[0],1])
        error = error/(4-missing)
        #e1 = 1+(nu.abs(n0dist[j])-d0)**2
        e1 = (1+nu.abs(1-nu.abs(n0dist[j])/d0))**2
        e2 = (1+error)**2
        e3 = 1 if missing < 2 else 2
        print('e',error)
        etot = (3.0/(e1+e2+e3))
        print('p',partners,len(x))
        pp = [x[j]]+[x[q] for q in partners]
        pp.sort()
        print('p',pp)
        #print('p',x[bm][tuple(partners)])
        #print('p',[x[p] for p in partners])
        #print('p',[x[p] for p in partners])
        print('e1,e2,e3',e1,e2,e3,etot)
        #hits.append(.5 if etot >.5 else 0)
        hits.append(etot)
    nu.savetxt('/tmp/h.txt',nu.column_stack((x,nu.array(hits))))

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
    thr = l0/20.
    if True:
        return nu.std(dxs[checkIdx]) < thr

    print(len(dxs),len(checkIdx))
    #nu.savetxt('/tmp/x.txt',dxs)
    print(xs)
    print(nu.arange(5,len(agents),5)-1)
    print(nu.nonzero(checkIdx)[0])
    print(dxs[checkIdx])
    print(nu.std(dxs[checkIdx]))
    #nu.savetxt('/tmp/x.txt',xs)
    nu.savetxt('/tmp/x.txt',nu.column_stack((xs[:-1],checkIdx,dxs)))
    return nu.std(dxs[checkIdx]) < thr
    #print(xx)

def findStaffLines(img,fn):
    N,M = img.shape
    lookatProportion = 1
    vsums = nu.sum(img,0)

    columns = selectColumns(vsums,lookatProportion)
    agents = []
    
    ap = AgentPainter(img)
    draw = True
    #draw = False
    taprev= []
    for i,c in enumerate(columns):
        print('column',i)
        agentsnew = getCrossings(img[:,c],agents,StaffLineAgent,ap,M,horz=c)

        if len(agents)> 1:
            agentsnew = mergeAgents(agentsnew)

    
        if len(agents)> 40:
            #fitLines(agentsnew,N,M)
            ta = sortAgents(agentsnew)
            r = assessLines(ta,N,M,i==27)
            if r:
                ap.reset()
                agents = ta#agentsnew[:]
                break
        else:
            ta = agentsnew[:]

        if draw:
            sagentsnew = set(ta)
            setagents = set(taprev)
            born = sagentsnew.difference(setagents)
            died = setagents.difference(sagentsnew)
            ap.reset()
            for a in born:
                ap.register(a)
            for a in died:
                ap.unregister(a)
            for a in ta:
                ap.drawAgent(a)
            print('drew agents',len(ta))
            ap.paintVLine(c)
            f0,ext = os.path.splitext(fn)
            print(f0,ext)
            #ap.writeImage(fn.replace('.png','-{0:04d}-c{1}.png'.format(i,c)))
            ap.writeImage(f0+'-{0:04d}-c{1}'.format(i,c)+'.png')
        
        # DON'T DELETE
        agents = agentsnew[:]
        taprev = ta[:]
    #nu.savetxt('/tmp/a.txt',nu.array([a.toVector() for a in agents]))
    #for i,a in enumerate(agents):
    #    print('{0} {1}'.format(i,a))
    #    nu.savetxt('/tmp/a{0:02d}.txt'.format(i),a.getScorehist())
    print('columns processed',int(lookatProportion*N))
    #agents = sortAgents(agents)
    j = 0
    for a in agents:
        if a.points.shape[0] > 1:
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
    #findStaffLines(img[1500:,:],fn)
    findStaffLines(img,fn)
