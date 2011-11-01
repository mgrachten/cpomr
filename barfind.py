#!/usr/bin/env python

import sys,os
from scipy import signal,cluster,spatial
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition, makeColors
from main import convolve

def tls(X):
    """total least squares for 2 dimensions
    """
    u,s,v = nu.linalg.svd(X)
    return nu.arctan2(v[0,1],v[1,1])/(2*nu.pi)

class AgentPainter(object):
    def __init__(self,img):
        self.img = nu.array((255-img,255-img,255-img))
        self.maxAgents = 50
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

class Agent(object):
    targetAngleDeg = 0
    minScore = 0
    def __init__(self,xy):
        self.points = nu.array(xy).reshape((1,2))
        self.angle = self.targetAngleDeg
        self.score = 0
        self.adopted = True
        self.age = 0
        self.nadopted = 0
    def __str__old(self):
        return 'Agent: point: {0}; angle: {1}; score: {2} age: {3}; points: {4}'.format(self.point,self.angle,self.score,self.age)
    def __str__(self):
        return 'Agent: angle: {3}; score: {0} age: {1}; points: {2}'.format(self.score,self.age,len(self.points),self.angle)
    def tick(self):
        self.age += 1
        if self.adopted:
            self.score += 1
        else:
            self.score -= 1
        self.adopted = False
        return not self.died()
    
    def award(self,xy0,xy1=None):
        #print('award ')
        #print('points',self.points)
        #print('point',xy0)
        self.adopted = True
        self.nadopted += 1
        self.points = nu.vstack((self.points,xy0))
        self.angle = (2*tls(self.points[:,(1,0)]-nu.mean(self.points[:,(1,0)])))%1
        
    def bid(self,xy0,xy1=None):
        #angle = (nu.arctan2(*(xy-nu.mean(self.points,0)))%nu.pi)/nu.pi
        a0 = nu.arctan2(*(xy0-nu.mean(self.points,0)))/nu.pi
        e0 = self.evaluateAngle(a0)
        #print('points',self.points)
        #print('point',xy0)
        #print('ae',a0,e0)
        return e0

    def evaluateAngle(self,a):
        ad = nu.array([a,a%1])-self.angle
        return ad[nu.argmin(nu.abs(ad))]

 
    def bidOld(self,xy0,xy1=None):
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

    def getScore(self):
        return score

    def died(self):
        return self.score < Agent.minScore

    def getAngle(self,xy):
        return nu.arctan2(*(self.point-xy))%nu.pi/nu.pi
        
    def getMeanPoints(self):
        return self.points

    def getMeanPointsOld(self):
        mp = nu.zeros((len(self.points),2))
        for i,p in enumerate(self.points):
            if len(p) > 1:
                mp[i,:] = nu.mean(nu.array(p).reshape((-1,2)),0)
            else:
                mp[i,:] = p[0]
        return mp

    def evaluate(self):
        mp = self.getMeanPoints()
        N = mp.shape[0]
        dm = nu.zeros((N,N))
        print(mp)
        #angles = []
        for i in range(N):
            for j in range(i):
                a = mp[i,:]
                b = mp[j,:]
                th_ab = nu.arctan2(*(b-a))
                se = 0
                for k in range(N):
                    if k != i and k !=j:
                        c = mp[k,:]
                        th_bc = nu.arctan2(*(c-b))
                        se += (nu.sin(th_bc-th_ab)*nu.sum((c-b)**2)**.5)**2
                dm[i,j] = (se/(N-2))**.5
                dm[j,i] = (se/(N-2))**.5
        l = cluster.hierarchy.linkage(spatial.distance.squareform(dm))
        dm = dm[:,nu.argsort(dm[10,:])]
        dm = dm[nu.argsort(dm[:,2]),:]
        nu.savetxt('/tmp/dm.txt',dm)
        maxDist = 7
        c = cluster.hierarchy.fcluster(l,maxDist,criterion='distance')
        idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
        kv = idict.items()
        print(c)
        kv.sort(key=lambda x: len(x[1]))
        winner = kv[-1][0]
        losers = nu.where(c != winner)
        winners = nu.where(c == winner)
        print('w',winners)
        print('l',losers)

    def evaluateNew(self):
        mp = self.getMeanPoints()
        N = mp.shape[0]
        dm = nu.zeros((N,N))
        print(mp)
        #angles = []
        for i in range(N):
            for j in range(i):
                a = mp[i,:]
                b = mp[j,:]
                th_ab = nu.arctan2(*(b-a))
                dm[i,j] = th_ab

    def getTrajectory(self):
        N = len(self.points)
        if N == 1:
            return self.points[0,:]
        #points = nu.round(nu.array([x[0] if len(x) == 1 else nu.mean(nu.array(x),0) 
        #          for x in self.points])).astype(nu.int)
        points = self.points
        tr = []
        i = 0
        while i < N-1:
            dx = points[i+1,1]-points[i,1]
            dy = points[i+1,0]-points[i,0]
            delta = max(dx,dy)+2
            z = nu.round(nu.column_stack((nu.linspace(points[i,0],points[i+1,0],delta),
                             nu.linspace(points[i,1],points[i+1,1],delta)))).astype(nu.int).reshape((-1,2))
            tr.append(z)
            i += 1
        return nu.vstack(tr)

    def awardOld(self,xy0,xy1=None):
        self.adopted = True
        self.nadopted += 1
        if xy1 != None:
            self.points.append((xy0,xy1))
        else:
            self.points.append((xy0,))

        if True:
            return True

        bestAngle = self.bid(xy0,xy1)
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
                #print('np',bestAngle,self.point,xy0,xy1,newpoint)
            else:
                print('Error, dont know what to do')
        self.point = ((self.nadopted-1)*self.point+newpoint)/self.nadopted
        self.points.append(newpoint)
     
                
class BarLineAgent(Agent):
    targetAngleDeg = 90
    minScore = -10
    
class StaffLineAgent(Agent):
    targetAngleDeg = 0
    minScore = -5
        
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
    print('protocandidates',candidates)
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
    maxDeviation = 3/360.
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
                #ap.drawAgent(c,agents[bestBidder])
            else:
                unadopted.append(i)
    for i in unadopted:
        newagent = AgentType(nu.mean(nu.array(candidates[i]),0))
        #ap.register(newagent)
        agents.append(newagent)
        #for j in candidates[i]:
            #print('unadopted',j)
    
    agentsn = []
    for a in agents:
        alive = a.tick()
        print('angle',a.angle,a.points)
        goodAngle = nu.abs(a.angle-a.targetAngleDeg/360.) < maxDeviation
        if alive and goodAngle:
            agentsn.append(a)
        else:
            pass #ap.unregister(a)
    
    return agentsn


#def plotTrajectories(img,col,rows):
#    for i in rows:
#        img[i,col] = 255

def findStaffLines(img,fn):
    N,M = img.shape
    maxStaffLines = 70
    lookatProportion = .2
    vsums = nu.sum(img,0)
    #vsums = vsums[nu.nonzero(vsums)[0]]
    colorder = nu.argsort(vsums)
    colorder = colorder[nu.nonzero(vsums[colorder])]

    #print(vsums[colorder[:10]])
    #print('first nonz:',nu.where(vsums[colorder]>0))

    agents = []
    
    ap = AgentPainter(img)
    #for c in colorder[:int(lookatProportion*N)]:
    for c in colorder[:100]:
        print('column',c)
        agents = getCrossings(img[:,c],agents,StaffLineAgent,ap,horz=c)
        print('agents',len(agents))
        #agents = agents[:maxStaffLines]
        agents.sort(key=lambda x: x.age*x.score)
        agents.reverse()
        for a in agents[:50]:
            print('{0}'.format(a))
    
    k = 10
    print(agents[k])
    #agents[k].evaluate()
    for a in agents[0:k+1]:
        ap.register(a)
        ap.drawAgent(a)
    ap.writeImage(fn)
    sys.exit()


    print(N/2)
    agents = agents[:maxStaffLines]
    agents.sort(key=lambda x: x.age*x.score)
    agents.reverse()
    for a in agents:
        print('{1} {0}'.format(a,a.point[0]))
    k = nu.argmin(nu.diff([a.score for a in agents]))
    print(k+1,'stafflines found')
    for a in agents[:k+1]:
        print('{1} {0}'.format(a,a.point[0]))
    print('nr of agents',len(agents))
    return agents[:10]


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
    findStaffLines(img[:410,:],fn)
    #findStaffLines(img[:100,:],fn)
    #nu.savetxt('/tmp/p.txt',nu.sum(img,1))
    sys.exit()    

#problematic:
"""
chopin:
7 5 1
18 1 1
35 2 7

"""
