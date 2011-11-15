#!/usr/bin/env python

import sys,os
from scipy import signal,cluster,spatial
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition, makeColors
from copy import deepcopy

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

    def isRegistered(self,agent):
        return self.agents.has_key(agent)
        
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

    def drawAgentGood(self,agent,rmin=-100,rmax=100):
        if self.agents.has_key(agent):
            #print('drawing')
            #print(agent)
            c = self.colors[self.agents[agent]]
            c1 = nu.minimum(255,c+50)
            c2 = nu.maximum(0,c-100)
            M,N = self.img.shape[1:]
            for r in range(rmin,rmax):
                x = -r*nu.sin(agent.getAngle()*nu.pi)+agent.getDrawMean()[0]
                y = -r*nu.cos(agent.getAngle()*nu.pi)+agent.getDrawMean()[1]
                #print(r,agent.getAngle(),agent.getDrawMean(),x,y)
                #print(x,y)
                if 0 <= x < M and 0 <= y < N:
                    alpha = min(.8,max(.1,.5+float(agent.score)/agent.age))
                    self.paint(nu.array((x,y)),c2,alpha)

            self.paintRect(agent.getDrawPoints()[0][0],agent.getDrawPoints()[0][0],
                           agent.getDrawPoints()[0][1],agent.getDrawPoints()[0][1],c)
            self.paintRect(agent.getDrawMean()[0]+2,agent.getDrawMean()[0]-2,
                           agent.getDrawMean()[1]+2,agent.getDrawMean()[1]-2,c)
            for p in agent.getDrawPoints():
                self.paint(p,c1)


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
            #a.getDrawMean()-nu.arange(0,self.img.shape[1])
            M,N = self.img.shape[1:]
            xbegin = agent.getDrawMean()[0]+(-agent.getDrawMean()[1])*nu.tan(((agent.angle-agent.targetAngle+.5)%1-.5)*nu.pi)
            xend = agent.getDrawMean()[0]+(N-agent.getDrawMean()[1])*nu.tan(((agent.angle-agent.targetAngle+.5)%1-.5)*nu.pi)
            if xbegin < 0 or xend >= M:
                print('undrawable agent',agent.getDrawMean(),xbegin,xend,M,agent.angle)
                return False
            dx = nu.abs(xend-xbegin)
            dy = N
            delta = max(dx,dy)+2
            z = nu.round(nu.column_stack((nu.linspace(xbegin,xend,delta),
                                          nu.linspace(0,N-1,delta)))).astype(nu.int)
            for p in z:
                try:
                    alpha = min(.8,max(.1,.5+float(agent.score)/agent.age))
                    self.paint(p,c2,alpha)
                except IndexError:
                    print(p,img.shape[1:])
                    
            #print(img.shape)
            #print('draw',agent.angle,agent.getDrawMean())
            #print(z)
            #print(agent.getDrawPoints())
            self.paintRect(agent.getDrawPoints()[0][0],agent.getDrawPoints()[0][0],
                           agent.getDrawPoints()[0][1],agent.getDrawPoints()[0][1],c)
            for p in agent.getDrawPoints():
                #for p in pp:
                self.paint(p,c1)
                if p[0]-1 >= 0:
                    self.paint((p[0]-1,p[1]),c1)
                if p[0]+1 < self.img.shape[0]:
                    self.paint((p[0]+1,p[1]),c1)
        else:
            sys.stderr.write('Warning, unknown agent\n')

    def paint(self,coord,color,alpha=1):
        #print('point',coord,img.shape)
        self.img[:,int(coord[0]),int(coord[1])] = (1-alpha)*self.img[:,int(coord[0]),int(coord[1])]+alpha*color

    def paintVLine(self,y,alpha=.5):
        self.img[:,:,y] = (1-alpha)*self.img[:,:,y]+alpha*0
    def paintHLine(self,x,alpha=.5):
        self.img[:,x,:] = (1-alpha)*self.img[:,x,:]+alpha*0

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

def tls(X):
    """total least squares for 2 dimensions
    """
    u,s,v = nu.linalg.svd(X)
    #v = v.T
    #return ((nu.arctan2(-v[1,1],v[0,1])-nu.pi)%nu.pi)/nu.pi
    #return ((nu.arctan2(-v[1,1],v[1,0])-nu.pi)%nu.pi)/nu.pi
    return (nu.arctan2(-v[1,1],v[1,0])/nu.pi+1)%1.0

def getError(x,a):
    return nu.sum(nu.dot(x,nu.array([nu.cos(a*nu.pi),-nu.sin(a*nu.pi)]).T)**2)**.5

def makeAgentClass(targetAngle,maxAngleDev,maxError,minScore,offset=0):
    class CustomAgent(Agent): pass
    CustomAgent.targetAngle = (targetAngle+1.0)%1
    CustomAgent.maxError = maxError
    CustomAgent.maxAngleDev = maxAngleDev
    CustomAgent.minScore = minScore
    CustomAgent.offset = offset
    return CustomAgent

class Agent(object):
    targetAngle = None
    maxError = None
    maxAngleDev = None
    minScore = None

    def __init__(self,xy0,xy1=None):
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

    def clone(self):
        clone = type(self)(nu.array((0,0)))
        clone.age = self.age
        clone.error = self.error
        clone.points = self.points
        clone.score = self.score
        clone.mean = self.mean
        clone.angleDev = self.angleDev
        clone.offset = self.offset
        clone.lineWidth = self.lineWidth[:]
        if self.offspring == 0:
            clone.id = self.id
        else:
            clone.id = '{0}_{1:02d}'.format(self.id,self.offspring)
        self.offspring += 1
        return clone

    def __str__(self):
        return 'Agent: {7}; angle: {3:03f}; error: {0:03f} age: {1}; points: {2}; score: {4}; mean: {5}; width: {6}'.format(self.error,self.age,self.points.shape[0],self.getAngle(),self.score,self.mean,self.getLineWidth(),self.id)

    def mergeable(self,other,track=False):
        e0 = getError(other.points-self.mean,self.getAngle())
        e1 = getError(self.points-other.mean,other.getAngle())
        e = (e0+e1)/2.0
        if track:
            print('errors')
            print(e0,e1,e)
        #return e if e < self.maxError else nu.inf
        return e

    def getLineWidth(self):
        return self.lw

    def getLineWidthStd(self):
        return self.lwstd

    def _addLineWidth(self,w):
        self.lineWidth.append(w)
        self.lw = nu.median(self.lineWidth)
        self.lwstd = nu.std(self.lineWidth)
    
    #def getAngle(self):
    #    return self.targetAngle + (self.angleDev-self.targetAngle+.5)%1-.5
    def getAngle(self):
        return self.targetAngle + self.angleDev

    def getDrawPoints(self):
        return self.points+nu.array([self.offset,0])
    def getDrawMean(self):
        return self.mean+nu.array([self.offset,0])

    def merge(self,other,track=False):
        #self.points = nu.vstack((self.points,other.points))
        if track:
            print(self.points)
            print(other.points)
        self.points = nu.array(tuple(set([tuple(y) for y in nu.vstack((self.points,other.points))])))

        if track:
            print(self.points)

        self.lineWidth = nu.append(self.lineWidth,other.lineWidth)
        if track:
            print('merge: self, other, new mean')
            print(self.mean,other.mean,nu.mean(self.points,0))
        self.mean = nu.mean(self.points,0)
        self.angleDev = tls(self.points-self.mean)
        self.error = getError(self.points-self.mean,self.getAngle())
        #self.scorehist = self.scorehist+other.scorehist
        self.age = max(self.age,other.age)
        self.score = self.score+other.score
        cp = os.path.commonprefix([self.id,other.id]).split('_')
        self.offspring = 0
        if len(cp) > 1:
            self.id = '_'.join(cp[:-1])
            
    def tick(self,immortal=False):
        self.offspring = 0
        self.age += 1
        if self.adopted:
            self.score += 1
        else:
            self.score -= 1
        #self.scorehist.append((self.age,self.score))
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
        if r:
            print('Died: {0}; angleOK: {1}; errorOK: {2}, scoreOK: {3}'.format(self,angleOK,errorOK,successRateOK))
            print('a,ta',self.angleDev,self.targetAngle)
            print('adev,maxadev',nu.abs(self.getAngle()-self.targetAngle),self.maxAngleDev)
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
        if slope == nu.tan(nu.pi*self.getAngle()):
            print('parallel',slope)
            return None

        # special case 2: vertical line
        if nu.isinf(slope):
            # x constant
            x = xy0[1]
            y = nu.tan(nu.pi*self.getAngle())*(x - self.mean[1])+self.mean[0]
            return nu.array((y,x))
        # special case 3: line undefined
        if nu.isnan(slope):
            # undefined slope, constant y
            return None

        x = b/(nu.tan(nu.pi*self.getAngle())-slope)
        r =  nu.array((slope*x+b,x))+self.mean
        return r

    def _getAngleDistance(self,a):
        #return -nu.arctan2(*(xy-self.mean))/nu.pi
        return (a-self.targetAngle+.5)%1-.5

    def _getAngle(self,xy):
        #return -nu.arctan2(*(xy-self.mean))/nu.pi
        return ((nu.arctan2(*((xy-self.mean)))/nu.pi)+1)%1

    def _getClosestAngle(self,a):
        return nu.sort([(a+1)%1,self.getAngle()-self.maxAngleDev,
                        self.getAngle()+self.maxAngleDev])[1]

    def preparePointAdd(self,xy0,xy1=None):
        if xy1 != None:
            error0 = nu.dot(xy0-self.mean,nu.array([nu.cos(self.getAngle()*nu.pi),-nu.sin(self.getAngle()*nu.pi)]))
            error1 = nu.dot(xy1-self.mean,nu.array([nu.cos(self.getAngle()*nu.pi),-nu.sin(self.getAngle()*nu.pi)]))
            lw = 1+nu.sum((xy0-xy1)**2)**.5
            if nu.sign(error0) != nu.sign(error1):
                xy = self.getIntersection(xy0,xy1)
            else:
                if lw <= self.getLineWidth() + self.getLineWidthStd():
                    # it's most probably a pure segment of the line, store the mean
                    xy = (xy0+xy1)/2.0
                else:
                    # too thick to be a line, store the point that has smallest error
                    xy = (xy0,xy1)[nu.argmin(nu.abs([error0,error1]))]
        else:
            lw = 1.0
            xy = xy0
        points = nu.vstack((self.points,xy))
        print('points')
        print(points)
        mean = nu.mean(points,0)
        print('mean')
        print(mean)
        tlsr = tls(points-mean)
        print('z points')
        print(points-mean)
        print('tlsr',tlsr)
        angleDev = ((tlsr-self.targetAngle)+.5)%1-.5
        error = getError(points-mean,angleDev+self.targetAngle)
        return error,angleDev,mean,lw,points

    def bid(self,xy0,xy1=None):
        error,angleDev,mean,lw,points = self.preparePointAdd(xy0,xy1=xy1)
        return error,angleDev,mean,lw,points

    def award(self,xy0,xy1=None):
        self.error,self.angleDev,self.mean,self.lw,self.points = self.preparePointAdd(xy0,xy1=xy1)


class BarLineAgent(Agent):
    targetAngle = .32 # in rad/(2*pi), for example .5 is vertical
    # maxError should depend on imageSize
    # good values (empirically established):
    # maxError=5 for images of approx 826x1169; seems to work also for images of 2550x3510
    # larger resolutions may need a higher value of maxError
    maxError = 2 # mean perpendicular distance of points to line (in pixels)
    maxAngleDev = 90/180. # 
    minScore = -2

class StaffLineAgent(Agent):
    targetAngle = 0 
    maxError = 5
    maxAngleDev = 2/180. # radians/pi
    minScore = -1

if __name__ == '__main__':
    StaffAgent = makeAgentClass(targetAngle=-.02,
                                maxAngleDev=4/180.,
                                maxError=20,
                                minScore=-2,
                                offset=0)
    
    #high
    #x01 = nu.array([200.0,200])
    #x11 = nu.array([200.0,205])
    #low
    x00 = nu.array([400.0,200])
    #x10 = nu.array([400.0,200])

    a0 = StaffAgent(x00)
    print(a0)
    y1 = nu.array((398,1000))
    y2 = nu.array((595,1000))
    print('a',a0._getAngle(y1))
    print('bid',a0.bid(y1,y2))
    print('award',a0.award(y1,y2))
    print(a0)
    print('pass?',a0.tick())
    
    print('\n\n')
    sys.exit()




    a0.award(x01)
    a1 = BarLineAgent(x10)
    a1.award(x00)
    a2 = BarLineAgent(x00)
    a2.award(x01)
    a2.award(x10)
    a2.award(x11)
    print(a0.mergeable(a1))
    print(a0)
    print(a1)
    print(a2)
    sys.exit()
    x2b = nu.array([5,14])
    #x2a = nu.array([15,10])
    #x2b = nu.array([5,14])
    #x3 = nu.array([-.1,-100])
    x3 = nu.array([5,-1])
    x3b = nu.array([-5,-1])
    #xy = nu.vstack((x1,x2))

    #print(a._getAngle(x2b))
    #print(a._getClosestAngle(a._getAngle(x2b)))
    b = a.bid(x2a,x2b)
    a.award(x2a,x2b)
   
    #a.mean = nu.array([0,0])
    #d1 = nu.array([10,1])
    #d2 = nu.array([-10,1])
    #print('d',a.getIntersection(d1,d2))
    
