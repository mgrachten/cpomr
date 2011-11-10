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
                x = -r*nu.sin(agent.getAngle()*nu.pi)+agent.mean[0]
                y = -r*nu.cos(agent.getAngle()*nu.pi)+agent.mean[1]
                #print(r,agent.getAngle(),agent.mean,x,y)
                #print(x,y)
                if 0 <= x < M and 0 <= y < N:
                    alpha = min(.8,max(.1,.5+float(agent.score)/agent.age))
                    self.paint(nu.array((x,y)),c2,alpha)

            self.paintRect(agent.points[0][0],agent.points[0][0],
                           agent.points[0][1],agent.points[0][1],c)
            self.paintRect(agent.mean[0]+2,agent.mean[0]-2,
                           agent.mean[1]+2,agent.mean[1]-2,c)
            for p in agent.points:
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
            #a.mean-nu.arange(0,self.img.shape[1])
            M,N = self.img.shape[1:]
            xbegin = agent.mean[0]+(-agent.mean[1])*nu.tan(((agent.angle-agent.targetAngle+.5)%1-.5)*nu.pi)
            xend = agent.mean[0]+(N-agent.mean[1])*nu.tan(((agent.angle-agent.targetAngle+.5)%1-.5)*nu.pi)
            if xbegin < 0 or xend >= M:
                print('undrawable agent',agent.mean,xbegin,xend,M,agent.angle)
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
    return ((nu.arctan2(-v[1,1],v[1,0])-nu.pi)%nu.pi)/nu.pi

def getError(x,a):
    return nu.sum(nu.dot(x,nu.array([nu.cos(a*nu.pi),-nu.sin(a*nu.pi)]).T)**2)**.5

class Agent(object):
    targetAngle = 0
    maxError = 5
    maxAngleDev = 1/180. # in degrees
    
    def __init__(self,xy0,xy1=None):
        if xy1 != None:
            xy = (xy0+xy1)/2.0
            self.lineThickness = nu.array([1+nu.sum((xy0-xy1)**2)**.5])
        else:
            xy = xy0
            self.lineThickness = nu.array([1.0])
        self.points = nu.array(xy).reshape((1,2))
        self.mean = xy
        self.angle = self.targetAngle
        self.error = 0
        self.score = 0
        self.adopted = True
        self.age = 0
        self.id = str(self.__hash__())
        self.offspring = 0
        #self.scorehist = []

    def clone(self):
        clone = type(self)(nu.array((0,0)))
        clone.age = self.age
        clone.error = self.error
        clone.points = self.points
        clone.score = self.score
        clone.mean = self.mean
        clone.angle = self.angle
        clone.lineThickness = self.lineThickness[:]
        if self.offspring == 0:
            clone.id = self.id
        else:
            clone.id = '{0}_{1:02d}'.format(self.id,self.offspring)
        self.offspring += 1
        return clone

    def __str__(self):
        return 'Agent: {7}; angle: {3:03f}; error: {0:03f} age: {1}; points: {2}; score: {4}; mean: {5}; thick: {6}'.format(self.error,self.age,self.points.shape[0],self.getAngle(),self.score,self.mean,self.getLineThickness(),self.id)

    def mergeable(self,other,track=False):
        e0 = getError(other.points-self.mean,self.getAngle())
        e1 = getError(self.points-other.mean,other.getAngle())
        e = (e0+e1)/2.0
        if track:
            print('errors')
            print(e0,e1,e)
        #return e if e < self.maxError else nu.inf
        return e

    def getLineThickness(self):
        return nu.median(self.lineThickness)

    def getAngle(self):
        return self.targetAngle + (self.angle-self.targetAngle+.5)%1-.5

    def merge(self,other,track=False):
        #self.points = nu.vstack((self.points,other.points))
        if track:
            print(self.points)
            print(other.points)
        self.points = nu.array(tuple(set([tuple(y) for y in nu.vstack((self.points,other.points))])))

        if track:
            print(self.points)

        self.lineThickness = nu.append(self.lineThickness,other.lineThickness)
        if track:
            print('merge: self, other, new mean')
            print(self.mean,other.mean,nu.mean(self.points,0))
        self.mean = nu.mean(self.points,0)
        self.angle = tls(self.points-self.mean)
        self.error = getError(self.points-self.mean,self.getAngle())
        #self.scorehist = self.scorehist+other.scorehist
        self.age = max(self.age,other.age)
        self.score = self.score+other.score
        cp = os.path.commonprefix([self.id,other.id]).split('_')
        self.offspring = 0
        if len(cp) > 1:
            self.id = '_'.join(cp[:-1])
        #else:
        #    self.id = '_'.join(cp[:-1])
            
    #def getScorehist(self):
    #    return nu.array(self.scorehist)

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
    
    def toVector(self):
        return nu.array((self.points.shape[0],
                         self.mean[0],
                         (self.angle+.5)%1-.5,
                        self.error,
                        self.score,
                        self.age))

    def died(self):
        #angleOK = nu.abs((self.angle*180-self.targetAngle+90)%180-90) <= self.maxAngleDev
        #angleOK = nu.abs((self.angle-self.targetAngle+.5)%1-.5) <= self.maxAngleDev
        angleOK = nu.abs(self.getAngle()-self.targetAngle) <= self.maxAngleDev
        errorOK = self.error <= self.maxError
        successRateOK = self.score >= self.minScore
        #print('agent',self.mean)
        #print(self.points)
        #print('angleOK',angleOK)
        #print('errorOK',errorOK)
        #print('successRateOK',successRateOK)
        r = not all((angleOK,errorOK,successRateOK))
        if r:
            print('Died: {0}; angleOK: {1}; errorOK: {2}, scoreOK: {3}'.format(self,angleOK,errorOK,successRateOK))
        return r
  
    def getIntersection(self,xy0,xy1):
        # NB: x and y coordinates are in reverse order
        # because of image conventions: x = vertical, y = horizontal
        #print('intersection',xy0,xy1)
        #print('mean,angle',self.mean,self.angle)
        dy,dx = nu.array(xy0-xy1,nu.float)
        # slope of new line
        slope = dy/dx
        ytilde,xtilde = xy0-self.mean
        # offset of new line
        b = (ytilde-slope*xtilde)
        # special case 1: parallel lines
        if slope == nu.tan(nu.pi*self.angle):
            print('parallel',slope)
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
        #print(x)
        r =  nu.array((slope*x+b,x))+self.mean
        #print('r',r)
        #sys.exit()
        return r
    
    def award(self,xy0,xy1=None):
        self.adopted = True
        track = nu.abs(self.mean[1]-2160) < 1
        if xy1 != None:
            error0 = nu.dot(xy0-self.mean,nu.array([nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]))
            error1 = nu.dot(xy1-self.mean,nu.array([nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]))
            lw = 1+nu.sum((xy0-xy1)**2)**.5
            if nu.sign(error0) != nu.sign(error1):
                xy = self.getIntersection(xy0,xy1)
            else:
                if track:
                    print('agent',self.mean)
                    print('award')
                    print('lw',lw)
                    print('self lw',self.getLineThickness())
                    print(xy0,xy1,lw,self.getLineThickness()+nu.std(self.lineThickness))
                if lw <= self.getLineThickness() + nu.std(self.lineThickness):
                    # it's most probably a pure segment of the line, store the mean
                    xy = (xy0+xy1)/2.0
                else:
                    xy = (xy0,xy1)[nu.argmin(nu.abs([error0,error1]))]
                if track:
                    print('xy',xy)
            self.lineThickness = nu.append(self.lineThickness,lw)
        else:
            self.lineThickness = nu.append(self.lineThickness,1.0)
            xy = xy0
        #print('adding:',xy)
        self.points = nu.vstack((self.points,xy))
        self.mean = nu.mean(self.points,0)
        self.angle = tls(self.points-self.mean)
        self.error = getError(self.points-self.mean,self.getAngle())
        if track:
            print('agent',self.mean,'got:')
            print(xy0,xy1,'error is now:',self.error)
    def _getAngle(self,xy):
        #print('attempt',0.67202086962263063,nu.arctan2(*((xy-self.mean)*nu.array([-1,1])))/nu.pi)
        #print('attempt',0.67202086962263063,((nu.arctan2(*(xy-self.mean))/nu.pi)+1)%1)
        #return -nu.arctan2(*(xy-self.mean))/nu.pi
        return ((nu.arctan2(*(xy-self.mean))/nu.pi)+1)%1

    def _getClosestAngle(self,a):
        return nu.sort([(a+1)%1,self.getAngle()-self.maxAngleDev,
                        self.getAngle()+self.maxAngleDev])[1]

    def bid(self,xy0,xy1=None):
        # distance of xy0 to the current line (defined by self.angle)
        if self.points.shape[0] == 1:
            # we have no empirical angle yet
            # find the optimal angle (within maxAngleDev), and give the error respective to that
            if xy1 == None:
                xyp1 = xy0
            else:
                xyp1 = xy1
            aa0 = self._getAngle(xy0)
            aa1 = self._getAngle(xyp1)
            a0 = self._getClosestAngle(aa0)
            a1 = self._getClosestAngle(aa1)
            #print('xy0 xyp1',xy0,xyp1)
            #print('aa0 aa1',aa0,aa1)
            #print('a0 a1',a0,a1)
            #print('selfangle',self.getAngle())
            angle = nu.sort([a1,a0,self.getAngle()])[1]
        else:
            angle = self.getAngle()
            #angle = self.angle

        #print('adjusting angle:',self.getAngle(),'to',angle)
        #print('cos angle, -sin angle:',nu.array([nu.cos(angle*nu.pi),-nu.sin(angle*nu.pi)]))
        apenalty = nu.abs(self.getAngle()-angle)
        error0 = nu.dot(xy0-self.mean,nu.array([nu.cos(angle*nu.pi),-nu.sin(angle*nu.pi)]))
        #print('error0',error0)
        if xy1 == None:
            return nu.abs(error0)+apenalty
        error1 = nu.dot(xy1-self.mean,nu.array([nu.cos(angle*nu.pi),-nu.sin(angle*nu.pi)]))
        #print('error1',error1)
        if nu.sign(error0) != nu.sign(error1):
            return 0.0+apenalty
        else:
            return min(nu.abs(error0),nu.abs(error1))+apenalty
        
    def getRotMatrix(self):
        return nu.array([[nu.sin(self.getAngle()*nu.pi),nu.cos(self.getAngle()*nu.pi)],
                         [nu.cos(self.getAngle()*nu.pi),-nu.sin(self.getAngle()*nu.pi)]])


class BarLineAgent(Agent):
    targetAngle = .32 # in rad/(2*pi), for example .5 is vertical
    # maxError should depend on imageSize
    # good values (empirically established):
    # maxError=5 for images of approx 826x1169; seems to work also for images of 2550x3510
    # larger resolutions may need a higher value of maxError
    maxError = 2 # mean perpendicular distance of points to line (in pixels)
    maxAngleDev = 1/180. # 
    minScore = -2

class StaffLineAgent(Agent):
    targetAngle = 0 
    maxError = 5
    maxAngleDev = 2/180. # radians/pi
    minScore = -1

if __name__ == '__main__':
    #high
    x01 = nu.array([200.0,200])
    x11 = nu.array([200.0,205])
    #low
    x00 = nu.array([400.0,200])
    x10 = nu.array([400.0,200])

    a0 = BarLineAgent(x00)
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
    print(nu.arctan2(*(x1-x2a))/nu.pi,2/180.)
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
    print(a)
    print(a.points)
    b = a.bid(x2a,x2b)
    print('bid result',b)
    sys.exit()
    print('tick',a.tick())
    print(a)
    sys.exit()
    print('')
    b = a.bid(x3,x3b)
    print('bid',b)
    print('a before adding')
    print(a)
    a.award(x3,x3b)
    print('tick',a.tick())
    print(a.points)
    #a.angle = .499999999
    #a.mean = nu.array([0,0])
    #d1 = nu.array([10,1])
    #d2 = nu.array([-10,1])
    #print('d',a.getIntersection(d1,d2))
    
