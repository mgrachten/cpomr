#!/usr/bin/env python

import sys,os
from scipy import signal,cluster,spatial
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition, makeColors

def tls(X):
    """total least squares for 2 dimensions
    """
    u,s,v = nu.linalg.svd(X)
    v = v.T
    r = ((nu.arctan2(-v[1,1],v[0,1])-nu.pi)%nu.pi)/nu.pi
    return r

def getError(x,a):
    return nu.sum(nu.dot(x,nu.array([nu.cos(a*nu.pi),-nu.sin(a*nu.pi)]).T)**2)**.5

class Agent(object):
    targetAngle = 0
    maxError = 5
    maxAngleDev = 1 # in degrees
    
    def __init__(self,xy):
        self.points = nu.array(xy).reshape((1,2))
        self.mean = xy
        self.angle = self.targetAngle
        self.error = 0
        self.score = 0
        self.adopted = True
        self.age = 0
        self.scorehist = []

    def __str__(self):
        return 'Agent: angle: {3:03f}; error: {0:03f} age: {1}; points: {2}; score: {5}; eval: {4}'.format(self.error,self.age,self.points.shape[0],self.angle,self.evaluate(),self.score)

    def mergeable(self,other):
        e0 = getError(other.points-self.mean,self.angle)
        e1 = getError(self.points-other.mean,other.angle)
        e = (e0+e1)/2.0
        #return e if e < self.maxError else nu.inf
        return e

    def merge(self,other):
        self.points = nu.vstack((self.points,other.points))
        self.mean = nu.mean(self.points,0)
        self.angle = tls(self.points-self.mean)
        self.error = getError(self.points-self.mean,self.angle)
        #self.scorehist = nu.vstack((self.scorehist,other.scorehist))
        self.scorehist = self.scorehist+other.scorehist
        self.age = max(self.age,other.age)
        self.score = self.score+other.score
        
    def getScorehist(self):
        return nu.array(self.scorehist)

    def tick(self):
        self.age += 1
        if self.adopted:
            self.score += 1
        else:
            self.score -= 1
        self.scorehist.append((self.age,self.score))
        self.adopted = False
        return not self.died()
    
    def toVector(self):
        return nu.array((self.points.shape[0],
                         self.mean[0],
                         (self.angle+.5)%1-.5,
                        self.error,
                        self.score,
                        self.age)
                        )

    def died(self):
        angleOK = nu.abs((self.angle*180-self.targetAngle+90)%180-90) <= self.maxAngleDev
        errorOK = self.error <= self.maxError
        successRateOK = self.score >= self.minScore
        return not all((angleOK,errorOK,successRateOK))
  
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
        return nu.array((slope*x+b,x))
    
    def evaluate(self):
        # + points
        # - error
        # - nu.abs(self.angle - self.targetAngle)
        # angleDiff = nu.abs((self.angle*180-self.targetAngle+90)%180-90)
        angleDiff = nu.abs((self.angle-self.targetAngle/180.+.5)%1-.5)
        v = (.01*self.age+self.points.shape[0]/(1.0+self.age))/(1+self.error+100*angleDiff)
        return v

    def award(self,xy0,xy1=None):
        self.adopted = True
        #self.nadopted += 1
        if xy1 != None:

            error0 = nu.dot(xy0-self.mean,nu.array([nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]))
            error1 = nu.dot(xy1-self.mean,nu.array([nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]))
            if nu.sign(error0) != nu.sign(error1):
                xy = self.getIntersection(xy0,xy1)
            else:
                xy = (xy0,xy1)[nu.argmin(nu.abs([error0,error1]))]
            
        else:
            xy = xy0
        #print(xy0,xy1,xy,self.mean)
        #print('awarded!0')
        #print(self.points)
        self.points = nu.vstack((self.points,xy))
        #print('awarded!1')
        #print(self.points)
        self.mean = nu.mean(self.points,0)
        self.angle = tls(self.points-self.mean)
        self.error = getError(self.points-self.mean,self.angle)
        #print('mae',self.mean,self.angle,self.error)

    def bid(self,xy0,xy1=None):
        #print('agent {0} bidding:'.format(self))
        #print(' mean: {0}:'.format(self.mean))

        # distance of xy0 to the current line (defined by self.angle)
        error0 = nu.dot(xy0-self.mean,nu.array([nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]))
        #print('\tp0 error',xy0,error0)
        if xy1 == None:
            return nu.abs(error0)
        error1 = nu.dot(xy1-self.mean,nu.array([nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]))
        #print('\tp1 error',xy1,error1)
        if nu.sign(error0) != nu.sign(error1):
            return 0.0
        else:
            return min(nu.abs(error0),nu.abs(error1))
        
    def getRotMatrix(self):
        return nu.array([[nu.sin(self.angle*nu.pi),nu.cos(self.angle*nu.pi)],
                         [nu.cos(self.angle*nu.pi),-nu.sin(self.angle*nu.pi)]])

    def getTrajectory(self):
        N = self.points.shape[0]
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


class BarLineAgent(Agent):
    targetAngle = 90
    maxError = 5
    
class StaffLineAgent(Agent):
    targetAngle = 0 # in degrees
    maxError = 5
    maxAngleDev = 1 # in degrees
    minScore = -1

if __name__ == '__main__':
    x1 = nu.array([255,29])
    x2a = nu.array([253,30])
    x2b = nu.array([256,30])
    #x3 = nu.array([-.1,-100])
    x3 = nu.array([5,-1])
    x3b = nu.array([-5,-1])
    #xy = nu.vstack((x1,x2))

    a = StaffLineAgent(x1)
    b = a.bid(x2a,x2b)
    print('bid result',b)
    a.award(x2a,x2b)
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
    
