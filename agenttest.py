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
    return -nu.arctan2(-v[1,1],v[0,1])/(2*nu.pi)

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
        self.adopted = True
        self.nadopted += 1
        self.points = nu.vstack((self.points,xy0))
        print('tls',tls(self.points-nu.mean(self.points,0)))
        self.angle = (2*tls(self.points[:,(1,0)]-nu.mean(self.points[:,(1,0)],0)))%1
        print('points',self.points)
        print('angle',self.angle)
        
    def bid(self,xy0,xy1=None):
        #angle = (nu.arctan2(*(xy-nu.mean(self.points,0)))%nu.pi)/nu.pi
        a0 = nu.arctan2(*(xy0-nu.mean(self.points,0)))/nu.pi
        e0 = self.evaluateAngle(a0)
        #print('points',self.points)
        #print('point',xy0)
        #print('ae',a0,e0)
        return e0

    def evaluateAngle(self,a):
        ad = nu.array([a,(a+1)%1])-self.angle
        return ad[nu.argmin(nu.abs(ad))]
        #return ((a-self.angle)+1)%1

 
     
                
class BarLineAgent(Agent):
    targetAngleDeg = 90
    minScore = -10
    
class StaffLineAgent(Agent):
    targetAngleDeg = 0
    minScore = -5

if __name__ == '__main__':
    x1 = nu.array([0,0])
    x2 = nu.array([1,100])
    #x1 = nu.array([0,0])
    a = StaffLineAgent(x1)
    maxAngle = 1/180.
    b = a.bid(x2)
    print(b,maxAngle)
    print(nu.abs(b)<maxAngle)
    a.award(x2)
