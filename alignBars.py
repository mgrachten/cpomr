#!/usr/bin/env python

import sys
import numpy as nu
from main import clusterCoords
import dtw
from scipy import cluster
from utilities import argpartition

def costf(x,y,i,j):
    return nu.sum((x[i]-y[j])**2)**.5

def barCost(x,y,i,j):
    p,c = dtw.dtw(x[i].d,y[j].d,
                  x[i].d.shape[0],
                  y[j].d.shape[0],
                  costf,
                  returnCost=True)
    #print('barcost',c)
    return c
    #return nu.abs(x[i].d.shape[0]-y[j].d.shape[0])

def cl(d,distance=1.0):
    if d.shape[0] == 0:
        return {}
    if d.shape[0] == 1:
        return {0:[0]}
    l = cluster.hierarchy.linkage(d)
    c = cluster.hierarchy.fcluster(l,distance,criterion='distance')
    return argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))

class Bar(object):
    def __init__(self,i,d):
        self.i = i
        self.d = d
        self.d = self.d[nu.argsort(self.d[:,1]),:]
        self.d = self.d[nu.argsort(self.d[:,0]),:]
        self.normalize()
    def normalize(self):
        means = nu.mean(self.d,0)
        std = nu.std(self.d,0)
        self.d = self.d-means
        for i,s in enumerate(std):
            if s > 0:
                self.d[:,0] /= s
    def __str__(self):
        return '{0} notes\n{1}'.format(self.d.shape[0],self.d)

class BarOMR(Bar):
    def __init__(self,i,d):
        self.i = i
        self.d = clusterNotes(nu.array(d,nu.float))
        #self.d = self.d[nu.argsort(self.d[:,0]),:]
        self.n = cl(self.d[:,0].reshape((-1,1)),2.0)
        for v in self.n.values():
            idx = tuple(v)
            self.d[idx,0] = nu.mean(self.d[idx,0])
        self.d[:,1] = -self.d[:,1]
        self.d = self.d[nu.argsort(self.d[:,1]),:]
        self.d = self.d[nu.argsort(self.d[:,0]),:]
        self.normalize()

def clusterNotes(n):
    return clusterCoords(n,distance=7)

def splitBars(x,barType=Bar):
    i = 0
    j = 1
    N = x.shape[0]
    bars = []
    while j < N:
        if x[i,0] != x[j,0]:
            bars.append(barType(x[i,0],x[i:j,(1,2)]))
            i = j
        j += 1
    bars.append(barType(x[i,0],x[i:j,(1,2)]))
    return bars

if __name__ == '__main__':
    fn1 = sys.argv[1]
    fn2 = sys.argv[2]
    d1 = nu.loadtxt(fn1,nu.int)
    d2 = nu.loadtxt(fn2,nu.int)
    #y = nu.array([[1,0],[2,0],[3.4,0],[2,1]])
    #x = nu.array([[0,1],[2,2],[3,2],[4,3],[2.5,2]])
    #print(dtw.dtw(x,y,len(x),len(y),costf,returnCost=True))
    #sys.exit()
    b1 = splitBars(d1,BarOMR)
    b2 = splitBars(d2,Bar)
    p = dtw.dtw(b1,b2,len(b1),len(b2),barCost)

    for i,j in p:
        print('')
        print(i,j)
        print(b1[i])
        print(b2[j])
