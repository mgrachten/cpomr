#!/usr/bin/env python

import sys
import numpy as nu
from main import clusterCoords
import dtw
from scipy import cluster
from utilities import argpartition

def costf(x,y,i,j):
    return nu.abs(x[i]-y[j])

def cl(d,distance=1.0):
    if d.shape[0] == 0:
        return {}
    if d.shape[0] == 1:
        return {0:d[0,0]}
    l = cluster.hierarchy.linkage(d)
    c = cluster.hierarchy.fcluster(l,distance,criterion='distance')
    idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
    print('c',idict)
    
    cdict = dict([(k,int(nu.round(nu.mean(d[tuple(v),:])))) for k,v in idict.items()])
    return cdict

class Bar(object):
    def __init__(self,i,d):
        self.i = i
        self.d = clusterNotes(nu.array(d,nu.float))
        self.d = self.d[nu.argsort(self.d[:,1]),:]
        self.n = cl(self.d[:,1].reshape((-1,1)))
        #for k,v in self.n
        #    print(k,v)
        #    self.d[k,1] = v
    def __str__(self):
        #return '{0}'.format(self.d[nu.argsort(self.d[:,1]),:])
        #return '{0}\n{1}'.format(self.d,clusterNotes(self.d))
        return '{0} notes\n{1}\n{2}'.format(self.d.shape[0],self.d,self.n)

def clusterNotes(n):
    return clusterCoords(n,distance=5)

def splitBars(x):
    i = 0
    j = 1
    N = x.shape[0]
    bars = []
    while j < N:
        if x[i,0] != x[j,0]:
            bars.append(Bar(x[i,0],x[i:j,(1,2)]))
            i = j
        j += 1
    return bars

if __name__ == '__main__':
    fn1 = sys.argv[1]
    fn2 = sys.argv[2]
    d1 = nu.loadtxt(fn1,nu.int)
    d2 = nu.loadtxt(fn2,nu.int)
    for x in splitBars(d1)[:3]:
        print(x.i)
        print(x)
    sys.exit()
    p = dtw.dtw(x,y,len(x),len(y),costf)
    for i,j in p:
        print(i,j,x[i],y[j])
