#!/usr/bin/env python

import sys
import numpy as nu
from main import clusterCoords
import dtw
from scipy import cluster
from utilities import argpartition

def costf(x,y,i,j):
    return nu.sum((x[i]-y[j])**2)**.5

barAlignments = {}
def barCost(x,y,i,j):
    global barAlignments
    p,c = dtw.dtw(x[i].d,y[j].d,
                  x[i].d.shape[0],
                  y[j].d.shape[0],
                  costf,
                  returnCost=True)
    #print('barcost',c)
    barAlignments[(i,j)] = p
    diagDevPenalty = 1000
    diagDeviation = diagDevPenalty*nu.abs(i-j)
    print(c)
    return c+diagDeviation


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
        self.dOrig = self.d.copy()
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
        assert d.shape[0] > 0
        self.pageNr = d[0,2]
        self.barNr = d[0,3]
        d = d[:,0:2]
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
        self.dOrig = self.d.copy()
        self.dOrig[:,1] = -self.dOrig[:,1]
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
            if barType == BarOMR:
                bars.append(barType(x[i,0],x[i:j,(1,2,3,4)]))
            else:
                bars.append(barType(x[i,0],x[i:j,(1,2)]))
            i = j
        j += 1
    if barType == BarOMR:
        bars.append(barType(x[i,0],x[i:j,(1,2,3,4)]))
    else:
        bars.append(barType(x[i,0],x[i:j,(1,2)]))
    return bars

def getScoreAndy(f):
    d = nu.genfromtxt(f,dtype=None,names=True,delimiter=',')
    return nu.column_stack((d['Bar'],100*d['Start_Beat'],d['Pitch'])).astype(nu.int)

def mapNotes2Coords(p,barOMR):
    ijj = argpartition(lambda x: x, p[:,0])
    coords = nu.array([nu.median(barOMR.dOrig[p[tuple(v),1],:].reshape((-1,2)),0)
              for k,v in ijj.items()])
    coords = nu.column_stack((barOMR.pageNr*nu.ones(coords.shape[0]),
                              barOMR.barNr*nu.ones(coords.shape[0]),
                              coords)).astype(nu.int)
    return coords

if __name__ == '__main__':
    fn1 = sys.argv[1]
    ffn2 = sys.argv[2:]
    #d1 = nu.loadtxt(fn1,nu.int)
    d1 = getScoreAndy(fn1)
    #d1 = d1[d1[:,0] < 12,:]
    #d2 = nu.empty((0,5),nu.int)
    d2 = []
    nbars = 0
    for i,fn2 in enumerate(ffn2):
        x = nu.loadtxt(fn2,nu.int)
        x = nu.column_stack((x,i*nu.ones(x.shape[0],nu.int),x[:,0]))
        x[:,0] += nbars
        nbars = x[-1,0]+1
        #d2 = nu.column_stack((d2,x))
        d2.append(x)
    d2 = nu.vstack(tuple(d2))
    #print(d2)
    b1 = splitBars(d1,Bar)
    b2 = splitBars(d2,BarOMR)
    #sys.exit()
    #p = dtw.dtw(b1,b2,len(b1),len(b2),barCost)
    p = dtw.dtw(b1,b2,3,3,barCost,memory=2)

    coords = []
    for i,j in p:
        print(i,j)
        #print(barAlignments[(i,j)])
        coords.append(mapNotes2Coords(barAlignments[(i,j)],b2[j]))
    coords = nu.vstack(tuple(coords))
    outfile='/tmp/notecoords.txt'
    nu.savetxt(outfile,coords,fmt='%d')
