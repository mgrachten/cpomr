#!/usr/bin/env python

import sys,os
from scipy import signal,cluster
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition

def smooth(x,k):
    f = signal.blackmanharris(k)#nu.ones(k)/nu.float(k)
    return nu.convolve(x,f,'same')

def makeHZ(img,fn):
    hl = nu.zeros(img.shape[0])
    nzz = [nu.nonzero(img[i,:])[0] for i in range(0,img.shape[0])]
    for i in range(1,img.shape[0]-1):
        nz = nu.array(list(set(nu.hstack((nzz[i-1],nzz[i],nzz[i+1])))))
        nz = nu.sort(nz)
        #nz = nu.nonzero(img[i,:])[0]
        dz = nu.diff(nz)
        dz[dz == 1] = 0
        nz = nu.nonzero(dz)[0]
        if len(nz) < 2:
            hl[i] = 0
        else:
            hl[i] = nu.max(nu.diff(nz))**2
        #nz = nu.sort(nz)
    hl = hl-nu.median(hl)
    hl[hl<0] = 0
    return hl
    #nu.savetxt('/tmp/hl.txt',hl)

def findBars(img,fn):
    # smooth to get rid of barline peaks
    # typical nr of staffs per page
    N = img.shape[0]
    S = N/100 # window size
    #S = img.shape[0]/35 # window size
    K = 12
    #hp = nu.sum(img,1)
    hp = makeHZ(img,fn)
    #hp = smooth(hp,S)
    med = nu.median(hp)
    hp[hp<med] = med
    nu.savetxt('/tmp/hp.txt',hp)
    k = N/nu.float(K)
    w = signal.gaussian(k,k/8.)/k
    c = nu.convolve(smooth(hp,S),w,'same')
    dc = nu.sign(nu.diff(c))
    d2c = nu.sign(nu.diff(dc))
    d2c = nu.insert(d2c,0,0)
    d2c = nu.insert(d2c,-1,0)
    #d2c = nu.convolve(d2c,nu.ones(S),'same')
    np = hp-nu.median(hp)
    dc2w = nu.zeros(np.shape)
    for i in nu.nonzero(d2c)[0]:
        if d2c[i] < 0:
            #dc2w[i] = nu.sum(np[i-S:i+S])
            dc2w[i] = nu.sum(nu.abs(nu.diff(np[i-N/(2*35):i+N/(2*35)])))
    dc2w[dc2w<0] = 0
    nu.savetxt('/tmp/dc2w.txt',dc2w)
    #pd = pair(nu.nonzero(dc2w)[0],dc2w)

    idx = nu.argsort(dc2w)[::-1]
    minSystems = 4
    maxSystems = 10
    sortedPeaks = dc2w[idx]
    print(len(sortedPeaks[sortedPeaks>0]))
    pidx = nu.arange(len(sortedPeaks))[(minSystems-1)*2:maxSystems*2+1:2]
    diffSortedPeaks = nu.diff(sortedPeaks[pidx])
    
    nu.savetxt('/tmp/sp.txt',sortedPeaks)
    nu.savetxt('/tmp/dsp.txt',nu.column_stack((pidx[:-1],diffSortedPeaks)))
    nu.savetxt('/tmp/c.txt',nu.column_stack((np,c,dc2w)))

    #thresholdPeak = minStaffs-1+nu.argmin(diffSortedPeaks[minStaffs-1:maxStaffs-1])
    nSystems = minSystems-1+nu.argmin(diffSortedPeaks)
    thresholdPeak = 2*nSystems+1
    #print(thresholdPeak)
    threshold = sortedPeaks[thresholdPeak]
    if threshold == 0:
        nSystems -= 1
        thresholdPeak = 2*nSystems+1
        threshold = sortedPeaks[thresholdPeak]
    #print(threshold)
    #dc2w[dc2w < 0 ] = 0
    print('number of systems found: {0}'.format(nSystems+1))

    tdc2w = dc2w.copy()
    tdc2w[tdc2w < threshold] = 0
    im_r = 255-img
    im_g = 255-img
    for i in nu.nonzero(tdc2w):
            im_g[i,:] = 0
            im_g[i-1,:] = 0
            im_g[i+1,:] = 0
    im_b = im_g
    writeImageData(os.path.join('/tmp',os.path.basename(fn).replace('.tif','.png')),img.shape,im_r,im_g,im_b)

def pair(idx,peaks):
    pairs = []
    maxDiff = 5
    ij = []
    for i,x in enumerate(idx):
        for j,y in enumerate(idx[:i]):
            pairs.append((nu.log(nu.abs(peaks[x]-peaks[y])),nu.abs(x-y)))
            ij.append((i,j))
    pairs = nu.array(pairs).reshape((-1,2))
    ij = nu.array(ij,nu.int)
    l = cluster.hierarchy.linkage(pairs)
    c = cluster.hierarchy.fcluster(l,maxDiff,criterion='distance')
    #c = cluster.hierarchy.fcluster(l,maxDiff)
    kv = nu.array([(k,len(v),nu.mean(pairs[tuple(v),1]),nu.std(pairs[tuple(v),1])) for k,v in argpartition(lambda x: x,c).items()])
    kv = kv[nu.argsort(kv[:,1]),:]
    kv = kv[kv[:,2] < 1000,:]
    kv = kv.astype(nu.int)
    print(kv)
    print('')
    clid = kv[-1,0]
    print(ij[c == clid])
    print(pairs[c == clid])
    print('')
    clid = kv[-2,0]
    print(ij[c == clid])
    print(pairs[c == clid])
    clid = kv[-3,0]
    print('')
    print(ij[c == clid])
    print(pairs[c == clid])


def findDistances(idx):
    distances = nu.diff(idx)
    #distances = []
    #for i,x in enumerate(idx):
    #    for j,y in enumerate(idx[:i]):
    #        distances.append(nu.abs(x-y))
    #distances.sort()

    return distances

if __name__ == '__main__':
    fn = sys.argv[1]
    try:
        img = 255-getImageData(fn)
    except IOError: 
        print('problem')
        sys.exit()#pass
    findBars(img,fn)
