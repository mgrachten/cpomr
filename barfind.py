#!/usr/bin/env python

import sys,os
from scipy import signal,cluster
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition

def smooth(x,k):
    f = signal.blackmanharris(k)#nu.ones(k)/nu.float(k)
    return nu.convolve(x,f,'same')

def makeHZ(img):
    hl = nu.zeros(img.shape[0])
    nzz = [nu.nonzero(img[i,:])[0] for i in range(0,img.shape[0])]
    for i in range(1,img.shape[0]-1):
        # sum nonzero pixels of three consecutive rows (to accommodate for angle))
        nz = nu.array(list(set(nu.hstack((nzz[i-1],nzz[i],nzz[i+1])))))
        nz = nu.sort(nz)
        # dz: the longest horizontal white space within the 3-pixel band
        dz = nu.diff(nz)
        dz[dz == 1] = 0
        # nz are the places where whitespace occurs
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

def findSystemLR(vp):
    # margin is proportion of pagewidth from sides where dont look for system boundaries 
    margin = .02
    # windowsize is the size of the step window used to detect the sides of the systems
    windowsize = 1/70.
    N = vp.shape[0]
    W = 2*int(nu.round((N*windowsize)/2.))
    w = nu.zeros(W)
    w[W/2:] = 1
    w = w-.5
    wl = nu.convolve(vp,w,'same')
    wr = nu.convolve(vp,w[::-1],'same')
    weight = nu.linspace(0,1,N)**2
    wlr = (1-weight)*wr+weight*wl
    pidx = getIdxOfMaxima(wlr)
    pidx = pidx[nu.logical_and(pidx > margin*N,pidx < N-N*margin)]
    nu.savetxt('/tmp/tst.txt',nu.column_stack((pidx,wlr[pidx])))
    nu.savetxt('/tmp/tst1.txt',wlr)
    sp = nu.argsort(wlr[pidx])[-2:]
    sp = nu.sort(sp)
    #print(pidx[sp[0]],pidx[sp[1]])
    return pidx[sp[0]],pidx[sp[1]]

def addPrior(pdf,proportion,prior=None):
    if prior == None:
        prior = nu.ones(pdf.shape[0])
    assert prior.shape[0] == pdf.shape[0]
    pdfsum = nu.sum(pdf)
    priorsum = nu.sum(prior)
    pdfFactor = (1-proportion)*pdf/pdfsum
    priorFactor = proportion*prior/priorsum
    return pdfFactor+priorFactor

def getIdxOfMaxima(v,sort=False):
    d2v = nu.diff(nu.sign(nu.diff(v)))
    idx = nu.nonzero(d2v == -2)[0]+1
    if sort:
        return idx[nu.argsort(v[idx])[::-1]]
    else:
        return idx

def findVertSystemLimits(centers,bottomCurve,topCurve):
    N = centers.shape[0]
    assert N > 0
    i = 0
    start = 0
    end = centers[i]
    limits = [[nu.argmax(topCurve[start:end])+start]]
    while i < N-1:
        i +=1
        start = end
        end = centers[i]
        limits[-1].append(nu.argmax(bottomCurve[start:end])+start)
        limits.append([nu.argmax(topCurve[start:end])+start])
    start = end
    limits[-1].append(nu.argmax(bottomCurve[start:])+start)
    return limits

def findBars(img,fn):
    # smooth to get rid of barline peaks
    # typical nr of staffs per page
    N = img.shape[0]
    S = N/100 # window size
    # K: nr of systems
    K = 4
    #hp = nu.sum(img,1)
    hp = makeHZ(img)
    vp = nu.sum(img,0)
    vp = vp-.5*nu.median(vp)
    sl,sr = findSystemLR(vp)
    #vv = nu.column_stack((vp,w))
    leftpdf = nu.sum(img[:,sl-10:sl+10],1)
    rightpdf = nu.sum(img[:,sr-10:sr+10],1)
    nu.savetxt('/tmp/sl.txt',leftpdf)
    nu.savetxt('/tmp/sr.txt',rightpdf)
    #hp = smooth(hp,S)
    med = nu.median(hp)
    hp[hp<med] = med

    prior = .5
    horzpdf = nu.log(addPrior(hp,proportion=.5))
    leftpdf = nu.log(addPrior(leftpdf,proportion=.5))
    rightpdf = nu.log(addPrior(rightpdf,proportion=.5))
    hp = horzpdf*leftpdf*rightpdf
    hp = hp-nu.mean(hp)
    nu.savetxt('/tmp/hp.txt',nu.column_stack((horzpdf,leftpdf,rightpdf,horzpdf*leftpdf*rightpdf)))
    # smooth curve to find systems
    k = N/nu.float(K)
    w = signal.gaussian(k,k/8.)/k
    c = nu.convolve(smooth(hp,S),w,'same')
    np = hp-nu.median(hp)
    dc2w = nu.zeros(np.shape)
    systemCenters = getIdxOfMaxima(c)
    avgSystemDist = nu.median(nu.diff(nu.sort(systemCenters)))
    for i in systemCenters:
        dc2w[i] = nu.sum(nu.abs(nu.diff(np[i-N/(2*35):i+N/(2*35)])))
    dc2w[dc2w<0] = 0
    nu.savetxt('/tmp/dc2w.txt',dc2w)
    #pd = pair(nu.nonzero(dc2w)[0],dc2w)

    wsys = signal.gaussian(avgSystemDist/5,avgSystemDist/5)/k
    print(wsys.shape)
    wsys = nu.hstack((wsys,-wsys))
    print(wsys.shape)
    cwtop = nu.convolve(hp,wsys,'same')
    cwbot = nu.convolve(hp,-wsys,'same')
    limits = findVertSystemLimits(systemCenters,cwbot,cwtop)

    im_r = 255-img
    im_g = 255-img
    for top,bot in limits:
            im_g[top,:] = 100
            im_r[bot,:] = 100
            c = int(nu.round((top+bot)/2.))
            im_g[c,:] = 100
            im_r[c,:] = 255
    im_b = im_g
    writeImageData(os.path.join('/tmp',os.path.basename(fn).replace('.tif','.png')),img.shape,im_r,im_g,im_b)
    sys.exit()

    nu.savetxt('/tmp/cw.txt',nu.column_stack((cwbot,cwtop)))
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
        img = 255-getPattern(fn,False,False)
        print(nu.max(img))
    except IOError as e: 
        print('problem')
        raise e
        sys.exit()#pass
    findBars(img,fn)
