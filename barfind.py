#!/usr/bin/env python

import sys,os
from scipy import signal
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern

def smooth(x,k):
    f = signal.blackmanharris(k)#nu.ones(k)/nu.float(k)
    return nu.convolve(x,f,'same')

def makeA440(fs,N):
    f = 100.0
    n = nu.linspace(0,N,N)
    return nu.sin(n*2*nu.pi*f/fs)

def findBars(img,fn):
    # smooth to get rid of barline peaks
    S = 30 # window size
    # typical nr of staffs per page
    K = 12
    hp = nu.sum(img,1)
    hp = smooth(hp,S)
    med = nu.median(hp)
    hp[hp<med] = med
    N = len(hp)
    nu.savetxt('/tmp/hp.txt',hp)
    k = N/nu.float(K)
    w = signal.gaussian(k,k/10.)/k
    c = nu.convolve(hp,w,'same')
    dc = nu.sign(nu.diff(c))
    d2c = nu.sign(nu.diff(dc))
    d2c = nu.insert(d2c,0,0)
    d2c = nu.insert(d2c,-1,0)
    #d2c = nu.convolve(d2c,nu.ones(S),'same')
    np = hp-nu.median(hp)
    dc2w = nu.zeros(np.shape)
    for i in nu.nonzero(d2c)[0]:
        dc2w[i] = nu.sum(np[i-S/2:i+S/2])
    idx = nu.argsort(dc2w)[::-1]
    #print(dc2w[idx][:20:2])
    minStaffs = 2
    maxStaffs = 30 
    #print(nu.diff(dc2w[idx][::2])[minSystems:maxSystems])
    sortedPeaks = dc2w[idx]
    diffSortedPeaks = nu.diff(sortedPeaks)
    thresholdPeak = minStaffs-1+nu.argmin(diffSortedPeaks[minStaffs-1:maxStaffs-1])
    epsilon = 1e-10
    nStaffs = thresholdPeak+1
    print('number of staffs found: {0}'.format(nStaffs))
    if nStaffs%2 != 0:
        print('warning: uneven number of staffs for {0}'.format(fn))
        if diffSortedPeaks[thresholdPeak-1] < diffSortedPeaks[thresholdPeak+1]:
            thresholdPeak -= 1
        else:
            thresholdPeak += 1
        nStaffs = thresholdPeak+1
        print('adjusted: found {0} staffs'.format(nStaffs))

    print('number of systems found: {0}'.format(nStaffs/2))

    threshold = sortedPeaks[thresholdPeak]-epsilon
    dc2w[dc2w <threshold] = 0
    im_r = 255-img
    im_g = 255-img
    for i in nu.nonzero(dc2w):
            im_g[i,:] = 0
            im_g[i-1,:] = 0
            im_g[i+1,:] = 0
    im_b = im_g
    writeImageData(os.path.join('/tmp',os.path.basename(fn)),img.shape,im_r,im_g,im_b)
    nu.savetxt('/tmp/sp.txt',sortedPeaks)
    nu.savetxt('/tmp/dsp.txt',diffSortedPeaks)
    nu.savetxt('/tmp/c.txt',nu.column_stack((np,c,dc2w)))
    #nSystems = nu.argmin(nu.diff(dc2w[idx][::2])[minSystems:maxSystems])
    #nStaffs = nu.argmax(nu.diff(dc2w[idx][minSystems*2:maxSystems*2]))
    #print((nStaffs+(minSystems-1)*2)/2)
    
    #nu.savetxt('/tmp/seven.txt',dc2w[idx][::2])
    #nu.savetxt('/tmp/seven.txt',nu.diff(dc2w[idx][::2]))

    #nu.savetxt('/tmp/c.txt',nu.column_stack((c,d2c,-d2c*np,np)))
    #nu.savetxt('/tmp/c.txt',nu.column_stack((c,d2c,dc2w,np)))
    #nu.savetxt('/tmp/dc.txt',dc)



if __name__ == '__main__':
    fn = sys.argv[1]
    try:
        img = 255-getImageData(fn)
        findBars(img,fn)
    except IOError: 
        pass
