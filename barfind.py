#!/usr/bin/env python

import sys,os
from scipy import signal,cluster
import numpy as nu
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import argpartition, partition
from main import convolve

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

def findSystemLRNew(v):
    vp = v.copy()
    # margin is proportion of pagewidth from sides where dont look for system boundaries 
    vpmed = nu.median(vp)
    vpmin,vpmax = nu.min(vp),nu.max(vp)
    print(vpmed)
    print(vpmin,vpmax)
    N = vp.shape[0]
    margin = .05
    lowidx = vp<(vpmin+(vpmed-vpmin)*margin)
    low = nu.sort(nu.nonzero(lowidx)[0])
    lidx = low<N/2
    vp[:int(nu.median(low[lidx]))] = vpmin
    vp[int(nu.median(low[nu.logical_not(lidx)])):] = vpmin
    vpmax = nu.max(vp)
    
    vp = normalize(vp)
    vpmed = nu.median(vp)
    vp = vp-vpmed/2.0

    K = N/60.
    wsys = signal.gaussian(K,K/5.0)
    wsys = nu.hstack((wsys,-wsys))
    nu.savetxt('/tmp/wsys.txt',wsys)
    wconv = nu.convolve(vp,wsys,'valid')
    #sidx = nu.argsort(wconv)
    k = int((N-wconv.shape[0])/2)
    left = k+nu.argmax(wconv[:int(N/2)])+K/2.0
    right = k+N/2+nu.argmin(wconv[int(N/2):])-K/2.0
    #k+sidx[0]-K/2.0
    #right = nu.argmin(wconv)+K/2.0
    wcon = nu.zeros(vp.shape[0])
    print(left,right)
    wcon[k:k+wconv.shape[0]] = wconv
    nu.savetxt('/tmp/n.txt',nu.column_stack((v,vp,wcon)))
    return int(left),int(right)

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
    # S: window size
    S = N/100 
    # K: nr of systems
    K = 4
    # impact of prior for different system pdfs to be multiplied
    prior = .5

    hp = makeHZ(img)
    vp = nu.sum(img,0)
    vp = vp-.5*nu.median(vp)
    sl,sr = findSystemLRNew(vp)
    # TODO make sure the selection doesn't give index errors in pathetic cases:
    borderW = 20
    leftpdf = nu.sum(img[:,sl-borderW:sl+borderW],1)
    rightpdf = nu.sum(img[:,sr-borderW:sr+borderW],1)

    horzpdf = nu.log(addPrior(hp,proportion=prior))
    leftpdf = nu.log(addPrior(leftpdf,proportion=prior))
    rightpdf = nu.log(addPrior(rightpdf,proportion=prior))
    hp = horzpdf*leftpdf*rightpdf
    hp = hp-nu.mean(hp)
    nu.savetxt('/tmp/hp.txt',nu.column_stack((horzpdf,leftpdf,rightpdf,horzpdf*leftpdf*rightpdf)))
    # smooth curve to find systems
    k = N/nu.float(K)
    w = signal.gaussian(k,k/8.)/k
    c = nu.convolve(smooth(hp,S),w,'same')
    np = hp-nu.median(hp)
    systemCenters = getIdxOfMaxima(c)
    avgSystemDist = nu.median(nu.diff(nu.sort(systemCenters)))

    #wsys = signal.gaussian(avgSystemDist/5,avgSystemDist/5)/k
    # TESTING: REMOVED SPURIOUS k
    wsys = signal.gaussian(avgSystemDist/5.,avgSystemDist/5.)
    wsys = nu.hstack((wsys,-wsys))
    cwtop = nu.convolve(hp,wsys,'same')
    cwbot = nu.convolve(hp,-wsys,'same')
    limits = findVertSystemLimits(systemCenters,cwbot,cwtop)
    #sys.exit()
    # save image
    im_r = 255-img
    im_g = 255-img
    for top,bot in limits:
            im_g[top,:] = 20
            im_r[top,:] = 50
            im_r[bot,:] = 50
            im_g[bot,:] = 20
            c = int(nu.round((top+bot)/2.))
            im_g[c,:] = 20
            im_r[c,:] = 20
    im_b = im_g
    writeImageData(os.path.join('/tmp',os.path.basename(fn).replace('.tif','.png')),img.shape,im_r,im_g,im_b)

def findBarsTry(img,fn):
    N,M = img.shape
    r = nu.zeros(N)
    sums = nu.sum(img,0)
    prior = .1
    for i in range(M):
        print(i,M)
        if sums[i] > 0:
            #r += nu.log(prior/N+(1-prior)*img[:,i]/sums[i])
            r += prior/N+(1-prior)*img[:,i]/sums[i]
    nu.savetxt(fn,normalize(r))


def smooth(x,k):
    smoothingkernel = (nu.ones(k)*k)**-1
    return nu.convolve(x,smoothingkernel,'same')

def downSample(d,k):
    """Downsample x by smoothing plus linear interpolation
    d -- a NxM matrix
    k -- downsampling parameter: the result will have N/2**k rows
    """
    xsmooth = d.copy()
    N,M = xsmooth.shape
    for i in range(1,M):
        xsmooth[:,i] = smooth(d[:,i],2**k)
    assert(N > 1)
    idx = nu.linspace(0,N-1,int(N/float(2**k)))
    origIdx = nu.arange(N)
    d = [interpolate.interp1d(origIdx,xsmooth[:,x])(idx) for x in range(M)]
    return nu.column_stack(d)


if __name__ == '__main__':
    fn = sys.argv[1]
    try:
        img = 255-getPattern(fn,False,False)
        print(nu.max(img))
    except IOError as e: 
        print('problem')
        raise e
        sys.exit()#pass

    findBarsTry(img,'/tmp/p.txt')
    N,M = img.shape
    K = N/3
    w = signal.gaussian(K,K/4.0).reshape((1,-1))
    nu.savetxt('/tmp/w.txt',w[0,:])
    img1 = convolve(img,w)
    img1[img1 < 0 ] = 0
    img1 = 1/(1+nu.exp(-100*(normalize(img1)-.5)))
    
    nu.savetxt('/tmp/c.txt',img1)
    findBarsTry(img1,'/tmp/p1.txt')

#problematic:
"""
chopin:
7 5 1
18 1 1
35 2 7

"""
