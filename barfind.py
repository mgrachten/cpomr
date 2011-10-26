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

def findSystemLRNew(v):
    vp = v.copy()
    # margin is proportion of pagewidth from sides where dont look for system boundaries 
    vpmin,vpmax = nu.min(vp),nu.max(vp)
    N = vp.shape[0]
    margin = .05
    low = nu.sort(nu.nonzero(vp<(vpmin+(vpmax-vpmin)*margin))[0])
    vp[:low[0]] = vpmin
    vp[low[-1]:] = vpmin
    #vp[:int(N*margin)] = vpmin
    #vp[N-int(N*margin):] = vpmin
    vp = (vp-vpmin)/(vpmax-vpmin)-.5
    #vpmin,vpmax = nu.min(vp),nu.max(vp)

    K = N/70.
    wsys = signal.gaussian(K,K/5.0)
    wsys = nu.hstack((wsys,-wsys))
    nu.savetxt('/tmp/wsys.txt',wsys)
    wconv = nu.convolve(vp,wsys,'valid')
    sidx = nu.argsort(wconv)
    k = int((N-wconv.shape[0])/2)
    left = k+sidx[-1]+K/2.0
    right = k+sidx[0]-K/2.0
    #right = nu.argmin(wconv)+K/2.0
    wcon = nu.zeros(vp.shape[0])
    print(left,right)
    wcon[k:k+wconv.shape[0]] = wconv
    nu.savetxt('/tmp/n.txt',nu.column_stack((vp,wcon)))
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

#problematic:
"""
chopin:
6 3 2
7 2 2
17 3 1
17 4 3 !
18 1 3
18 1 9
"""
