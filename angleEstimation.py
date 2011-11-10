#!/usr/bin/env python

import sys
import numpy as nu
from scipy import signal
from stafffind import selectColumns
from imageUtil import normalize

def smooth(x,k):
    return nu.convolve(x,signal.hanning(k),'same')

def findPeaks(v):
    """find the peaks in a smooth curve
    """
    x = nu.zeros(len(v))
    x[2:len(x)-2] = nu.diff(nu.diff(nu.sign(nu.diff(v)),2))
    x[x<2] = 0
    peaks = nu.nonzero(x)[0]
    return peaks

def findValleys(v):
    """find the valleys in a smooth curve
    """
    x = nu.zeros(len(v))
    sdx = nu.sign(nu.diff(v))
    nz = nu.nonzero(sdx)[0]
    valleybegins = nu.nonzero(nu.diff(sdx[nz]) > 0)[0]
    valleys = .5+(nz[valleybegins]+nz[valleybegins+1])/2.0
    return valleys.astype(nu.int)

def findSystems(r):
    vhist = nu.sum(r,1)
    N = len(vhist)
    typicalNrOfSystemPerPage = 6
    K = int(N/(2*typicalNrOfSystemPerPage))+1
    #nu.savetxt('/tmp/vh.txt',vhist)
    vhist = smooth(vhist,K)
    #nu.savetxt('/tmp/vhs.txt',vhist)
    return findValleys(vhist)


def getOffset(v1,v2,kmax=50):
    N = len(v1)
    assert N == len(v2)
    dotproducts = []
    rng = range(-kmax,kmax+1)
    for i in rng:
        b = max(0,i)
        e = min(N,N-i)
        vv1 = v1[:e]
        vv2 = v2[b:]
        dp = nu.dot(vv1,vv2)/float(len(vv1))
        dotproducts.append(dp)
    ndp = normalize(nu.array(dotproducts))
    return ndp, rng[nu.argmax(ndp)]

def angleEstimator(img):
    # horizontal projection
    # smooth
    N,M = img.shape
    s = nu.append(findSystems(img),M)
    start = 0
    for i,end in enumerate(s):
        if False: #i != 1:
            continue
        print('row segment',i)
        print(start,end)
        vsums = nu.sum(img[start:end,:],0)
        c,b = selectColumns(vsums,2)
        if len(c) < 2:
            print('empty')
            continue
        angles = []
        for j in range(len(c)-1):
        #for j in range(int(len(c)/2)):
            dps,m = getOffset(img[start:end,c[j]],img[start:end,c[j+1]])
            if nu.min(dps) > 0:
                pass #print(c[j],c[j+1],None)
            else:
                dx = c[j]-c[j+1]
                dy = m
                angle = (nu.arctan2(dy,dx)/nu.pi+.5)%1-.5
                #print(c[j],c[j+1],m,180*angle)
                angles.append(angle)
                #nu.savetxt('/tmp/c{0}-{1}.txt'.format(c[j],c[j+1]),img[start:end,(c[j],c[j+1])])
                #nu.savetxt('/tmp/c{0}-{1}a.txt'.format(c[j],c[j+1]),dps)
        bins,lims = nu.histogram(180*nu.array(angles),1000)
        #sbins = smooth(bins,5)
        nu.savetxt('/tmp/angles{0}.txt'.format(i),nu.column_stack((.5*(lims[1:]+lims[:-1]),bins,bins)))
        maxi = nu.argmax(bins)
        print(nu.median(180*nu.array(angles)),(lims[maxi]+lims[maxi+1])/2.0)
        start = end
    # find local minima
    
    # split at local minima
    
    # select Col

if __name__ == '__main__':
    pass #x = sys.argv[1]
