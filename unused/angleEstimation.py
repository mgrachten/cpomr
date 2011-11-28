#!/usr/bin/env python

import sys
import numpy as nu
from scipy import signal
from stafffind import selectColumns
from imageUtil import normalize, smooth, findValleys
from scipy.stats import distributions

def findStaffs(r):
    vhist = nu.sum(r,1)
    N = len(vhist)
    typicalNrOfSystemPerPage = 6
    K = int(N/(2*typicalNrOfSystemPerPage))+1
    svhist = smooth(vhist,K)
    nu.savetxt('/tmp/vh.txt',nu.column_stack((vhist,svhist)))
    return findValleys(svhist)

def getOffset(v1,v2,dx,maxAngle):
    #kmax = min(70,int(1.5*nu.ceil(dx*nu.tan(maxAngle*nu.pi))))
    #print(kmax
    kmax = 50
    N = len(v1)
    assert N == len(v2)
    dotproducts = []
    rng = []
    for i in range(-kmax,kmax+1):
        b = min(max(0,i),N)
        e = max(0,min(N,N-i))
        if e > 0:
            vv1 = v1[:e]
            vv2 = v2[b:]
        #print(len(vv1),len(vv2),b,e)
            dp = nu.dot(vv1,vv2)
            dotproducts.append(dp)
            rng.append(i)
    ndp = normalize(nu.array(dotproducts))
    return ndp, rng[nu.argmax(ndp)]

def angleEstimator(img):
    # horizontal projection
    # smooth
    #img = 127
    N,M = img.shape
    s = nu.append(findStaffs(img),N)
    start = 0
    # the range for which we look for angles: [-maxAngle,maxAngle]
    maxAngle = 2/180.
    histrange = (-maxAngle*180,maxAngle*180)
    hbins = 600
    gangles = []
    shists = []
    for i,end in enumerate(s):
        print('row segment',i)
        print(start,end)
        vsums = nu.sum(img[start:end,:],0)
        c,b = selectColumns(vsums,3)
        if len(c) < 2:
            print('empty')
            continue
        angles = []
        i1 = int(0*len(c)/10)
        i2 = int(2*len(c)/10)
        #for j in range(0,len(c)-1):
        for j in range(i1,i2):
            dx = c[j]-c[j+1]
            dps,dy = getOffset(img[start:end,c[j]],img[start:end,c[j+1]],nu.abs(dx),maxAngle)
            if nu.min(dps) < 0.5: # min is only > 0 with zero range
                angle = (nu.arctan2(dy,dx)/nu.pi+.5)%1-.5
                angles.append(angle)
        angles = 180*nu.array(angles)
        print(len(angles),'points')
        bins,lims = nu.histogram(angles,bins=hbins,range=histrange)
        nu.savetxt('/tmp/a{0}.txt'.format(i),angles)
        gangles.append(angles.reshape((-1,1)))
        nbins = nu.array([bins[k]-nu.mean(bins[max(0,k-25):min(len(bins),k+25)]) for k in range(len(bins))])
        nu.savetxt('/tmp/angles{0}.txt'.format(i),nu.column_stack((.5*(lims[1:]+lims[:-1]),bins,nbins)))
        shists.append(bins)
        start = end
    bins,lims = nu.histogram(nu.vstack(gangles),bins=hbins,range=histrange)
    angles = (lims[1:]+lims[:-1])/2.0
    nu.savetxt('/tmp/gh.txt',nu.column_stack((angles,bins)))
    sbins = smooth(bins,50)
    amax = angles[nu.argmax(sbins)]
    print(nu.argmax(sbins))
    print(amax)
    weights = distributions.norm(amax,.5).pdf(angles)
    nu.savetxt('/tmp/w0.txt',weights)
    nu.savetxt('/tmp/gangles.txt',nu.column_stack((.5*(lims[1:]+lims[:-1]),bins,sbins,weights)))
    for i,shist in enumerate(shists):
        nu.savetxt('/tmp/b{0}.txt'.format(i),shist*weights)
        print(i,nu.argmax(shist*weights),angles[nu.argmax(shist*weights)])


if __name__ == '__main__':
    pass #x = sys.argv[1]
