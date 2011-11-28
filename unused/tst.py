#!/usr/bin/env python

import sys,os
import numpy as nu
from scipy import signal,cluster,stats
from numpy.fft import fft2, ifft2

from imageUtil import getImageData, writeImageDataOldOld, makeMask, normalize
from scoreVocabulary import makeVocabulary

def convolve(image1, image2, pad=True):
    """
    Not so simple convolution 
    source: http://www.rzuser.uni-heidelberg.de/~ge6/Programing/convolution.html
    """
    #The size of the images:
    r1,c1 = image1.shape
    r2,c2 = image2.shape

    r = r1+r2
    c = c1+c2

    #For nice FFT, we need the power of 2:
    if pad:
        pr2 = int(nu.log(r)/nu.log(2.0) + 1.0 )
        pc2 = int(nu.log(c)/nu.log(2.0) + 1.0 )
        rOrig = r
        cOrig = c
        r = 2**pr2
        c = 2**pc2
    #end of if pad

    #numpy fft has the padding built in, which can save us some steps
    #here. The thing is the s(hape) parameter:
    #fftimage = FFt(image1,s=(r,c)) * FFt(image2,s=(r,c))
    fftimage = fft2(image1, s=(r,c))*fft2(image2[::-1,::-1],s=(r,c))

    if pad:
        return (ifft2(fftimage))[:rOrig,:cOrig].real
    else:
        return (ifft2(fftimage)).real

def getCoords(img,patfile,outfile):
    pat = getImageData(patfile)-.5
    #pat = preprocessPattern(pat -.5,os.path.basename(patfile))
    pat = pat*makeMask(pat)
    r = convolve(img,pat)
    N,M = img.shape
    K,L = r.shape
    w = int(nu.round((K-N)/2))
    h = int(nu.round((L-M)/2))
    r = r[w:w+N,h:h+M]
    imax = nu.max(r)
    imin = nu.min(r)
    r = (r-imin)/(imax-imin)
    expectedNrOfInstances = 10.0
    percentile = (1-expectedNrOfInstances/r.size)
    thr = stats.mstats.mquantiles(r,[percentile])[0]
    r[r < thr] = .2*r[r < thr]
    nu.savetxt(outfile+'.txt',r)
    writeImageDataOldOld(outfile,nu.array(r*255,nu.uint8))

def smooth(x,k):
    return nu.convolve(x,signal.hanning(k),'same')

if __name__ == '__main__':
    imgfile = sys.argv[1]
    img = getImageData(imgfile)
    img = img -.5
    patfile = sys.argv[2]
    outfile = os.path.join('/tmp/',os.path.basename(imgfile))
    getCoords(img,patfile,outfile)
    
