#!/usr/bin/env python

import sys
from PIL import Image
import numpy as nu
from scipy import signal,cluster
from numpy.fft import fft2, ifft2
from multiprocessing import Pool

from utilities import argpartition 


def getImageData(filename):
    imageFh = Image.open(filename)
    #data = nu.array(list(imageFh.getdata()))
    s = list(imageFh.size)
    s.reverse()
    data = nu.array(imageFh.getdata()).reshape(tuple(s))
    img_min = nu.min(data)
    img_max = nu.max(data)
    return 1-(nu.array(data,nu.float)-img_min)/(img_max-img_min)

def writeImageData(filename,data,size=None):
    """
    Write the value of numpy array data as an image to filename. If
    size is not specified, the shape of the array is taken to
    determine the image size. The file encoding is guessed from the
    filename (extension). The image is greyscale.
    """
    if size == None:
        size = list(data.shape)+[1]*(2-len(list(data.shape)))
        size.reverse()
    data.resize((data.size,))
    imageFh = Image.new('L',size)
    imageFh.putdata(data)
    imageFh.save(filename)
    return True


def convolve(image1, image2, MinPad=True, pad=True):
    """
    Not so simple convolution 
    source: http://www.rzuser.uni-heidelberg.de/~ge6/Programing/convolution.html
    """
    #Just for comfort:
    FFt = fft2
    iFFt = ifft2

    #The size of the images:
    r1,c1 = image1.shape
    r2,c2 = image2.shape

    #MinPad results simpler padding,smaller images:
    if MinPad:
        r = r1+r2
        c = c1+c2
    else:
        #if the Numerical Recipies says so:
        r = 2*max(r1,r2)
        c = 2*max(c1,c2)

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
    fftimage = FFt(image1, s=(r,c))*FFt(image2[::-1,::-1],s=(r,c))

    if pad:
        return (iFFt(fftimage))[:rOrig,:cOrig].real
    else:
        return (iFFt(fftimage)).real


def preprocessPattern(bar):
    N,M = bar.shape
    n = nu.arange(N)
    n = n*n[::-1]
    m = nu.arange(M)
    m = m*m[::-1]
    o,p = nu.meshgrid(n,m)
    return bar*o.T*p.T

def getCoords(img,pat,thr=.9):
    r = convolve(img,pat,True,True)
    N,M = img.shape
    K,L = r.shape
    w = (K-N)/2
    h = (L-M)/2
    r = r[w:w+N,h:h+M]
    imax = nu.max(r)
    imin = nu.min(r)
    r = (r-imin)/(imax-imin)
    r[r < thr] = 0
    #nu.savetxt('/tmp/r',r[::-1,:])
    return r,nu.column_stack(nu.nonzero(r))

def getCoordsNew(patfile,img,thr=.9):
    pat = getImageData(patfile)
    pat = preprocessPattern(pat -.5)
    r = convolve(img,pat,True,True)
    N,M = img.shape
    K,L = r.shape
    w = (K-N)/2
    h = (L-M)/2
    r = r[w:w+N,h:h+M]
    imax = nu.max(r)
    imin = nu.min(r)
    r = (r-imin)/(imax-imin)
    r[r < thr] = 0
    #nu.savetxt('/tmp/r',r[::-1,:])
    return r,nu.column_stack(nu.nonzero(r))

def smooth(x,k):
    return nu.convolve(x,signal.hanning(k),'same')

def findPeaks(v):
    """find the peaks in a smooth curve
    """
    x = nu.zeros(len(v))
    x[2:len(x)-2] = nu.diff(nu.diff(nu.sign(nu.diff(v)),2))
    x[x<2] = 0
    peaks = nu.nonzero(x)[0]
    if True:
        return peaks
    # superfluous
    if len(peaks) < 2:
        return peaks
    w = nu.mean(nu.diff(peaks))*.5
    w = nu.int(2*(w/2)+1)
    hw = int(nu.floor(w/2))
    y = x[:]
    for p in peaks:
        y[p-hw:p+hw+1] = .5
    nu.savetxt('/tmp/s.txt',y)
    nu.savetxt('/tmp/v.txt',v)
    nu.savetxt('/tmp/p.txt',x)
    return peaks#x,w

def findSystems(r):
    vhist = nu.sum(r,1)
    N = len(vhist)
    typicalNrOfSystemPerPage = 6
    K = int(N/(2*typicalNrOfSystemPerPage))+1
    vhist = smooth(vhist,K)
    return findPeaks(vhist)
#nu.savetxt('/tmp/peaks.txt',findPeaks(vhist))
#nu.savetxt('/tmp/vhist.txt',vhist)

def findBars(peaks,rbar):
    img = nu.zeros(rbar.shape,nu.uint)
    w = nu.mean(nu.diff(peaks))*.5
    bar_hcoords = []
    for i,peak in enumerate(peaks):
        hhist = nu.sum(rbar[peak-int(w/2):peak+int(w/2),:],0)
        typicalNrOfBarsPerSystem = 5
        hhist = smooth(hhist,typicalNrOfBarsPerSystem)
        barpeaks = findPeaks(hhist)
        barpeaks.sort()
        bar_hcoords.append(barpeaks)
        for bp in barpeaks:
            img[peak,bp] = 256
    writeImageData('/tmp/bars.png',img)
    return bar_hcoords
    #w = nu.int(2*(w/2)+1)
    #hw = int(nu.floor(w/2))

def clusterInBar(coords,x1,x2,y1,y2):
    inbar = coords[nu.logical_and(nu.logical_and(coords[:,0]>x1,coords[:,0]<x2),
                          nu.logical_and(coords[:,1]>y1,coords[:,1]<y2)),:]
    
    if inbar.shape[0] == 0:
        return None
    l = cluster.hierarchy.linkage(inbar)
    c = cluster.hierarchy.fcluster(l,2,criterion='distance')
    idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
    patternLocations = nu.empty((len(idict),2))
    for i,(k,v) in enumerate(idict.items()):
        patternLocations[i,:] = nu.mean(inbar[tuple(v),:],0)
    return nu.array(patternLocations,nu.int)

if __name__ == '__main__':
    global img
    pool = Pool()
    barfile = sys.argv[1]
    ppatfile = sys.argv[2]
    imgfile = sys.argv[3]
    img = getImageData(imgfile)
    img = img -.5
    #bar = getImageData(barfile)
    #bar = preprocessPattern(bar -.5)
    #ppat = getImageData(ppatfile)
    #ppat = preprocessPattern(ppat -.5)

    # find system/bar coordinates
    #rbar,cbar = getCoordsNew(barfile,shm,thr=.75)
    barresult = pool.apply_async(getCoordsNew,(barfile,img,.75))
    #pat,globalPatCoords = getCoords(ppatfile,shm,thr=.8)
    presult = pool.apply_async(getCoordsNew,(ppatfile,img,.8))
    aresult = pool.apply_async(getCoordsNew,(ppatfile,img,.8))
    bresult = pool.apply_async(getCoordsNew,(ppatfile,img,.8))
    cresult = pool.apply_async(getCoordsNew,(ppatfile,img,.8))
    dresult = pool.apply_async(getCoordsNew,(ppatfile,img,.8))
    eresult = pool.apply_async(getCoordsNew,(ppatfile,img,.8))
    fresult = pool.apply_async(getCoordsNew,(ppatfile,img,.8))
    gresult = pool.apply_async(getCoordsNew,(ppatfile,img,.8))

    pool.close()
    pool.join()
    
    rbar,cbar = barresult.get()
    pat,globalPatCoords = presult.get()

    system_vcoords = findSystems(rbar)
    assert len(system_vcoords) > 0
    system_vcoords.sort()
    bar_hcoords = findBars(system_vcoords,rbar)
    
    # determine widths of systems
    assert len(system_vcoords) > 1
    system_vcoords = nu.array(system_vcoords,nu.float)
    sbounds = (system_vcoords[1:]+system_vcoords[:-1])/2
    sbounds = nu.insert(sbounds,0,2*system_vcoords[0]-sbounds[0])
    sbounds = nu.append(sbounds,2*system_vcoords[-1]-sbounds[-1])
    globalpat = nu.zeros(img.shape,nu.float)



    for i,s_vc in enumerate(system_vcoords):
        b_hcs = bar_hcoords[i]
        systemwidth = b_hcs[-1]-b_hcs[0]
        barwidths = nu.diff(b_hcs)
        minWidth = .1*systemwidth
        # filter out unlikily short bars
        baridx = nu.arange(len(barwidths))[barwidths >= minWidth]
        assert len(b_hcs) > 1
        for j in baridx:
            x1,x2 = sbounds[i],sbounds[i+1]
            y1,y2 = b_hcs[j],b_hcs[j+1]
            rpad = clusterInBar(globalPatCoords,x1,x2,y1,y2)
            if rpad is not None:
                for k in rpad:
                    globalpat[k[0],k[1]] = 1
            globalpat[x1:x2,y1] = .5
            globalpat[x1:x2,y2] = .5
            globalpat[x1,y1:y2] = .5
            globalpat[x2,y1:y2] = .5
    writeImageData('/tmp/pat.png',nu.array(globalpat*256,nu.int))



