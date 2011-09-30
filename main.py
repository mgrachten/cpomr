#!/usr/bin/env python

import sys,os
import numpy as nu
from scipy import signal,cluster
from numpy.fft import fft2, ifft2
from multiprocessing import Pool

from utilities import argpartition, FakePool
from imageUtil import getImageData, writeImageData, makeMask, normalize
from scoreVocabulary import makeVocabulary

def convolve(image1, image2, MinPad=True, pad=True):
    """
    Not so simple convolution 
    source: http://www.rzuser.uni-heidelberg.de/~ge6/Programing/convolution.html
    """
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
    fftimage = fft2(image1, s=(r,c))*fft2(image2[::-1,::-1],s=(r,c))

    if pad:
        return (ifft2(fftimage))[:rOrig,:cOrig].real
    else:
        return (ifft2(fftimage)).real

def getCoords(img,patfile,thr=.9,returnImage=False,returnCoords=True):
    pat = getImageData(patfile)-.5
    #pat = preprocessPattern(pat -.5,os.path.basename(patfile))
    pat = pat*makeMask(pat)
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
    if returnCoords and returnImage:
        return r, nu.column_stack(nu.nonzero(r))
    elif returnCoords:
        coords = nu.column_stack(nu.nonzero(r))
        del r
        return coords
    elif returnImage:
        return r
    else:
        del r
        return None


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

def findSystems(r):
    vhist = nu.sum(r,1)
    N = len(vhist)
    typicalNrOfSystemPerPage = 6
    K = int(N/(2*typicalNrOfSystemPerPage))+1
    vhist = smooth(vhist,K)
    return findPeaks(vhist)

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
    return bar_hcoords

def clusterInBar(coords,x1,x2,y1,y2):
    margin = 5
    inbar = coords[nu.logical_and(nu.logical_and(coords[:,0]>x1+margin,coords[:,0]<x2-margin),
                          nu.logical_and(coords[:,1]>y1+margin,coords[:,1]<y2-margin)),:]
    
    if inbar.shape[0] == 0:
        return None
    if inbar.shape[0] == 1:
        return inbar
    l = cluster.hierarchy.linkage(inbar)
    c = cluster.hierarchy.fcluster(l,2,criterion='distance')
    idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
    patternLocations = nu.empty((len(idict),2))
    for i,(k,v) in enumerate(idict.items()):
        patternLocations[i,:] = nu.mean(inbar[tuple(v),:],0)
    return nu.array(patternLocations,nu.int)

class Bar(object):
    def __init__(self,vcenter,left,right,top,bottom):
        self.vcenter = vcenter
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
    def drawOnImage(self,img):
        
def getBarBBs(barItem):
    barImage = nu.zeros(img.shape)
    for barPatternFile in barItem.getFiles():
        barImg += getCoords(img,barPatternFile,.8,True,False)

    print('finding systems...')
    barImage = barImage/len(barItem.getFiles())
    system_vcoords = findSystems(barImage)
    assert len(system_vcoords) > 0
    system_vcoords.sort()
    print('finding bars...')
    bar_hcoords = findBars(system_vcoords,barImage)
    del barImage

    validSystemIdx = []
    for i,v in enumerate(system_vcoords):
        if len(bar_hcoords[i]) >= 2:
            validSystemIdx.append(i)
    validSystemIdx = nu.array(validSystemIdx,nu.uint)
    system_vcoords = system_vcoords[validSystemIdx]
    bar_hcoords = [bar_hcoords[x] for x in validSystemIdx]
    # determine widths of systems
    system_vcoords = nu.array(system_vcoords,nu.float)
    sbounds = (system_vcoords[1:]+system_vcoords[:-1])/2
    sbounds = nu.insert(sbounds,0,2*system_vcoords[0]-1.5*sbounds[0])
    sbounds = nu.append(sbounds,2*system_vcoords[-1]-1.5*sbounds[-1])

    return system_vcoords, bar_hcoords

def processPage(img,vocabulary,pool,outname):
    barItem = vocabulary.getBar()
    if barItem is None:
        return False


    vocresults = []
    voclabels = vocabulary.getLabels()
    for label in voclabels:
        vi = vocabulary.getItem(label)
        patternFile = vi.getFiles()[0]
        thr = vi.getThresholds()[0]
        vocresults.append(pool.apply_async(getCoords,(img,patternFile,thr,False,True)))

    pool.close()
    pool.join()

    assert len(system_vcoords) > 1


    globalpat = nu.zeros(img.shape,nu.float)
    
    patternCoords = []
    for vr in vocresults:
        print('waiting for pattern results...')
        patcoords = vr.get()
        patternCoords.append(patcoords)

    print('allocating patterns to bars...')
    for i,s_vc in enumerate(system_vcoords):
        b_hcs = bar_hcoords[i]
        if len(b_hcs) < 1:
            continue
        systemwidth = b_hcs[-1]-b_hcs[0]
        barwidths = nu.diff(b_hcs)
        minWidth = .0*systemwidth
        # filter out unlikily short bars
        baridx = nu.arange(len(barwidths))[barwidths >= minWidth]
        assert len(b_hcs) > 1
        for j in baridx:
            x1,x2 = sbounds[i],sbounds[i+1]
            y1,y2 = b_hcs[j],b_hcs[j+1]
            for q,pc in enumerate(patternCoords):
                #print('pc',q,len(pc))
                rpad = clusterInBar(pc,x1,x2,y1,y2)
                if rpad is not None:
                    vi = vocabulary.getItem(voclabels[q])
                    viimg = vi.getImage()
                    viw,vih = viimg.shape
                    for k in rpad:
                        r = 2
                        #globalpat[k[0]-r:k[0]+r,k[1]-r:k[1]+r] = 1
                        globalpat[k[0]-int(viw/2):k[0]-int(viw/2)+viw,
                                  k[1]-int(vih/2):k[1]-int(vih/2)+vih] = viimg
            globalpat[x1:x2,y1] = 1
            globalpat[x1:x2,y2] = 1
            globalpat[x1,y1:y2] = 1
            globalpat[x2,y1:y2] = 1
    #alphaChannel = nu.array(nu.array(globalpat,nu.bool)*200,nu.uint8)
    #alphaChannel = nu.array(globalpat*200,nu.uint8)
    #alphaChannel = nu.zeros(globalpat.shape,nu.uint8)+0
    nPageImg = normalize(img)
    im_r = nu.minimum(nPageImg,1-.5*globalpat)
    im_r = nu.array((1-im_r)*255,nu.uint8)
    im_g = nu.maximum(nPageImg,globalpat)
    del nPageImg
    im_g = nu.array((1-im_g)*255,nu.uint8)
    im_b = im_g
    print(nu.min(im_g),nu.max(im_g))
    writeImageData(outname,img.shape,im_r,im_g,im_b)
                         

if __name__ == '__main__':
    #pool = Pool()
    pool = FakePool()
    vocabularyDir = './vocabularies/dme-4096'
    vocabulary = makeVocabulary(vocabularyDir)
    imgfile = sys.argv[1]
    img = getImageData(imgfile)
    img = img -.5
    #print(img[10,10])
    #sys.exit()
    outname= os.path.join('/tmp/',os.path.basename(imgfile))
    processPage(img,vocabulary,pool,outname)
    pool.close()
    pool.join()
