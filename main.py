#!/usr/bin/env python

import sys,os,resource
import numpy as nu
from scipy import signal,cluster,stats
from numpy.fft import fft2, ifft2
from multiprocessing import Pool,Lock
import gc
import shared

from utilities import argpartition, FakePool,partition
from imageUtil import getImageData, writeImageData, makeMask, normalize
from scoreVocabulary import makeVocabulary

class DataManager(object):
    def __init__(self):
        self.objects = {}
    def setObject(self,name,o):
        self.objects[name] = o
    def delObject(self,name):
        del self.objects[name]
        gc.collect()
    def getObject(self,name):
        return self.objects[name]

dm = DataManager()

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

    #gc.collect()
    if pad:
        return (ifft2(fftimage))[:rOrig,:cOrig].real
    else:
        return (ifft2(fftimage)).real


def convolvePattern(patImg,pageImg):
    r = convolve(pageImg,patImg,True,True)
    N,M = pageImg.shape
    K,L = r.shape
    w = (K-N)/2
    h = (L-M)/2
    return r[w:w+N,h:h+M]

def convolveOnShared(patImg,name):
    global dm
    return convolvePattern(patImg,dm.getObject(name))

def getCoords(patImg,thr=.9,returnImage=False,returnCoords=True):
    r = convolveOnShared(patImg,'img')
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

def clusterCoords(coords):
    if coords.shape[0] == 0:
        return None
    if coords.shape[0] == 1:
        return coords
    l = cluster.hierarchy.linkage(coords)
    c = cluster.hierarchy.fcluster(l,2,criterion='distance')
    idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
    patternLocations = nu.empty((len(idict),2))
    for i,(k,v) in enumerate(idict.items()):
        patternLocations[i,:] = nu.mean(coords[tuple(v),:],0)
    return nu.array(patternLocations,nu.int)

class Bar(object):
    def __init__(self,vcenter,left,right,top,bottom):
        self.vcenter = vcenter
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.annotations = []
    def __str__(self):
        return 'Bar:\n\tleft: {0}\n\tright: {1}\n\ttop: {2}\n\tbottom: {3}'.format(self.left,
                                                                                   self.right,
                                                                                   self.top,
                                                                                   self.bottom)
    def getSubImage(self,img):
        return img[self.top:self.bottom,self.left:self.right]

    def addAnnotation(self,label,coordinates):
        self.annotations.append((label,coordinates))

    def getShape(self):
        return (self.bottom-self.top,self.right-self.left)

    def drawOnImage(self,img):
        img[self.top,self.left:self.right] = 1
        img[self.bottom,self.left:self.right] = 1
        img[self.top:self.bottom,self.left] = 1
        img[self.top:self.bottom,self.right] = 1
        return img

def getBarBBs(barItem):
    global dm
    barResults = []
    dm.setObject('pool',Pool())
    for i,barPatternFile in enumerate(barItem.getFiles()):
        barImg = barItem.getImage(0)
        barResults.append(dm.getObject('pool').apply_async(getCoords,(barImg,
                                                                      barItem.getThreshold(i),
                                                                      True,False)))
        gc.collect()

    dm.getObject('pool').close()
    dm.getObject('pool').join()
    dm.delObject('pool')
    gc.collect()
    barImage = nu.zeros(dm.getObject('img').shape)
    for r in barResults:
        barImage += r.get()
    
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
    # amount by which the 1/2 height of the first and last system
    # is scaled:
    ff = 1.5
    sbounds = (system_vcoords[1:]+system_vcoords[:-1])/2
    sbounds = nu.insert(sbounds,0,int((1+ff)*system_vcoords[0]-ff*sbounds[0]))
    sbounds = nu.append(sbounds,int((1+ff)*system_vcoords[-1]-ff*sbounds[-1]))

    bars = []
    for i,system_vcenter in enumerate(system_vcoords):
        b_hcs = bar_hcoords[i]
        if len(b_hcs) < 1:
            continue
        systemwidth = b_hcs[-1]-b_hcs[0]
        barwidths = nu.diff(b_hcs)
        minWidth = .02*systemwidth
        # filter out unlikily short bars
        baridx = nu.arange(len(barwidths))[barwidths >= minWidth]
        for j in baridx:
            top,bottom = int(nu.round(sbounds[i])),int(nu.round(sbounds[i+1]))
            left,right = int(nu.round(b_hcs[j])),int(nu.round(b_hcs[j+1]))
            bars.append(Bar(system_vcenter,left,right,top,bottom))
    return bars

def findPatternInBar(patImage,barIdx):
    o = nu.zeros((5,5))
    o[1:4,1:4] = .25
    o[2,2] = .5
    o[0,:] = -.5
    o[4,:] = -.5
    o[:,0] = -.5
    o[:,4] = -.5
    x0 = convolvePattern(patImage,dm.getObject('bars')[barIdx].getSubImage(dm.getObject('img')))
    x1 = convolvePattern(o,normalize(x0)-.5)
    return x1

def findPattern(vocItem):
    global dm
    results = {}
    for i in range(len(dm.getObject('bars'))):
        for j in range(len(vocItem.getFiles())):
            results[(vocItem.label,j,i)] = dm.getObject('pool').apply_async(findPatternInBar,(vocItem.getImage(j),i))
    return results
    
def processPage(vocabulary,outname):
    global dm
    barItem = vocabulary.getBar()
    if barItem is None:
        return False
    dm.setObject('bars',getBarBBs(barItem))

    vocItems = [vocabulary.getItem(label) for label in vocabulary.getLabels()]
    dm.setObject('pool',Pool())
    patResults = {}
    for vocItem in vocItems:
        # accumulate worker result objects
        patResults = dict(patResults.items()+findPattern(vocItem).items())

    dm.getObject('pool').close()
    dm.getObject('pool').join()
    dm.delObject('pool')
    keysPerPattern = partition(lambda x: x[:2], patResults.keys())
    for dum,keys in keysPerPattern.items():
        print(dum)
        acc = []
        accvec = nu.array([])
        keys.sort(key=lambda x: x[2])
        for k in keys:
            acc.append(patResults[k].get())
            accvec = nu.append(accvec,acc[-1])
        totalPixelSize = len(accvec)
        #accvec = 1-normalize(accvec)
        nu.savetxt('/tmp/v_{0}_{1}.txt'.format(dum[0],dum[1]),accvec)
        bins,centers = nu.histogram(accvec,50)
        print(bins[-20:])
        print('bin ratio',bins[-1]/float(bins[-2]))
        #diffbins = nu.diff(bins)
        # big peak
        #bigpeakidx = nu.argmin(diffbins)
        #mx = nu.max(accvec)
        #print('max/bigpeak',mx/centers[bigpeakidx])
        #print('max/median',mx/nu.median(accvec))
        #print('criterion',nu.log(nu.median(accvec)))
        # discard everything up til big peak
        #diffbins[:bigpeakidx] = 0
        # discard everything dropping off
        #diffbins[bigpeakidx:][diffbins[bigpeakidx:] < 0] = 0
        #candidates = nu.nonzero(diffbins)[0]
        
        if False: #len(candidates) > 0:
            thr = centers[candidates[-1]]
        else:
            thr = nu.inf
        del accvec
        print(thr)
        for i,img in enumerate(acc):
            #idx = img < thr
            #img[idx] = .2*img[idx]
            #img[idx] = 0
            nu.savetxt('/tmp/bp_{0}_{1}_{2}.txt'.format(dum[0],dum[1],i),img[::-1,:])
            #coords = clusterCoords(nu.column_stack(nu.nonzero(img)))
            #if coords != None:
            #    dm.getObject('bars')[i].addAnnotation(dum[0],coords)
    #for k,v in patResults.items():
    #    # wait for result objects to return
    #    nu.savetxt('/tmp/bp_{0}_{1}_{2}.txt'.format(k[0],k[1],k[2]),v.get()[::-1,:])
    newImg = nu.zeros(dm.getObject('img').shape,nu.float)
    for i,bar in enumerate(dm.getObject('bars')):
        print('bar {0}: {1}'.format(i,bar.annotations))
        bar.drawOnImage(newImg)
    normImg = normalize(dm.getObject('img'))
    im_r = nu.minimum(normImg,1-.5*newImg)
    im_r = nu.array((1-im_r)*255,nu.uint8)
    im_g = nu.maximum(normImg,newImg)
    im_g = nu.array((1-im_g)*255,nu.uint8)
    im_b = im_g
    writeImageData(outname,newImg.shape,im_r,im_g,im_b)
        
    sys.exit()

    for i,pr in enumerate(patResults):
        print(vocItems[i].label)
        for baridx,p in enumerate(pr):
            nu.savetxt('/tmp/bp_{0}_{1}.txt'.format(vocItems[i].label,baridx),p.get()[::-1,:])
        #pats.append([p.get() for p in pr])

    #pimg,pcoord = getCoords(bar.getSubImage(img),patternFile,thr,True,True,False,False)
    #patresults.append(shared.pool.apply_async(getCoords,(bar,patternFile,thr,True,True)))
    #print(clusterCoords(pcoord))
    #nu.savetxt('/tmp/o.txt',pimg)
    #sys.exit()


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
    #vocabularyDir = './vocabularies/dme-4096'
    vocabularyDir = './vocabularies/dme-2048'
    vocabulary = makeVocabulary(vocabularyDir)
    imgfile = sys.argv[1]
    print('main',os.getpid())
    dm.setObject('img',getImageData(imgfile) -.5)
    outname= os.path.join('/tmp/',os.path.basename(imgfile))
    processPage(vocabulary,outname)
