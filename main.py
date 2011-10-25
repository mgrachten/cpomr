#!/usr/bin/env python

import sys,os,resource
import numpy as nu
from scipy import signal,cluster,stats,mgrid
from numpy.fft import fft2, ifft2
from multiprocessing import Pool,Lock
import multiprocessing.sharedctypes as mps


from utilities import argpartition, FakePool,partition
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from scoreVocabulary import makeVocabulary

def gauss_kern(size, sizey=None,w=.3):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    #g = nu.exp(-(x**2/float(size)+y**2/float(sizey)))
    g = nu.exp(-(x**2/(w*float(size**2))+y**2/(w*float(sizey**2))))
    return g
    #return g / g.sum()

class DataManager(object):
    def __init__(self):
        self.objects = {}
        self.shapes = {}
        self.dtypes = {}
        self.typemap = {nu.int8:mps.ctypes.c_int8}
    def setObject(self,name,o):
        self.objects[name] = o
    def delObject(self,name):
        del self.objects[name]
    def getObject(self,name):
        return self.objects[name]
    def setArray(self,name,img):
        self.shapes[name] = img.shape
        self.dtypes[name] = img.dtype
        self.objects[name] = mps.RawArray(self.typemap[img.dtype.type],img.reshape(-1))
        return self.objects[name]
    def getArray(self,name):
        return nu.ndarray(self.shapes[name],self.dtypes[name],self.objects[name])

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
    #f1 = fft2(image1, s=(r,c))
    #fftimage = f1*fft2(image2[::-1,::-1],s=(r,c))
    fftimage = fft2(image1, s=(r,c))*fft2(image2[::-1,::-1],s=(r,c))
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
    if name == 'img':
        return convolvePattern(patImg,dm.getArray(name))
    else:
        return convolvePattern(patImg,dm.getObject(name))

def getCoords(patImg,thr=.9,returnImage=False,returnCoords=True):
    r = convolveOnShared(patImg,'img')
    rmin = nu.min(r)
    rang = nu.max(r)-rmin
    nthr = thr*rang+rmin
    r = r >= nthr
    if returnCoords and returnImage:
        return r, nu.column_stack(nu.nonzero(r))
    elif returnCoords:
        return nu.column_stack(nu.nonzero(r))
    elif returnImage:
        return r
    else:
        return None

def getCoordsNonBool(patImg,thr=.9,returnImage=False,returnCoords=True):
    r = convolveOnShared(patImg,'img')
    rmin = nu.min(r)
    rang = nu.max(r)-rmin
    if True:
        return r
    r = (r-rmin)/rang
    #nthr = thr*rang+rmin
    idx = r < thr
    r[idx] = 0
    if returnCoords and returnImage:
        return r, nu.column_stack(nu.nonzero(r))
    elif returnCoords:
        return nu.column_stack(nu.nonzero(r))
    elif returnImage:
        return r
    else:
        return None

def findPatternInBar(patImage,barIdx):
    return convolvePattern(patImage,dm.getObject('bars')[barIdx].getSubImage(dm.getArray('img')))

def findPattern(vocItem):
    global dm
    results = {}
    for i in range(len(dm.getObject('bars'))):
        for j in range(len(vocItem.getFiles())):
            results[(vocItem.label,j,i)] = dm.getObject('pool').apply_async(findPatternInBar,(vocItem.getImage(j),i))
    emptyBar = nu.zeros(dm.getObject('bars')[0].getShape(),nu.int8)-128
    return results

def convolvePatternFunc((img,pat,fun)):
    return fun(convolvePattern(img,pat))

def calibrateVocItem(vocItem):
    global dm
    # jitter std as a proportion of img range
    emptyBar = nu.zeros(dm.getObject('bars')[0].getShape(),nu.int8)-128
    s = 30
    results = []
    for j in range(len(vocItem.getFiles())):
        jitterStd = vocItem.getThreshold(j)
        selfimg = jitterImageEdges(vocItem.getImage(j),jitterStd)
        v = []
        for k in range(s):
            v.append(dm.getObject('pool').map_async(convolvePatternFunc,[(vocItem.getImage(j),emptyBar,nu.max),
                                                                   (vocItem.getImage(j),selfimg,nu.max)]))
        results.append(v)
    for j in range(len(vocItem.getFiles())):
        v = nu.median(nu.array([x.get() for x in results[j]]),0)
        vocItem.setCalibration(j,v[0],v[1])

def calibrateBar(barItem):
    global dm
    # jitter std as a proportion of img range
    #emptyBar = 
    rimg = nu.array(nu.random.random(dm.getArray('img').shape),nu.int8)*255-128
    #rimg = nu.zeros(dm.getArray('img').shape,nu.int8)-128
    s = 30
    results = []
    for j in range(len(barItem.getFiles())):
        jitterStd = barItem.getThreshold(j)
        selfimg = jitterImageEdges(barItem.getImage(j),jitterStd)
        v = []
        for k in range(s):
            v.append(dm.getObject('pool').map_async(convolvePatternFunc,[(barItem.getImage(j),rimg,nu.max),
                                                                   (barItem.getImage(j),selfimg,nu.max)]))
        results.append(v)
    for j in range(len(barItem.getFiles())):
        v = nu.median(nu.array([x.get() for x in results[j]]),0)
        print(v)
        barItem.setCalibration(j,v[0],v[1])

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

def clusterInBar(coords,x1,x2,y1,y2,distance=2):
    margin = 5
    inbar = coords[nu.logical_and(nu.logical_and(coords[:,0]>x1+margin,coords[:,0]<x2-margin),
                          nu.logical_and(coords[:,1]>y1+margin,coords[:,1]<y2-margin)),:]
    
    if inbar.shape[0] == 0:
        return None
    if inbar.shape[0] == 1:
        return inbar
    l = cluster.hierarchy.linkage(inbar)
    c = cluster.hierarchy.fcluster(l,distance,criterion='distance')
    idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
    patternLocations = nu.empty((len(idict),2))
    for i,(k,v) in enumerate(idict.items()):
        patternLocations[i,:] = nu.mean(inbar[tuple(v),:],0)
    return nu.array(patternLocations,nu.int)

def clusterCoords(coords,distance=3):
    if coords.shape[0] == 0:
        return None
    if coords.shape[0] == 1:
        return coords
    l = cluster.hierarchy.linkage(coords)
    c = cluster.hierarchy.fcluster(l,distance,criterion='distance')
    idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
    patternLocations = nu.empty((len(idict),2))
    for i,(k,v) in enumerate(idict.items()):
        patternLocations[i,:] = nu.mean(coords[tuple(v),:],0)
    return nu.array(patternLocations,nu.int)

def clusterCoordsNonBool(m):
    coords = nu.column_stack(nu.nonzero(m))
    if coords.shape[0] == 0:
        return None
    if coords.shape[0] == 1:
        return coords
    l = cluster.hierarchy.linkage(coords)
    c = cluster.hierarchy.fcluster(l,2,criterion='distance')
    print(coords)
    print(c)
    idict = argpartition(lambda x: x[0], nu.column_stack((c,nu.arange(len(c)))))
    patternLocations = nu.empty((len(idict),2))
    for i,(k,v) in enumerate(idict.items()):
        clmax = nu.argmax([m[coord[0],coord[1]] for coord in coords[tuple(v),:]])
        patternLocations[i,:] = coords[tuple(v),:][clmax]
    return nu.array(patternLocations,nu.int)

class Bar(object):
    def __init__(self,vcenter,left,right,top,bottom):
        self.vcenter = vcenter
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.annotations = {}

    def addAnnotation(self,label,coordinates):
        self.annotations[label] = self.annotations.get(label,[])+[coordinates]

    def __str__(self):
        return 'Bar:\n\tleft: {0}\n\tright: {1}\n\ttop: {2}\n\tbottom: {3}'.format(self.left,
                                                                                   self.right,
                                                                                   self.top,
                                                                                   self.bottom)
    def getSubImage(self,img):
        return img[self.top:self.bottom,self.left:self.right]

    def getBB(self):
        return nu.array((self.top,self.left,self.bottom,self.right),nu.int)

    def getShape(self):
        return (self.bottom-self.top,self.right-self.left)

    def getLeft(self):
        return self.left
    def getTop(self):
        return self.top
    def drawOnImage(self,img):
        img[self.top,self.left:self.right] = 1
        img[self.bottom,self.left:self.right] = 1
        img[self.top:self.bottom,self.left] = 1
        img[self.top:self.bottom,self.right] = 1
        return img

def formatBarNotes(i,b):
    return nu.column_stack((nu.zeros(len(b.annotations.get('n',[])),nu.int)+i,
                            [(x[0],x[1]) for x in b.annotations.get('n',[])]))
    #return nu.column_stack((nu.zeros(len(b.annotations.get('n',[])),nu.int)+i,
    #                        [(x[0]-b.vcenter,x[1]) for x in b.annotations.get('n',[])]))

def getBarBBs(barItem):
    global dm
    barResults = []
    print('preparing bar detection ...'),
    dm.setObject('pool',Pool())
    for i,barPatternFile in enumerate(barItem.getFiles()):
        barImg = barItem.getImage(0)
        barResults.append(dm.getObject('pool').apply_async(getCoords,(barImg,
                                                                      barItem.getThreshold(i),
                                                                      True,False)))

    #calibrateBar(barItem)
    dm.getObject('pool').close()
    dm.getObject('pool').join()
    dm.delObject('pool')
    barResults[0] = barResults[0].get()
    for r in barResults[1:]:
        barResults[0] = nu.logical_or(barResults[0],r.get())
    print('Done')
    print('finding systems...'),
    barResults = barResults[0]
    # TMP
    train = False
    if train:
        # for train mode,  use getCoordsNonBool, and calibratebar
        nu.savetxt('/tmp/o.txt',barItem.normalize(0,barResults,True))
        sys.exit()
        #barResults = normalize(barResults)
        normImg = nu.array(255-normalize(dm.getArray('img'))*255,nu.uint8)

        barimg = nu.array(255-barResults*255,nu.uint8)
        alpha = .1
        im = alpha*normImg + (1-alpha)*barimg #nu.minimum(normImg,barimg)
        im_g = im #barimg
        writeImageData('/tmp/bars/br.png'.format(i),barResults.shape,im,im_g,im)
        
        #barcoords = clusterCoords(nu.column_stack(nu.nonzero(barResults)))
        barcoords = clusterCoordsNonBool(barResults)
        #barcoords = nu.column_stack(nu.nonzero(barResults))
        bs = nu.array(barItem.getImage(0).shape)
        bars = []
        print(len(barcoords))

        for i,cc in enumerate(barcoords):
            tl = nu.array(nu.round(cc-(bs-1)/2.0),nu.int)
            br = nu.array(tl+bs,nu.int)
            imr = 255-normalize(dm.getArray('img')[tl[0]:br[0],tl[1]:br[1]])*255
            bars.append(imr)
            im = nu.array(imr,nu.uint8)
            writeImageData('/tmp/bars/b{0:04d}.png'.format(i),bs,im,im,im)
        bars = nu.array(bars)
        bar = nu.median(bars,axis=0)
        imr = normalize(bar)*255
        im = nu.array(imr,nu.uint8)
        #alpha = nu.std(bars,axis=0)
        alpha = nu.mean(nu.abs(bars-bar),axis=0)
        oldalpha = normalize(barItem.masks[0])
        alpha = nu.array(255*(.3*oldalpha+.7*(1-normalize(alpha))),nu.uint8)
        #alpha = nu.array(barItem.masks[0]/2.0+alpha/2.0,nu.uint8)
        im = nu.array(barItem.pureimages[0]/2.0+im/2.0,nu.uint8)
        writeImageData('/tmp/bars/bmean.png'.format(i),bs,im,im,im,alpha)
    
        sys.exit()
    # END TMP
    system_vcoords = findSystems(barResults)
    assert len(system_vcoords) > 0
    system_vcoords.sort()
    print('Done')
    print('finding bars...'),
    bar_hcoords = findBars(system_vcoords,barResults)
    del barResults

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
    ff = 1.1
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
            hextra = 0
            vextra = 5
            top,bottom = int(nu.round(sbounds[i]-vextra)),int(nu.round(sbounds[i+1]+vextra))
            left,right = int(nu.round(b_hcs[j]-hextra)),int(nu.round(b_hcs[j+1]+hextra))
            bars.append(Bar(system_vcenter,left,right,top,bottom))

    print('Done')
    return bars

def processPage(vocabulary,outname):
    global dm
    barItem = vocabulary.getBar()
    if barItem is None:
        return False
    dm.setObject('bars',getBarBBs(barItem))

    vocItems = [vocabulary.getItem(label) for label in vocabulary.getLabels()]

    print('preparing pattern detection ...'),
    dm.setObject('pool',Pool())
    patResults = {}

    for vocItem in vocItems:
        calibrateVocItem(vocItem)
        # accumulate worker result objects
        patResults = dict(patResults.items()+findPattern(vocItem).items())

    dm.getObject('pool').close()
    dm.getObject('pool').join()
    dm.delObject('pool')
    print('Done')

    keysPerPattern = partition(lambda x: x[:2], patResults.keys())
    coords = {}
    for dum,keys in keysPerPattern.items():
        label,labelim = dum
        print('processing pattern "{0}", image {1}...'.format(label,labelim)),
        coords[dum] = []
        keys.sort(key=lambda x: x[2])

        vocItem = vocabulary.getItem(label)
        for k in keys:
            bar = dm.getObject('bars')[k[2]]
            m = patResults[k].get()
            barleft = bar.getLeft()
            bartop = bar.getTop()
            thr = 1
            mb = vocItem.normalize(labelim,m,True,thr)
            cc = clusterCoords(nu.column_stack(nu.nonzero(mb)))
            if cc != None:
                cc += (bartop,barleft)
                coords[dum] = coords[dum]+[tuple(x) for x in cc]
                for x in cc:
                    bar.addAnnotation(label,tuple(x))
        print('Done')

    bars = dm.getObject('bars')
    bars = [bar for bar in bars if len(bar.annotations) > 0]
    #d = nu.empty((0,3),nu.int)
    d = []
    barbb = []
    for i,b in enumerate(bars):
        print('bar',i)
        print(b.annotations.get('n',[]))
        if b.annotations.has_key('n'):
            fmn = formatBarNotes(i,b)
            #d = nu.vstack((d,fmn))
            d.append(fmn)
            barbb.append(b.getBB())
    d = nu.vstack(tuple(d))
    barbb = nu.vstack(tuple(barbb))
    nu.savetxt(os.path.splitext(outname)[0]+'.txt',d[:,(0,2,1)],fmt='%d')
    nu.savetxt(os.path.splitext(outname)[0]+'barbb.txt',barbb,fmt='%d')
    dm.setObject('bars',bars)

    print('drawing results to image...'),
    # draw results to image
    newImg = nu.zeros(dm.getArray('img').shape,nu.float)
    for i,bar in enumerate(dm.getObject('bars')):
        bar.drawOnImage(newImg)
    normImg = normalize(dm.getArray('img'))
    im_r = nu.minimum(normImg,1-.5*newImg)
    im_r = nu.array((1-im_r)*255,nu.uint8)
    im_g = nu.maximum(normImg,newImg)
    im_g = nu.array((1-im_g)*255,nu.uint8)
    im_b = im_g.copy()
    alpha = .5
    for i,(k,v) in enumerate(coords.items()):
        for c in v:
            vi = vocabulary.getItem(k[0])
            sh = nu.array(vi.getImage(k[1]).shape)
            tl = nu.array(c-sh/2.0,nu.int)
            br = nu.array(c+sh/2.0,nu.int)
            im_r[tl[0]:br[0],tl[1]:br[1]] *= (1-alpha)
            im_g[tl[0]:br[0],tl[1]:br[1]] *= (1-alpha)
            im_b[tl[0]:br[0],tl[1]:br[1]] *= (1-alpha)
            im_r[tl[0]:br[0],tl[1]:br[1]] += alpha*vi.getColor()[0]
            im_g[tl[0]:br[0],tl[1]:br[1]] += alpha*vi.getColor()[1]
            im_b[tl[0]:br[0],tl[1]:br[1]] += alpha*vi.getColor()[2]
        
    writeImageData(outname,newImg.shape,im_r,im_g,im_b)
    print('Done')

if __name__ == '__main__':
    #vocabularyDir = './vocabularies/dme-4096'
    vocabularyDir = './vocabularies/dme-2048/vocabulary-debussy-childrenscorner.txt'
    print('loading vocabulary...'),
    vocabulary = makeVocabulary(vocabularyDir)
    print('Done')
    imgfile = sys.argv[1]
    print('reading score page {0}...'.format(imgfile)),
    #dm.setArray('img',(255-nu.array(getImageData(imgfile),nu.int8)-128))
    img = nu.array(127-getPattern(imgfile,useMask=False,alphaAsMaskIfAvailable=False),nu.int8)
    dm.setArray('img',img)
    print('Done')
    outname= os.path.join('/tmp/',os.path.basename(imgfile))
    processPage(vocabulary,outname)

