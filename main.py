#!/usr/bin/env python

import sys,os
from PIL import Image
import numpy as nu
from scipy import signal,cluster
from numpy.fft import fft2, ifft2
from multiprocessing import Pool

from utilities import argpartition, FakePool
from smooth import makeMask,normalize

def getImageData(filename):
    imageFh = Image.open(filename)
    #data = nu.array(list(imageFh.getdata()))
    s = list(imageFh.size)
    s.reverse()
    data = nu.array(imageFh.getdata()).reshape(tuple(s))
    img_min = nu.min(data)
    img_max = nu.max(data)
    return 1-(nu.array(data,nu.float)-img_min)/(img_max-img_min)

def writeImageDataNew(filename,data,color=(1,1,1),alphaChannel=None):
    size = tuple(reversed(data.shape))
    #img = Image.new('RGBA',size)
    dmin = nu.min(data)
    dmax = nu.max(data)
    if dmin >= 0 and dmax <= 1: 
        ndata = data
    else:
        ndata = (data-dmin)/(dmax-dmin)

    if alphaChannel is None:
        alphaChannel = nu.ones(data.shape,nu.uint8)*255
    assert alphaChannel.size == data.size
    im_r = Image.fromarray(nu.array(255*float(color[0])*ndata,nu.uint8)) # monochromatic image
    im_g = Image.fromarray(nu.array(255*float(color[1])*ndata,nu.uint8)) # monochromatic image
    im_b = Image.fromarray(nu.array(255*float(color[2])*ndata,nu.uint8)) # monochromatic image
    ach = Image.fromarray(alphaChannel) # monochromatic image
    imgrgba = Image.merge('RGBA', (im_r,im_g,im_b,ach)) # color image
    imgrgba.save(filename)

def writeImageDataNewNew(filename,size,im_r=None,im_g=None,im_b=None,alphaChannel=None):
    #size = tuple(reversed(size))
    #img = Image.new('RGBA',size)
    print(nu.min(im_r),nu.max(im_r))
    if im_r is None:
        im_r = nu.zeros(size,nu.uint8)*255
    else:
        im_r = nu.array(im_r,nu.uint8)
    if im_g is None:
        im_g = nu.zeros(size,nu.uint8)*255
    else:
        im_g = nu.array(im_g,nu.uint8)
    if im_b is None:
        im_b = nu.zeros(size,nu.uint8)*255
    else:
        im_b = nu.array(im_b,nu.uint8)
    if alphaChannel is None:
        alphaChannel = nu.ones(size,nu.uint8)*255
    else:
        alphaChannel = nu.array(alphaChannel,nu.uint8)

    imgrgba = Image.merge('RGBA', (Image.fromarray(im_r),
                                   Image.fromarray(im_g),
                                   Image.fromarray(im_b),
                                   Image.fromarray(alphaChannel))) # color image
    imgrgba.save(filename)

def writeImageData(filename,data,size=None):
    """Write the value of numpy array data as an image to filename. If
    size is not specified, the shape of the array is taken to
    determine the image size. The file encoding is guessed from the
    filename (extension). The image is greyscale."""
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


def preprocessPattern(bar,fn):
    N,M = bar.shape
    n = nu.arange(N)
    n = n*n[::-1]
    m = nu.arange(M)
    m = m*m[::-1]
    o,p = nu.meshgrid(n,m)

    mask = makeMask(bar)

    #mask = nu.abs(mask)
    nu.savetxt('/tmp/n-{0}.txt'.format(fn),mask)
    nu.savetxt('/tmp/np-{0}.txt'.format(fn),mask*bar)
    
    nu.savetxt('/tmp/p-{0}.txt'.format(fn),bar)
    #nu.savetxt('/tmp/m-{0}.txt'.format(fn),o.T*p.T)
    #return bar*o.T*p.T
    return bar*mask

def preprocessPatternNew(bar):
    N,M = bar.shape
    mask = nu.zeros(bar)
    mask[1:,:] += nu.diff(bar,0)
    mask[:,1:] += nu.diff(bar,1)
    nu.savetxt('/tmp/m.txt',mask)
    #return bar*o.T*p.T
    return bar

def getCoords(img,patfile,thr=.9,returnImage=False,returnCoords=True):
    pat = getImageData(patfile)
    pat = preprocessPattern(pat -.5,os.path.basename(patfile))
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
    #writeImageData('/tmp/bars.png',img)
    return bar_hcoords
    #w = nu.int(2*(w/2)+1)
    #hw = int(nu.floor(w/2))

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

class VocabularyItem(object):
    def __init__(self,l,dirname):
        parts = l.strip().split()
        self.label = parts[0]
        self.thresholds = [float(x) for x in parts[1::2]]
        self.files = [os.path.join(dirname,x) for x in parts[2::2]]
        
    def getThresholds(self):
        return self.thresholds
    def getFiles(self):
        return self.files
    def getImage(self,i=0):
        pat = getImageData(self.files[i])
        nu.savetxt('/tmp/{0}.txt'.format(self.label),normalize(pat))
        return normalize(pat)
    def __str__(self):
        return '{0} {1}'.format(self.label,self.files)

class Vocabulary(object):
    def __init__(self):
        self.vocItems = {}
    def addItem(self,vi):
        self.vocItems[vi.label] = vi
    def getItem(self,label):
        return self.vocItems.get(label,None)

class ScoreVocabulary(Vocabulary):
    def __init__(self):
        super(ScoreVocabulary,self).__init__()
        self.bar = None
    def addItem(self,vi):
        if vi.label == 'bar':
            self.bar = vi
        else:
            self.vocItems[vi.label] = vi
    def getItem(self,label):
        return self.vocItems.get(label,None)
    def getLabels(self):
        return self.vocItems.keys()
    def getBar(self):
        return self.bar

def makeVocabulary(vocabularyDir):
    vocfile = os.path.join(vocabularyDir,'vocabulary.txt')
    vocabulary = ScoreVocabulary()
    with open(vocfile,'r') as f:
        for l in f.readlines():
            if l[0] is not '#':
                vocabulary.addItem(VocabularyItem(l,vocabularyDir))
    return vocabulary


def processPage(img,vocabulary,pool,outname):
    barItem = vocabulary.getBar()
    if barItem is None:
        return False

    barresults = []
    for barPatternFile in barItem.getFiles():
        barresults.append(pool.apply_async(getCoords,(img,barPatternFile,.8,True,False)))

    vocresults = []
    voclabels = vocabulary.getLabels()
    for label in voclabels:
        vi = vocabulary.getItem(label)
        patternFile = vi.getFiles()[0]
        thr = vi.getThresholds()[0]
        vocresults.append(pool.apply_async(getCoords,(img,patternFile,thr,False,True)))

    pool.close()
    pool.join()
    barImage = nu.zeros(img.shape)
    for br in barresults:
        print('waiting for bar result...')
        barImage += br.get()
    print('finding systems...')
    barImage = barImage/len(barresults)
    #rbar,cbar = barresult.get()
    system_vcoords = findSystems(barImage)
    assert len(system_vcoords) > 0
    system_vcoords.sort()
    print('finding bars...')
    bar_hcoords = findBars(system_vcoords,barImage)
    del barImage
    assert len(system_vcoords) > 1

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
    sbounds = nu.insert(sbounds,0,2*system_vcoords[0]-sbounds[0])
    sbounds = nu.append(sbounds,2*system_vcoords[-1]-sbounds[-1])

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
    im_b = im_g #nu.array((1-normalize(pageImg))*255,nu.uint8)
    print(nu.min(im_g),nu.max(im_g))
    #writeImageDataNewNew('/tmp/pat.png',pageImg.shape,im_r,im_g,im_b)
    writeImageDataNewNew(outname,img.shape,im_r,im_g,im_b)
                         
                         

    #writeImageData('/tmp/pat.png',nu.array(normalize(1-normalize(pageImg)+globalpat)*200,nu.uint8))


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
