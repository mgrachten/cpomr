#!/usr/bin/env python


#    Copyright 2012, Maarten Grachten.
#
#    This file is part of CPOMR.
#
#    CPOMR is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    CPOMR is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with CPOMR.  If not, see <http://www.gnu.org/licenses/>.
#

import sys,os,logging
from PIL import Image
import numpy as nu
from scipy import signal

def selectColumns(vsums,bins):
    N = len(vsums)
    nzidx = nu.nonzero(vsums)[0]
    binSize = int(nu.floor(len(nzidx)/float(bins)))
    #print(len(nzidx),binSize,bins)
    idxm = nzidx[:bins*binSize].reshape((bins,binSize))
    for i in range(bins):
        idxm[i,:] = idxm[i,nu.argsort(vsums[idxm[i,:]])]

    columns = idxm.T.ravel()
    colBins = nu.array(range(bins)*binSize)

    columns = nu.append(columns,nzidx[bins*binSize:])
    # incorrect, but mostly irrelevant:
    colBins = nu.append(colBins,nu.zeros(len(nzidx)-bins*binSize))
    assert len(columns) == len(colBins)
    return columns,colBins

class Rotator(object):
    def __init__(self,theta,og,ol):
        self.og = og
        self.ol = ol
        self.theta = theta

    def rotate(self,x,y=None):
        if y == None:
            return nu.column_stack(self._rotate(x[:,0],x[:,1]))
        else:
            return self._rotate(x,y)

    def derotate(self,x,y=None):
        if y == None:
            return nu.column_stack(self._derotate(x[:,0],x[:,1]))
        else:
            return self._derotate(x,y)

    def _rotate(self,xx,yy):
        xxr = nu.cos(self.theta*nu.pi)*(xx-self.og[0])-nu.sin(self.theta*nu.pi)*(yy-self.og[1])
        yyr = nu.sin(self.theta*nu.pi)*(xx-self.og[0])+nu.cos(self.theta*nu.pi)*(yy-self.og[1])
        return xxr+self.ol[0],yyr+self.ol[1]

    def _derotate(self,xxr,yyr):
        xx = nu.cos(-self.theta*nu.pi)*(xxr-self.ol[0])-nu.sin(-self.theta*nu.pi)*(yyr-self.ol[1])
        yy = nu.sin(-self.theta*nu.pi)*(xxr-self.ol[0])+nu.cos(-self.theta*nu.pi)*(yyr-self.ol[1])
        return xx+self.og[0],yy+self.og[1]


def findBeginEnd(N,maxes,mins,margin=0):
    over = nu.where(maxes > N-margin)[0]
    under = nu.where(mins < margin)[0]
    return (nu.max(under) if len(under) > 0 else 0,
            nu.min(over) if len(over) > 0 else len(maxes))

def trimCoordinates(shape,xx,yy,margin=0):
    """
    If necessary, trim the coordinates of xx and yy (jointly),
    such that all values in xx and yy are valid inside shape
    """
    xb,xe = findBeginEnd(shape[0],nu.max(xx,1),nu.min(xx,1),margin)
    yb,ye = findBeginEnd(shape[1],nu.max(yy,0),nu.min(yy,0),margin)
    return xx[xb:xe,yb:ye],yy[xb:xe,yb:ye]

def getAntiAliasedImg(img,xx,yy,trim=True):
    #jitter = .001
    #xx += nu.random.normal(0,jitter,xx.shape)
    #yy += nu.random.normal(0,jitter,yy.shape)
    if trim:
        xx,yy = trimCoordinates(img.shape,xx,yy,margin=1)

    xf = nu.floor(xx).astype(nu.int)
    xc = nu.ceil(xx).astype(nu.int)
    yf = nu.floor(yy).astype(nu.int)
    yc = nu.ceil(yy).astype(nu.int)

    wxc = xx%1
    wxf = 1-wxc
    wyc = yy%1
    wyf = 1-wyc

    return (((wxf+wyf)*img[xf,yf] +
	     (wxf+wyc)*img[xf,yc] +
	     (wxc+wyf)*img[xc,yf] +
	     (wxc+wyc)*img[xc,yc])/4.0).astype(nu.uint8)

def smooth(x,k):
    return nu.convolve(x,signal.hanning(k),'same')

def findPeaks(x):
    return nu.where(nu.diff(nu.sign(nu.diff(x)))<0)[0]+1

def findValleys(x):
    return nu.where(nu.diff(nu.sign(nu.diff(x)))>0)[0]+1

def findPeaksOld(v):
    """find the peaks in a smooth curve
    """
    x = nu.zeros(len(v))
    x[2:len(x)-2] = nu.diff(nu.diff(nu.sign(nu.diff(v)),2))
    x[x<2] = 0
    peaks = nu.nonzero(x)[0]
    return peaks

def findValleysOld(v):
    """find the valleys in a smooth curve
    """
    x = nu.zeros(len(v))
    sdx = nu.sign(nu.diff(v))
    nz = nu.nonzero(sdx)[0]
    valleybegins = nu.nonzero(nu.diff(sdx[nz]) > 0)[0]
    valleys = .5+(nz[valleybegins]+nz[valleybegins+1])/2.0
    return valleys.astype(nu.int)

def normalize(x):
    xmin = float(nu.min(x))
    xmax = float(nu.max(x))
    #assert xmin < xmax
    if xmin < xmax:
        return (nu.array(x,nu.float32)-xmin)/(xmax-xmin)
    else:
        return nu.ones(x.shape,nu.float32)

def makeDiffImage(img):
    dimg0 = nu.diff(img,axis=0)
    dimg0 = nu.vstack((dimg0[0,:].reshape(1,-1),(dimg0[1:,:]+dimg0[:-1,:])/2.0,dimg0[-1,:].reshape(1,-1)))
    dimg1 = nu.diff(img,axis=1)
    dimg1 = nu.hstack((dimg1[:,0].reshape(-1,1),(dimg1[:,1:]+dimg1[:,:-1])/2.0,dimg1[:,-1].reshape(-1,1)))
    return normalize(dimg0+dimg1)

def jitterImageEdges(img,var):
    irange = float(nu.max(img))-float(nu.min(img))
    dimg = makeDiffImage(img)
    jimg = img+dimg*nu.random.normal(0,var*irange,img.shape)
    if nu.issubdtype(img.dtype,nu.int):
        r = 2**(img.dtype.itemsize*8-1)
        nu.clip(jimg,a_min=-r,a_max=r-1,out=jimg)
    elif nu.issubdtype(img.dtype,nu.float):
        pass
    elif nu.issubdtype(img.dtype,nu.complex):
        pass
    else:
        ## assume uint of some size
        nu.clip(jimg,a_min=0,a_max=2**(img.dtype.itemsize*8)-1,out=jimg)
    return nu.array(jimg,img.dtype)

def makeMask(img):
    imin,imax = nu.min(img),nu.max(img)
    mask = nu.zeros(img.shape,nu.float)-imin
    K = 5
    N,M = img.shape
    for i in range(N):
        for j in range(M):
            o1 = max(0,i-K)
            o2 = max(0,j-K)
            subimage = img[o1:min(i+K,N),o2:min(j+K,M)]
            k = float(nu.argmax(subimage))
            mask[i,j] = subimage[k/subimage.shape[1],k%subimage.shape[1]]
    return (mask-imin)/(imax-imin)

def getImageData(filename):
    imageFh = Image.open(filename)
    #data = nu.array(list(imageFh.getdata()))
    s = list(imageFh.size)
    s.reverse()
    data = nu.array(imageFh.getdata(),nu.uint8).reshape(tuple(s))
    return data

def avgChannels(img):
    if img.shape[1] > 1:
        return nu.mean(img,axis=1)
    else:
        return img


def getPattern(filename,useMask=True,alphaAsMaskIfAvailable=True):
    imageFh = Image.open(filename)
    #data = nu.array(list(imageFh.getdata()))

    assert imageFh.mode.startswith('L') or imageFh.mode.startswith('RGB') or imageFh.mode.startswith("1")
    s = tuple(reversed(imageFh.size))

    fimg = nu.array(imageFh.getdata(),nu.uint8).reshape((s[0]*s[1],-1))

    #nch = img.shape[1]
    nch = len(imageFh.mode)
    hasAlpha = imageFh.mode[-1] == 'A'

    # get greyscale img info, averaging RGB channels if necessary
    if hasAlpha:
        img = avgChannels(fimg[:,:-1])
    else:
        img = avgChannels(fimg)
    #nu.savetxt('/tmp/i.txt',img.reshape(s))
    if useMask and not (alphaAsMaskIfAvailable and hasAlpha):
        # center around zero and 
        mask = makeMask(255-img.reshape(s)).reshape(img.shape)
    elif hasAlpha and alphaAsMaskIfAvailable:
        # CHECK IF TRANSPARENCY IS 0 OR 255!!!
        mask = nu.array(fimg[:,-1],nu.float)/255
        trivialAlpha = nu.max(mask)<=nu.min(mask)
        log = logging.getLogger(__name__)
        if trivialAlpha:
            log.info('Pattern {0} has a trivial\nalpha-channel. Using standard masking procedure instead.\n'.format(filename))
            mask = makeMask(255-img.reshape(s)).reshape(img.shape)

    if useMask:
        return nu.array((127-img)*mask,nu.int8).reshape(s)
    else:
        return nu.array(img,nu.uint8).reshape(s)

def getImageAndMask(filename,useMask=True,alphaAsMaskIfAvailable=True):
    imageFh = Image.open(filename)
    #data = nu.array(list(imageFh.getdata()))
    assert imageFh.mode.startswith('L') or imageFh.mode.startswith('RGB')
    s = tuple(reversed(imageFh.size))

    fimg = nu.array(imageFh.getdata(),nu.uint8).reshape((s[0]*s[1],-1))
    #nch = img.shape[1]
    nch = len(imageFh.mode)
    hasAlpha = imageFh.mode[-1] == 'A'

    # get greyscale img info, averaging RGB channels if necessary
    if hasAlpha:
        img = avgChannels(fimg[:,:-1])
    else:
        img = avgChannels(fimg)
    nu.savetxt('/tmp/i.txt',img.reshape(s))
    if useMask and not (alphaAsMaskIfAvailable and hasAlpha):
        # center around zero and 
        mask = makeMask(255-img.reshape(s)).reshape(img.shape)
    elif hasAlpha and alphaAsMaskIfAvailable:
        # CHECK IF TRANSPARENCY IS 0 OR 255!!!
        mask = nu.array(fimg[:,-1],nu.float)/255
        trivialAlpha = nu.max(mask)<=nu.min(mask)
        log = logging.getLogger(__name__)
        if trivialAlpha:
            log.info('Pattern {0} has a trivial\nalpha-channel. Using standard masking procedure instead.\n'.format(filename))
            mask = makeMask(255-img.reshape(s)).reshape(img.shape)

    if useMask:
        #return nu.array((127-img)*mask,nu.int8).reshape(s)
        return img.reshape(s), nu.array(255*mask,nu.uint8).reshape(s)
    else:
        return nu.array(img,nu.uint8).reshape(s)

def writeImageDataOld(filename,data,color=(1,1,1),alphaChannel=None):
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

def writeImageData(filename,size,im_r=None,im_g=None,im_b=None,alphaChannel=None):
    #size = tuple(reversed(size))
    #img = Image.new('RGBA',size)
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

def writeImageDataOldOld(filename,data,size=None):
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
