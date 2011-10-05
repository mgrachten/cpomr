#!/usr/bin/env python

import sys,os
from PIL import Image
import numpy as nu
from scipy import signal

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
    mask = nu.zeros(img.shape,nu.float)
    K = 3
    N,M = img.shape
    for i in range(N):
        for j in range(M):
            o1 = max(0,i-K)
            o2 = max(0,j-K)
            subimage = img[o1:min(i+K,N),o2:min(j+K,M)]
            k = float(nu.argmax(subimage))
            v = k/subimage.shape[1]
            w = k%subimage.shape[1]
            #d = 1.0/(1+((o1+v-i)**2+(o2+w-j)**2)**.5)
            mask[i,j] = subimage[v,w]
    return normalize(mask)

def getImageData(filename):
    imageFh = Image.open(filename)
    #data = nu.array(list(imageFh.getdata()))
    s = list(imageFh.size)
    s.reverse()
    data = nu.array(imageFh.getdata(),nu.uint8).reshape(tuple(s))
    return data

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
