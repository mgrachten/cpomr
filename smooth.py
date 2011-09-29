import numpy as nu
from scipy import signal

def normalize(x):
    xmin = nu.min(x)
    xmax = nu.max(x)
    assert xmin < xmax
    return (x-xmin)/(xmax-xmin)

def makeMask(img):
    mask = nu.zeros(img.shape,nu.float)
    K = 3
    N,M = img.shape
    for i in range(N):
        for j in range(M):
            o1 = max(0,i-K)
            o2 = max(0,j-K)
            subimage = img[o1:min(i+K,N),o2:min(j+K,M)]
            k = nu.argmax(subimage)
            v = k/subimage.shape[1]
            w = k%subimage.shape[1]
            d = 1.0/(1+((o1+v-i)**2+(o2+w-j)**2)**.5)
            mask[i,j] = subimage[v,w]
    return normalize(mask)
