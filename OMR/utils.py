#!/usr/bin/env python

import sys
import numpy as nu

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


if __name__ == '__main__':
    pass
