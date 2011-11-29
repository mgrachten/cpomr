#!/usr/bin/env python

import sys
import numpy as nu
from utilities import getter
from imageUtil import getAntiAliasedImg
from utils import Rotator

def getVCenterOfBarline(bimg,kRange):
    """
    This function finds the vertical center of bar lines, by
    computing the product of the upper and lower half around
    the hypothetical center. When the hypothetical center is 
    correct, the product will be maximal/minimal (depending on
    the semantics of the color values).
    """
    N,M = bimg.shape
    kMin = int(N*(1-kRange)/2.0)
    kMax = int(N*(1+kRange)/2.0)
    scores = []
    krng = range(kMin,kMax+1)
    for k in krng:
        w = min(k,N-k-1)
        scores.append(nu.sum(((bimg[k-w:k,:]-127)*(bimg[k+w:k:-1,:]-127)))/w)
    return krng[nu.argmin(scores)]

class Bar(object):
    def __init__(self,system,agent):
        self.agent = agent
        self.system = system

        # proportion of the system img segment height that the
        # barline maybe shifted vertically to find the best fit
        self.kRange = .1
        self.widthFactor = 4 # neighbourhood is widthFactor times barlineWidth wide
        self.heightFactor = 1.2 # neighbourhood is heightFactor times systemHeight high

    @getter
    def getBarHCoords(self):
        b,e = (nu.array((-.5,.5))+self.widthFactor/2.0)*self.agent.getLineWidth()
        return nu.array((nu.floor(b),nu.ceil(e))).astype(nu.int)

    @getter
    def getNeighbourhood(self):

        # approximate system top and bottom (based on global staff detection)
        system0Top = self.system.getRotator().rotate(self.system.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.system.getRotator().rotate(self.system.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        sysHeight = (system1Bot-system0Top)

        whalf = int(self.agent.getLineWidth()*self.widthFactor/2.)
        hhalf = int(sysHeight*self.heightFactor/2.)

        yTop = self.agent.getDrawMean()[1]+(system0Top-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.getAngle())
        yBot = self.agent.getDrawMean()[1]+(system1Bot-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.getAngle())
        middle = (nu.array((system0Top,yTop))+nu.array((system1Bot,yBot)))/2.0
        r = Rotator(self.agent.getAngle()-.5,middle,nu.array((0,0.)))
        xx,yy = nu.mgrid[-hhalf:hhalf,-whalf:whalf]
        xxr,yyr = r.derotate(xx,yy)

        # check if derotated neighbourhood is inside image
        minx,maxx,miny,maxy = nu.min(xxr),nu.max(xxr),nu.min(yyr),nu.max(yyr)
        M,N = self.system.getCorrectedImgSegment().shape
        if minx < 0 or miny < 0 or maxx >= M or maxy >= N:
            return None

        cimg = getAntiAliasedImg(self.system.getCorrectedImgSegment(),xxr,yyr)
        vCorrection = getVCenterOfBarline(cimg,self.kRange)-hhalf
        # TODO: check if barneighbourhood is inside corrected system segment
        cimg = getAntiAliasedImg(self.system.getCorrectedImgSegment(),xxr+vCorrection,yyr)
        return cimg
    
    def findVerticalStaffLinePositions(self):
        # approximate system top and bottom (based on global staff detection)
        system0Top = self.system.getRotator().rotate(self.system.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.system.getRotator().rotate(self.system.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        sysHeight = (system1Bot-system0Top)
        hhalf = int(sysHeight*self.heightFactor/2.)
        sld = self.system.staffs[0].getStaffLineDistance()
        krng = range(-int(nu.ceil(.5*sld)),int(nu.ceil(.5*sld)))
        hsums = nu.sum(self.getNeighbourhood(),1)
        hslw = int(nu.floor(.5*self.system.getStaffLineWidth()))

        inStaff = nu.arange(nu.round(self.system.getStaffLineWidth()))

        staff0Tops = hhalf-.5*sysHeight-hslw+nu.arange(5)*sld
        staff1Tops = hhalf+.5*sysHeight-hslw-nu.arange(4,-1,-1)*sld
        staff0Rows = nu.round(nu.array([s+inStaff for s in staff0Tops]).flat[:]).astype(nu.int)
        staff1Rows = nu.round(nu.array([s+inStaff for s in staff1Tops]).flat[:]).astype(nu.int)
        f0 = krng[nu.argmax([nu.sum(hsums[staff0Rows+k]) for k in krng])]
        f1 = krng[nu.argmax([nu.sum(hsums[staff1Rows+k]) for k in krng])]
        return nu.append(staff0Tops+f0,staff1Tops+f1)

if __name__ == '__main__':
    pass
