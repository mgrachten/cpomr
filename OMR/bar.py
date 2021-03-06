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

import logging
import numpy as nu
from misc.utilities import cachedProperty
from imageUtil import getAntiAliasedImg,findValleys,findPeaks,smooth,Rotator
from agentPainter import AgentPainter
import scipy.stats

logging.basicConfig(format='%(levelname)s: [%(name)s] %(message)s',level=logging.INFO)

class Bar(object):
    def __init__(self,scoreImg,kl1,kl2):
        self.scoreImg = scoreImg
        self.sys1,self.line1 = kl1
        self.sys2,self.line2 = kl2

    def getBarline1(self):
        return self.scoreImg.systems[self.sys1].barLines[self.line1]

    def getBarline2(self):
        return self.scoreImg.systems[self.sys2].barLines[self.line2]

    @cachedProperty
    def cornerCoordinates(self):
        k = self.sys1
        l = self.line1
        bbs = [self.scoreImg.systems[k].rotator.derotate(self.scoreImg.systems[k].barLines[l].barVCoords[nu.array((0,-1))])]
        l += 1
        while k < self.sys2:
            n = len(self.scoreImg.systems[k].barLines)
            if l == 0 and self.scoreImg.systems[k].barLines[l].estimatedType != LeftBarLine:
                # TODO: add coordinates at beginning of bar, if there's no barline there
                #lastVCoords[:,1] = self.scoreImg.systems[k].leftRight[1]
                #bbs.append(self.scoreImg.systems[k].rotator.derotate(lastVCoords))
                pass
            while l < n:
                bbs.append(self.scoreImg.systems[k].rotator.derotate(self.scoreImg.systems[k].barLines[l].barVCoords[nu.array((0,-1))]))
                l += 1
            if self.scoreImg.systems[k].barLines[-1].estimatedType != RightBarLine:
                lastVCoords = self.scoreImg.systems[k].barLines[-1].barVCoords[nu.array((0,-1))]
                lastVCoords[:,1] = self.scoreImg.systems[k].leftRight[1]
                bbs.append(self.scoreImg.systems[k].rotator.derotate(lastVCoords))
            k += 1
            l = 0
        # k == self.sys2, l == 0
        while l < self.line2:
            bbs.append(self.scoreImg.systems[k].rotator.derotate(self.scoreImg.systems[k].barLines[l].barVCoords[nu.array((0,-1))]))
            l += 1
        bbs.append(self.scoreImg.systems[k].rotator.derotate(self.scoreImg.systems[k].barLines[l].barVCoords[nu.array((0,-1))]))
        return bbs

    @cachedProperty
    def boundingBoxes(self):
        cc = self.cornerCoordinates
        bb = []
        for i in range(0,len(cc),2):
            topleft = cc[i][0,:].astype(nu.int)
            botright = cc[i+1][1,:].astype(nu.int)
            bb.append((topleft,botright))
        return nu.array(bb).ravel()

    def drawAsRect(self,k=0,color=(100,100,100),alpha=.5):
        cc = self.boundingBoxes.reshape((-1,4))
        for c in cc:
            self.scoreImg.ap.paintRectangle(c[:2],c[2:],color,alpha)
            
    def drawText(self,text,size=2,color=(100,100,100),alpha=.5,position='above'):
        #w = int(5*nu.mean([s.staffLineDistance for s in self.scoreImg.systems]))
        sld = self.scoreImg.systems[self.sys1].staffLineDistance
        w = int(3*sld)
        textSize = max(10,int(size*sld))
        if position == 'above':
            pos = nu.array([-w,w])+self.cornerCoordinates[0][0,:]
        else:
            pos = nu.array([w,w])+self.cornerCoordinates[0][1,:]-nu.array([w+textSize,0])/4.
        self.scoreImg.ap.drawText(text,pos.astype(nu.int),size = textSize,color=color,alpha=alpha)
        
    def draw(self,k=0,ptoggle=True):
        #b0 = self.getBBs()[0]
        bb = self.cornerCoordinates
        alpha = .9
        color = (150,0,0)
        w = int(5*nu.mean([s.staffLineDistance for s in self.scoreImg.systems]))
        above = nu.array([w,0])
        below = nu.array([w,0])

        b0 = bb[0]
        textSize = max(10,int(.3*w))
        self.scoreImg.ap.drawText('{0}'.format(k),b0[1,:]+below+(nu.array([-w-textSize,w])/4.).astype(nu.int),
                                  size = textSize,color=color,alpha=alpha)
        self.scoreImg.ap.paintLineSegment(b0[0,:]-above,b0[1,:]+below,color=color,alpha=alpha)

        for i in range(0,len(bb)-1,2):
            b0 = bb[i]
            b1 = bb[i+1]
            if ptoggle:
                coord1 = b0[0,:]-above
                coord2 = b1[0,:]-above
            else:
                coord1 = b0[1,:]+below
                coord2 = b1[1,:]+below
            self.scoreImg.ap.paintLineSegment(coord1,coord2,color=color,alpha=alpha)

class LeftBarLine(object): pass
class RightBarLine(object): pass
class MiddleBarLine(object): pass

class BarCandidate(object):
    BAR = 0
    DOUBLE_BAR = 1
    INVALID = 2
    
    def __init__(self,system,agent):
        self.agent = agent
        self.system = system

        # proportion of the system img segment height that the
        # barline maybe shifted vertically to find the best fit
        self.kRange = .1
        self.widthFactor = 8 # neighbourhood is widthFactor times barlineWidth wide
        self.heightFactor = 1.2 # neighbourhood is heightFactor times systemHeight high
        self.rotator = None
        self.doubleGapFactor = .1 # treat as double bar if diffSums has an internal gap > (dgp * main gap)

    @cachedProperty
    def features(self):
        img = self.neighbourhood.astype(nu.float)
        staffLineWidth = self.system.staffLineWidth # avg stafflinewidth
        staffLineDist = self.system.staffLineDistance # avg staffline distance
        # TODO: apply naming convention for properties
        staffTops = nu.array(self.getVerticalStaffLinePositions).astype(nu.float)
        hcoords = nu.array(self.barHCoordsLocal)
        t0,b0,t1,b1 = nu.round(staffTops[nu.array((0,4,5,9))])

        barMarginV = nu.round(max(1,.5*staffLineWidth))
        barMarginH = nu.round(max(1,.5*staffLineWidth))

        if len(hcoords) == 2:
            l,r = hcoords.astype(nu.int)
            barMarginH = max(0,min(barMarginH,nu.floor(((r-l)-1)/2.0)))
            coordsH = nu.arange(l+barMarginH,r-barMarginH).astype(nu.int)
        else:
            l0,r0,l1,r1 = hcoords.astype(nu.int)
            barMarginH = max(0,min(barMarginH,min(nu.floor(((r0-l0)-1)/2.0),nu.floor(((r1-l1)-1)/2.0))))
            coordsH = nu.append(nu.arange(l0+barMarginH,r0-barMarginH),
                                    nu.arange(l1+barMarginH,r1-barMarginH)).astype(nu.int)

        # the part of the barline in the upper and lower staffs (black)
        coordsV = nu.append(nu.arange(t0+barMarginV,b0-barMarginV),
                            nu.arange(t1+barMarginV,b1-barMarginV)).astype(nu.int)

        crit1 = nu.mean(img[coordsV,:][:,coordsH])/255.

        coordsV = nu.arange(b0+barMarginV,t1-barMarginV).astype(nu.int)
        # the part of the barline between the staffs (black)
        crit2 = nu.mean(img[coordsV,:][:,coordsH])/255.

        # the part above and below the barline (white)
        height = round(staffLineDist)
        coordsV = nu.append(nu.arange(t0-height-barMarginV,t0-barMarginV),
                            nu.arange(b1+barMarginV,b1+barMarginV+height)).astype(nu.int)
        crit3 = nu.mean(img[coordsV,:][:,coordsH])/255.

        # space between staffs (white)
        width = round(staffLineWidth)
        interStafflineMargin = 2
        be = nu.column_stack((nu.floor(staffTops)[1:]-interStafflineMargin,
                              interStafflineMargin+nu.ceil(staffTops+staffLineWidth)[:-1]))
        be = nu.vstack((be[:4],be[5:]))
        coordsV = nu.hstack([nu.arange(e,b) for b,e in be]).astype(nu.int)
        b,e = int(hcoords[0]),int(hcoords[-1])
        coordsH = nu.append(nu.arange(b-width,b)-interStafflineMargin,
                            nu.arange(e,e+width)+interStafflineMargin).astype(nu.int)
        N = len(coordsH)
        mid = (N-1)/2.0
        # weight pixels by their (inv) distance to the barline
        weights = scipy.stats.norm.pdf(nu.arange(N),mid,N/8.0)
        weights = weights/nu.sum(weights)
        sums = nu.sum(img[coordsV,:],0)[coordsH]
        sums = sums/(255*len(coordsV))
        crit0 = nu.sum(weights*sums)
        return nu.array([crit0,crit1,crit2,crit3])

    @cachedProperty
    def confidenceOrig(self):
        c0intercept = -13.09
        class0 = nu.array([-9.08,11.99,7.53,-1.99])
        return nu.dot(self.features,class0)+c0intercept

    @cachedProperty
    def confidence(self):
        c0intercept = -13.09
        class0 = nu.array([-9.08,11.99,7.53,-1.99])
        extraFeature = self.features[0]*(1-self.features[1])-.5
        extraFeatureWeight = .1
        return nu.dot(self.features,class0)+c0intercept+extraFeature*extraFeatureWeight

    def refine(self):
        # vertically adjust neighbourhood to center system
        self.neighbourhood = self._getNeighbourhood(nu.array((self.vCorrection,0.0)))
        if self.neighbourhood == None:
            return False
        hi0,lo0,hi1,lo1 = self.barVCoordsLocal.astype(nu.int)
        hcoords = self.barHCoordsLocal.astype(nu.int)
        h = lo1-hi0
        img = self.neighbourhood.astype(nu.float)
        # compute the std/mean blackness of the hypothesized bar
        nstd = [nu.std(img[hi0:lo1,hcoords[0]+1:hcoords[1]])/nu.mean(img[hi0:lo1,hcoords[0]+1:hcoords[1]])]
        if len(hcoords) == 4:
            nstd.append(nu.std(img[hi0:lo1,hcoords[2]+1:hcoords[3]])/nu.mean(img[hi0:lo1,hcoords[2]+1:hcoords[3]]))
        nstd = nu.array(nstd)
        # if the std/mean blackness is too high, decide this is not a bar
        self.invalid = nu.max(nstd) > .5
        return not self.invalid

    @cachedProperty
    def diffSums(self):
        return nu.sum(nu.diff(self.neighbourhood.astype(nu.float),axis=1),0)

    @cachedProperty    
    def barVCoordsLocal(self):
        """Assumes vCorrection has been applied to self.neighbourhood
        """
        staff0Top,staff0Bot = self.system.staffs[0].topBottomLeft
        staff1Top,staff1Bot = self.system.staffs[1].topBottomLeft
        w = staff1Bot-staff0Top
        offset = (self.neighbourhood.shape[0]-w)/2.0
        return nu.array([staff0Top,staff0Bot,staff1Top,staff1Bot])-staff0Top+offset

    @cachedProperty
    def barVCoords(self):
        assert self.rotator != None
        # TODO: check if using zero for other coordinate is the right thing to do
        # alternative: use HCoords
        offset = self.neighbourhood.shape[0]/2.0
        return self.rotator.derotate(nu.column_stack((self.barVCoordsLocal-offset,
                                                      nu.zeros(self.barVCoordsLocal.shape[0]))))

    @cachedProperty
    def barHCoordsLocal(self):
        mid = self.diffSums.shape[0]/2.0
        self.weights = scipy.stats.norm.pdf(nu.arange(self.diffSums.shape[0]),mid,self.agent.getLineWidth())
        self.curve = self.diffSums*self.weights
        peaks = findPeaks(self.curve)
        valleys = findValleys(self.curve)
        lpeaks = peaks[peaks <= mid]
        rvalleys = valleys[valleys >= mid]
        if len(lpeaks) == 0 or len(rvalleys) == 0:
            log = logging.getLogger(__name__)
            log.warn('Could not process bar correctly, image resolution may be too low')
            whalf = self.agent.getLineWidth()/2.
            return nu.array((mid-whalf,mid+whalf)).astype(nu.int)
        hi = lpeaks[nu.argmax(self.curve[lpeaks])]
        lo = rvalleys[nu.argmin(self.curve[rvalleys])]
        npeaks = peaks[nu.logical_and(peaks>hi,peaks<lo)] 
        nvalleys = valleys[nu.logical_and(valleys>hi,valleys<lo)] 
        npeaks = npeaks[nu.argsort(self.curve[npeaks])[::-1]]
        nvalleys = nvalleys[nu.argsort(self.curve[nvalleys])]
        isDouble = False
        if len(npeaks) > 0 and len(nvalleys) > 0 and npeaks[0]>nvalleys[0]:
            mainGap = self.curve[hi]-self.curve[lo]
            lo1 = nvalleys[0]
            hi1 = npeaks[0]
            subGap = self.curve[hi1]-self.curve[lo1]
            isDouble = subGap/mainGap > self.doubleGapFactor
            if isDouble:
                return nu.array((hi,lo1+1,hi1,lo+1))
        return nu.array((hi,lo+1))

    @cachedProperty
    def barHCoords(self):
        assert self.rotator != None
        # TODO: check if using zero for other coordinate is the right thing to do
        # alternative: use VCoords
        return self.rotator.derotate(nu.column_stack((nu.zeros(self.barHCoordsLocal.shape[0]),self.barHCoordsLocal)))

    @cachedProperty
    def estimatedType(self):
        """
        Estimate whether this candidate is a LEFT, MIDDLE, RIGHT or INVALID bar. It
        takes into account all barcandidates on the page, and for that it depends on
        self.approximateNeighbourhood (watch out for cyclic dependencies)
        """
        
        if self.approximateNeighbourhood == None:
            return None
        lrmedian = nu.median(nu.array([x for y in [[bc.leftRightAbsDiffSums for bc in system.barCandidates
                                                    if bc.approximateNeighbourhood != None] 
                                       for system in self.system.scrImage.systems] for x in y]),0)
        statistic = nu.array(self.leftRightAbsDiffSums)
        statistic[statistic != 0] /= lrmedian[statistic != 0]
        if statistic[0] < .5:
            if statistic[1] < .5:
                return None
            else:
                return LeftBarLine
        else:
            if statistic[1] < .5:
                return RightBarLine
            else:
                return MiddleBarLine
    
    def _leftRightAbsDiffSums(self,lrs):
        assert self.approximateNeighbourhood != None
        bimg = self.approximateNeighbourhood.astype(nu.float)
        ll,lr,rl,rr = lrs
        hsumsl = nu.sum(bimg[:,ll:lr],1)**2
        hsumsr = nu.sum(bimg[:,rl:rr],1)**2
        hsl = nu.abs(nu.diff(hsumsl - nu.mean(hsumsl)))
        hsr = nu.abs(nu.diff(hsumsr - nu.mean(hsumsr)))
        m = max(1,max(nu.max(hsl),nu.max(hsr)))
        return nu.sum(hsl)/m,nu.sum(hsr)/m
        
    @cachedProperty
    def leftRightAbsDiffSums(self):
        """Returns the (normalized) sum(diff(abs)) of the 
        horizontally summed left and right halves of the image.
        This helps distinguish between within-system, left-border 
        and right-border bars, respectively. The decision is best 
        made after normalizing values over all bar candidates in 
        the page.
        """
        assert self.approximateNeighbourhood != None
        W = self.approximateNeighbourhood.shape[1]
        w = .3*self.system.staffLineDistance
        N = int(nu.floor(W/2.0))
        lr = int(nu.round(max(0,N-.5*self.agent.getLineWidth())))
        ll = int(nu.round(max(0,lr-w)))
        rl = int(nu.round(min(W,N+.5*self.agent.getLineWidth())))
        rr = int(nu.round(min(W,rl+w)))
        return self._leftRightAbsDiffSums((ll,lr,rl,rr))

    @cachedProperty
    def neighbourhood(self):
        return self.approximateNeighbourhood

    @cachedProperty
    def approximateNeighbourhood(self):
        return self._getNeighbourhood()

    @cachedProperty
    def rotator(self):
        return self.rotator
        
    @cachedProperty
    def vCorrection(self):
        img = self.approximateNeighbourhood.astype(nu.float)
        l,r = self.barHCoordsLocal[nu.array((0,-1))]
        if self.estimatedType == LeftBarLine:
            img = img[:,l:]
        elif self.estimatedType == RightBarLine:
            img = img[:,:r]
        staff0Top,staff0Bot = self.system.staffs[0].topBottomLeft
        staff1Top,staff1Bot = self.system.staffs[1].topBottomLeft
        extra = 5
        L = int(nu.ceil(staff1Bot-staff0Top)+2*extra)
        comb = nu.zeros(L)
        comb[nu.round(nu.linspace(staff0Top,staff0Bot-1,5)-staff0Top+extra).astype(nu.int)] += 1
        comb[nu.round(nu.linspace(staff1Top,staff1Bot-1,5)-staff0Top+extra).astype(nu.int)] += 1
        comb = comb[::-1]
        conv = nu.convolve(nu.sum(img,1),comb,'valid')
        offset = nu.argmax(conv)
        vCorrection = offset-img.shape[0]/2.0+comb.shape[0]/2.0
        return vCorrection

    def _getNeighbourhood(self,midCorrection=None):
        # approximate system top and bottom (based on global staff detection)
        system0Top = self.system.rotator.rotate(self.system.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.system.rotator.rotate(self.system.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        sysHeight = (system1Bot-system0Top)
        yTop = self.agent.getDrawMean()[1]+(system0Top-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.angle)
        yBot = self.agent.getDrawMean()[1]+(system1Bot-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.angle)
        middle = (nu.array((system0Top,yTop))+nu.array((system1Bot,yBot)))/2.0
        if midCorrection != None:
            middle += midCorrection
        self.rotator = Rotator(self.agent.angle-.5,middle,nu.array((0,0.)))
        hhalf = int(sysHeight*self.heightFactor/2.)
        whalf = int(1.8*self.system.staffLineDistance)
        xx,yy = nu.mgrid[-hhalf:hhalf,-whalf:whalf]
        xxr,yyr = self.rotator.derotate(xx,yy)
        # check if derotated neighbourhood is inside image
        minx,maxx,miny,maxy = nu.min(xxr),nu.max(xxr),nu.min(yyr),nu.max(yyr)
        M,N = self.system.correctedImgSegment.shape
        # leave a border of one pixel for antialiasing:
        if minx < 1 or miny < 1 or maxx >= M-1 or maxy >= N-1:
            log = logging.getLogger(__name__)
            log.warn('Barline neighbourhood (center [{0},{1}]) does' \
                         'not fit inside system neighbourhood' \
                         ''.format(*self.system.rotator.derotate(middle.reshape((1,2)))[0,:].astype(nu.int)))
            return None
        cimg = getAntiAliasedImg(self.system.correctedImgSegment,xxr,yyr)
        return cimg

    @cachedProperty
    def getVerticalStaffLinePositions(self):
        # approximate system top and bottom (based on global staff detection)
        system0Top = self.system.rotator.rotate(self.system.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.system.rotator.rotate(self.system.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        sysHeight = (system1Bot-system0Top)
        hhalf = int(sysHeight*self.heightFactor/2.)
        sld = self.system.staffs[0].staffLineDistance
        krng = range(-int(nu.ceil(.5*sld)),int(nu.ceil(.5*sld)))
        hsums = nu.sum(self.neighbourhood,1)
        hslw = int(nu.floor(.5*self.system.staffLineWidth))

        inStaff = nu.arange(nu.round(self.system.staffLineWidth))

        staff0Tops = hhalf-.5*sysHeight-hslw+nu.arange(5)*sld
        staff1Tops = hhalf+.5*sysHeight-hslw-nu.arange(4,-1,-1)*sld
        staff0Rows = nu.round(nu.array([s+inStaff for s in staff0Tops]).flat[:]).astype(nu.int)
        staff1Rows = nu.round(nu.array([s+inStaff for s in staff1Tops]).flat[:]).astype(nu.int)
        f0 = krng[nu.argmax([nu.sum(hsums[staff0Rows+k]) for k in krng])]
        f1 = krng[nu.argmax([nu.sum(hsums[staff1Rows+k]) for k in krng])]
        return nu.append(staff0Tops+f0,staff1Tops+f1)
    
if __name__ == '__main__':
    pass
