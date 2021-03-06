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

import sys,os, pickle, logging
import numpy as nu
from scipy.stats import distributions

from misc.utilities import cachedProperty
from imageUtil import writeImageData, getPattern, findValleys, smooth, normalize
from agentPainter import AgentPainter
from verticalSegment import VerticalSegment, identifyNonStaffSegments
from system import System
from staff import Staff
from bar import RightBarLine,LeftBarLine,Bar

def log():
    return logging.getLogger(__name__)

class ScoreImage(object):
    def __init__(self,fn):
        self.fn = fn
        self.typicalNrOfSystemPerPage = 6
        self.maxAngle = 1.5/180.
        self.nAnglebins = 600
        self.colGroups = 11
        self.bgThreshold = 20

    @cachedProperty
    def ap(self):
        return AgentPainter(self.img)

    @cachedProperty
    def img(self):
        log().info('Loading image: {0}'.format(self.fn))
        try:
            img = 255-getPattern(self.fn,False,False)
        except IOError as e: 
            log().error('Problem loading image...')
            raise e
        return self.preprocessImage(img)

    def preprocessImage(self,img):
        imin,imax = nu.min(img),nu.max(img)
        istd = nu.std(img)
        imed = nu.median(img)
        minImageHeight = 1500
        minImageWidth = minImageHeight*.75
        if img.shape[1] < minImageWidth:
            log().warn('Image resolution may be too low for accurate OMR, recognition may be slow')
            log().warn('For good results, provide images with a width of at least {0} pixels'.format(int(minImageWidth)))
        if istd == 0:
            log().warn('Blank image')

        if imed > imin+(imax-imin)/2.:
            log().warn('Image appears inverted, Inverting image')
            img = 255-img
        log().info('White-thresholding image (threshold: {0:.1f} %)'.format(100*self.bgThreshold/255.))
        img[img< self.bgThreshold] = 0
        return img

    def getWidth(self):
        return self.img.shape[1]
    def getHeight(self):
        return self.img.shape[0]

    def selectStaffs(self,staffs):
        # staffs get selected if their avg staffline distance (ASD) is
        # larger than thresholdPropOfMax times the largest ASD over all staffs

        maxStaffLineDistDev = .05
        slDists = nu.array([staff.staffLineDistance for staff in staffs])
        #log.info('avg staff line distance per staff:')
        # take the largest avg staff distance as the standard,
        # this discards mini staffs
        medDist = nu.median(slDists)
        staffs = [staff for staff in staffs if
                  nu.sum([nu.abs(x-medDist) for x in 
                          staff.staffLineDistances])/(medDist*5) < maxStaffLineDistDev]
        origNrStaffs = len(staffs)

        log().info('Selecting {0} staffs from candidate list, discarding {1} staff(s)'.format(len(staffs),
                                                                                         origNrStaffs-len(staffs)))
        return staffs

    @cachedProperty
    def staffs(self):
        draw = False
        staffs = []

        for i,vs in enumerate(self.getStaffSegments()):
            #self.ap.paintHLine(vs.bottom)
            x = nu.arange(vs.top,vs.bottom)
            #self.ap.paintRav(nu.column_stack((x,i*2*nu.ones(len(x)))),color=(10,10,10))
            log().info('Processing staff segment {0}'.format(i))
            #vs.draw = i==2
            staffs.extend([Staff(self,s,vs.top,vs.bottom) for s in vs.staffLines])

        staffs = self.selectStaffs(staffs)

        if len(staffs)%2 != 0:
            log().warn('Detected unequal number of staffs for file:\n\t{0}'.format(self.fn))
            log().info('TODO: retry to find an equal number of staffs')

        if draw:
            for staff in staffs:
                staff.draw()
            self.ap.writeImage('tst.png')
            self.ap.reset()

        return staffs

    @cachedProperty
    def systems(self):
        staffs = self.staffs
        if len(staffs)%2 != 0:
            log().critical('Cannot deal with unequal number of staffs under the current assumption of double staff systems')
            log().warn('Discarding any detected staffs')
            return []
        else:
            return [System(self,(staffs[i],staffs[i+1]),i/2) for i in range(0,len(staffs),2)]


    def drawCorrectedImage(self):
        fn = os.path.splitext(os.path.basename(self.fn))[0]
        if len(self.systems) == 0:
            log().critical('No systems found, cannot draw corrected image')
            return False
        shapes = nu.array([system.correctedImgSegment.shape for system in self.systems])
        ssH = nu.sum(shapes[:,0])
        ssW = nu.max(shapes[:,1])
        ssImg = nu.zeros((ssH,ssW),nu.uint8)
        x0 = 0
        for system in self.systems:
            h,w = system.correctedImgSegment.shape
            horzOffset = 0#int(nu.floor((ssW-w)/2.))
            ssImg[x0:x0+h,horzOffset:horzOffset+w] = system.correctedImgSegment
            x0 += h
        ap1 = AgentPainter(ssImg)
        ap1.writeImage(fn+'-corrected.png')

    @cachedProperty
    def hSums(self):
        return nu.sum(self.img,1)

    @cachedProperty
    def vSegments(self):
        K = int(self.getHeight()/(2*self.typicalNrOfSystemPerPage))+1
        sh = smooth(self.hSums,K)
        #nu.savetxt('/tmp/vh1.txt',nu.column_stack((self.hSums,sh)))
        segBoundaries = nu.append(0,nu.append(findValleys(sh),self.getHeight()))
        vsegments = []
        for i in range(len(segBoundaries)-1):
            vsegments.append(VerticalSegment(self,segBoundaries[i],segBoundaries[i+1],
                                             colGroups = self.colGroups,
                                             maxAngle = self.maxAngle,
                                             nAngleBins = self.nAnglebins))
        nonStaff = identifyNonStaffSegments(vsegments,self.getHeight(),self.getWidth())
        for i in nonStaff:
            vsegments[i].flagNonStaff()
        return vsegments

    def getNonStaffSegments(self):
        return [vs for vs in self.vSegments if not vs.hasStaff()]

    def getStaffSegments(self):
        return [vs for vs in self.vSegments if vs.hasStaff()]
        #d = partition(lambda x: x.hasStaff(),self.vSegments)
        #return d[True], d[False]

    @cachedProperty
    def weights(self):
        globalAngleHist = smooth(nu.sum(nu.array([s.angleHistogram for s 
                                                  in self.vSegments]),0),50)
        angles = nu.linspace(-self.maxAngle,self.maxAngle,self.nAnglebins+1)[:-1] +\
            (float(self.maxAngle)/self.nAnglebins)

        amax = angles[nu.argmax(globalAngleHist)]
        return distributions.norm(amax,.5/180.0).pdf(angles)

    @cachedProperty
    def bars(self):
        """
        Return bars
        """
        bars = []
        bl = [(i,j) for i in range(len(self.systems)) for j in range(len(self.systems[i].barLines))]
        i1,i2 = 0,1
        while i2 < len(bl):
            k1,l1 = bl[i1]
            k2,l2 = bl[i2]
            bl1 = self.systems[k1].barLines[l1]
            bl2 = self.systems[k2].barLines[l2]
            if (bl1.estimatedType != RightBarLine and
                bl2.estimatedType != LeftBarLine):
                bars.append(Bar(self,(k1,l1),(k2,l2)))
                i1 = i2
            else:
                if bl1.estimatedType == RightBarLine:
                    i1 += 1
            i2 += 1
        return bars

    def drawAnnotatedScore(self,bar_start=0):
        if len(self.systems) == 0:
            log().warn('No systems found in image {0}'.format(self.fn))
            return False
        color1 = (100,0,100)
        color2 = (0,200,0)
        alpha = .2
        textcolor = (150,0,0)
        for k,bar in enumerate(self.bars):
            color = color1 if (bar_start+k)%2 == 0 else color2
            #bar.draw(bar_start+k,ptoggle)
            bar.drawAsRect(bar_start+k,color,alpha)
            bar.drawText('{0}'.format(bar_start+k),color=textcolor,alpha=1)


    def drawAnnotatedScore2(self,bar_start=0):
        if len(self.systems) == 0:
            log().warn('No systems found in image {0}'.format(self.fn))
            return False

        bb = [[bar_start+i]+list(bar.boundingBoxes) for i,bar in enumerate(self.bars)]
        ptoggle = False
        color1 = (100,10,50)
        color2 = (10,100,50)
        alpha = .3
        for k,bar in enumerate(bb):
            #bar.draw(bar_start+k,ptoggle)
            i = bar[0]
            cc = nu.array(bar[1:]).reshape((-1,4))
            color = color1 if ptoggle else color2
            for c in cc:
                self.ap.paintRectangle(c[:2],c[2:],color,alpha)
            ptoggle = not ptoggle
            
    @cachedProperty
    def filenameBase(self):
        return os.path.splitext(os.path.basename(self.fn))[0]

    #     for system in self.systems:
    #         #ap = AgentPainter(system.correctedImgSegment)
    #         M,N = system.correctedImgSegment.shape
    #         shrink = 3
    #         bb = nu.array([[1,1],
    #                        [M-shrink,N-shrink],
    #                        [1,N-shrink],
    #                        [M-shrink,1]])
    #         bbr = system.rotator.derotate(bb)
    #         self.ap.paintLineSegment(bbr[0,:],bbr[3,:],color=color,alpha=alpha)
    #         self.ap.paintLineSegment(bbr[1,:],bbr[2,:],color=color,alpha=alpha)
    #         self.ap.paintLineSegment(bbr[0,:],bbr[2,:],color=color,alpha=alpha)
    #         self.ap.paintLineSegment(bbr[1,:],bbr[3,:],color=color,alpha=alpha)
    #         #ap.writeImage('system-{0:02d}.png'.format(system.n))
    #     self.ap.writeImage(self.fn)            
        
if __name__ == '__main__':
    pass
