#!/usr/bin/env python

import sys,os, pickle, logging
import numpy as nu
from utilities import cachedProperty
from imageUtil import writeImageData, getPattern, findValleys, smooth, normalize
from scipy.stats import distributions

from agentPainter import AgentPainter
from verticalSegment import VerticalSegment, identifyNonStaffSegments
from system import System
from staff import Staff
from bar import RightBarLine,LeftBarLine,Bar
from itertools import chain

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

        for k,system in enumerate(self.systems):
            system.leftRight

        ptoggle = False
        for k,bar in enumerate(self.bars):
            bar.draw(bar_start+k,ptoggle)
            ptoggle = not ptoggle
            #bbs = b.getBBs()
            # vertical lines:
            # first barline
            
    def barBBs(self,start_bar=0):
        bb = []
        for k,bar in enumerate(self.bars):
            print(bar.getBBs())

    @cachedProperty
    def filenameBase(self):
        return os.path.splitext(os.path.basename(self.fn))[0]

    def drawImage(self):
        """OBSOLETE"""
        if len(self.systems) == 0:
            log().warn('No systems found in image {0}'.format(self.fn))
            return False

        w = int(5*nu.mean([s.staffLineDistance for s in self.systems]))
        above = nu.array([w,0])
        below = nu.array([w,0])
        alpha = .9
        color = (150,0,0)
        goodColor = (0,150,0)
        badColor = (250,0,0)
        for k,b in enumerate(self.bars):
            bbs = b.getBBs()
            # vertical lines:
            # first barline
            color = goodColor if b.getBarline1().confidence > 0 else badColor
            self.ap.paintLineSegment(bbs[0][0,:]-above,bbs[0][1,:]+below,color=color,alpha=alpha)
            # number
            self.ap.drawText('{0}'.format(k),bbs[0][0,:]-above+(nu.array([w,w])/4.).astype(nu.int),
                             size = max(10,int(.3*w)),color=color,alpha=alpha)

            self.ap.drawText('{0:.1f}'.format(b.getBarline1().confidence),bbs[0][0,:]-above+(nu.array([2*w,0])/4.).astype(nu.int),
                             size = max(10,int(.3*w)),color=color,alpha=alpha)

            color = goodColor if b.getBarline2().confidence > 0 else badColor
            # last barline
            self.ap.paintLineSegment(bbs[-1][0,:]-above,bbs[-1][1,:]+below,color=color,alpha=alpha)
            
            if b.sys1 == b.sys2:
                # lower horizontal
                coord1 = bbs[0][0,:]
                coord2 = bbs[-1][0,:]
                self.ap.paintLineSegment(coord1-above,coord2-above,color=color,alpha=alpha)
                # upper horizontal
                coord1 = bbs[0][1,:]
                coord2 = bbs[-1][1,:]
                self.ap.paintLineSegment(coord1+below,coord2+below,color=color,alpha=alpha)
            
            #for bb in bbs:
            for i in range(len(bbs)-1):
                l1 = bbs[i]
                l2 = bbs[i+1]
                if i == len(bbs)-2: #l1.sys1 == l2.sys2:
                    endline = l2
                else:
                    hoffset = w
                    endline = l1+nu.array([0,hoffset])
                # lower horizontal
                coord1 = l1[0,:]
                coord2 = endline[0,:]
                self.ap.paintLineSegment(coord1-above,coord2-above,color=color,alpha=alpha)
                # upper horizontal
                coord1 = l1[1,:]
                coord2 = endline[1,:]
                self.ap.paintLineSegment(coord1+below,coord2+below,color=color,alpha=alpha)
                
        alpha = .1
        color = (10,150,10)

        for system in self.systems:
            #ap = AgentPainter(system.correctedImgSegment)
            M,N = system.correctedImgSegment.shape
            shrink = 3
            bb = nu.array([[1,1],
                           [M-shrink,N-shrink],
                           [1,N-shrink],
                           [M-shrink,1]])
            bbr = system.rotator.derotate(bb)
            self.ap.paintLineSegment(bbr[0,:],bbr[3,:],color=color,alpha=alpha)
            self.ap.paintLineSegment(bbr[1,:],bbr[2,:],color=color,alpha=alpha)
            self.ap.paintLineSegment(bbr[0,:],bbr[2,:],color=color,alpha=alpha)
            self.ap.paintLineSegment(bbr[1,:],bbr[3,:],color=color,alpha=alpha)
            #ap.writeImage('system-{0:02d}.png'.format(system.n))
        self.ap.writeImage(self.fn)            
        
if __name__ == '__main__':
    pass
