#!/usr/bin/env python

import sys
import numpy as nu
from utilities import getter
from imageUtil import writeImageData, getPattern, findValleys, smooth, normalize
from scipy.stats import distributions

from agent import AgentPainter
from verticalSegment import VerticalSegment, identifyNonStaffSegments
from system import System
from staff import Staff

class ScoreImage(object):
    def __init__(self,fn):
        self.fn = fn
        self.typicalNrOfSystemPerPage = 6
        self.maxAngle = 1.5/180.
        self.nAnglebins = 600
        self.colGroups = 11
        self.bgThreshold = 20
        self.ap = AgentPainter(self.getImg())

    @getter
    def getImg(self):
        print('Loading image...'),
        sys.stdout.flush()
        try:
            img = 255-getPattern(self.fn,False,False)
        except IOError as e: 
            print('problem')
            raise e
        print('Done')
        img[img< self.bgThreshold] = 0
        return img

    def getWidth(self):
        return self.getImg().shape[1]
    def getHeight(self):
        return self.getImg().shape[0]

    @getter
    def getStaffs(self):
        # staffs get selected if their avg staffline distance (ASD) is
        # larger than thresholdPropOfMax times the largest ASD over all staffs
        thresholdPropOfMax = .75
        staffs = []
        for i,vs in enumerate(self.getStaffSegments()):
            print('Processing staff segment {0}'.format(i))
            staffs.extend([Staff(self,s,vs.top,vs.bottom) for s in vs.getStaffLines()])
        for staff in staffs:
            print(staff)
        slDists = nu.array([staff.getStaffLineDistance() for staff in staffs])
        print('avg staff line distance per staff:')
        print(slDists)
        maxDist = nu.max(slDists)
        print('original nr of staffs',len(staffs))
        staffs = list(nu.array(staffs)[slDists >= thresholdPropOfMax*maxDist])
        print('new nr of staffs',len(staffs))
        if len(staffs)%2 != 0:
            print('WARNING: detected unequal number of staffs!')
            print('TODO: retry to find an equal number of staffs')
        return staffs

    @getter
    def getSystems(self):
        staffs = self.getStaffs()
        assert len(staffs)%2 == 0
        return [System(self,(staffs[i],staffs[i+1])) for i in range(0,len(staffs),2)]

    def drawImage(self):
        # draw segment boundaries
        for i,vs in enumerate(self.getVSegments()):
            self.ap.paintHLine(vs.bottom,step=2)

        for vs in self.getNonStaffSegments():
            for j in range(vs.top,vs.bottom,4):
                self.ap.paintHLine(j,alpha=0.9,step=4)
                
        sysSegs = []
        for i,system in enumerate(self.getSystems()):
            if i == 0:
                sys.stdout.write('drawing system {0}\n'.format(i))
                sys.stdout.flush()
                system.draw()
                sysSegs.append(system.getCorrectedImgSegment())
                barAgents = system.getBarLines()
                for a in barAgents:
                    self.ap.register(a)
                    self.ap.drawAgent(a,-300,300,system.getRotator())
        self.ap.writeImage(self.fn)
        if True:
            return True
        shapes = nu.array([ss.shape for ss in sysSegs])
        ssH = nu.sum(shapes[:,0])
        ssW = nu.max(shapes[:,1])
        ssImg = nu.zeros((ssH,ssW),nu.uint8)
        x0 = 0
        for ss in sysSegs:
            h,w = ss.shape
            horzOffset = 0#int(nu.floor((ssW-w)/2.))
            ssImg[x0:x0+h,horzOffset:horzOffset+w] = ss
            x0 += h
        ap1 = AgentPainter(ssImg)
        ap1.writeImage(self.fn.replace('.png','-corr.png'))


    @getter
    def getHSums(self):
        return nu.sum(self.getImg(),1)

    @getter
    def getVSegments(self):
        K = int(self.getHeight()/(2*self.typicalNrOfSystemPerPage))+1
        sh = smooth(self.getHSums(),K)
        #nu.savetxt('/tmp/vh1.txt',nu.column_stack((self.getHSums(),sh)))
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
        return [vs for vs in self.getVSegments() if not vs.hasStaff()]

    def getStaffSegments(self):
        return [vs for vs in self.getVSegments() if vs.hasStaff()]
        #d = partition(lambda x: x.hasStaff(),self.getVSegments())
        #return d[True], d[False]

    @getter    
    def getWeights(self):
        globalAngleHist = smooth(nu.sum(nu.array([s.getAngleHistogram() for s 
                                                  in self.getVSegments()]),0),50)
        angles = nu.linspace(-self.maxAngle,self.maxAngle,self.nAnglebins+1)[:-1] +\
            (float(self.maxAngle)/self.nAnglebins)

        amax = angles[nu.argmax(globalAngleHist)]
        return distributions.norm(amax,.5/180.0).pdf(angles)


if __name__ == '__main__':
    pass
