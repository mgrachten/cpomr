#!/usr/bin/env python

import sys,os
import numpy as nu
from utilities import getter
from imageUtil import writeImageData, getPattern, findValleys, smooth, normalize
from scipy.stats import distributions

from agent import AgentPainter
from verticalSegment import VerticalSegment, identifyNonStaffSegments
from system import System
from staff import Staff
from bar import Bar

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

    def selectStaffs(self,staffs):
        # staffs get selected if their avg staffline distance (ASD) is
        # larger than thresholdPropOfMax times the largest ASD over all staffs
        #thresholdPropOfMax = .75
        maxStaffLineDistDev = .05
        print('original nr of staffs',len(staffs))
        slDists = nu.array([staff.getStaffLineDistance() for staff in staffs])
        print('avg staff line distance per staff:')
        print(slDists)
        # take the largest avg staff distance as the standard,
        # this discards mini staffs
        medDist = nu.median(slDists)
        print('staff line distances')
        staffs = [staff for staff in staffs if
                  nu.sum([nu.abs(x-medDist) for x in 
                          staff.getStaffLineDistances()])/(medDist*5) < maxStaffLineDistDev]
        #staffs = list(nu.array(staffs)[slDists >= thresholdPropOfMax*maxDist])
        #slDistStds = nu.array([staff.getStaffLineDistanceStd() for staff in staffs])
        #print('sd staff line distance per staff:')
        #print(slDistStds)
        #medStd = nu.median(slDistStds)
        #staffs = list(nu.array(staffs)[slDistStds <= .04*maxDist])
        print('new nr of staffs',len(staffs))
        return staffs

    @getter
    def getStaffs(self):
        staffs = []
        for i,vs in enumerate(self.getStaffSegments()):
            self.ap.paintHLine(vs.bottom)
            x = nu.arange(vs.top,vs.bottom)
            self.ap.paintRav(nu.column_stack((x,i*2*nu.ones(len(x)))),color=(10,10,10))
            print('vs',i,vs.top,vs.bottom)
            print('Processing staff segment {0}'.format(i))
            #vs.draw = i==2
            staffs.extend([Staff(self,s,vs.top,vs.bottom) for s in vs.getStaffLines()])
        staffs = self.selectStaffs(staffs)
        for staff in staffs:
            print(staff)
            staff.draw()
        self.ap.writeImage('tst.png')
        self.ap.reset()
        if len(staffs)%2 != 0:
            print('WARNING: detected unequal number of staffs!')
            print(self.fn)
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

        bf = []
        for i,system in enumerate(self.getSystems()):
            if True: #i==1: 
                sys.stdout.write('drawing system {0}\n'.format(i))
                sys.stdout.flush()
                #system.dodraw = True
                system.draw()
                sysSegs.append(system.getCorrectedImgSegment())
                barAgents = [x.agent for x in system.getBars()]
                
                #est = []
                #for j,a in enumerate(barAgents):
                #    b = Bar(system,a)
                #    bf.append(nu.append(system.getRotator().derotate(a.getDrawMean().reshape((1,2)))[0,:],b.getFeatures()))
                    #est.append([j]+list(1000*b.getEstimates()))
                #nu.savetxt('/tmp/s.txt',nu.array(bf).astype(nu.int),fmt='%d')
                #nu.savetxt('/tmp/est.txt',nu.array(est).astype(nu.int),fmt='%d')

                for a in barAgents:
                    self.ap.register(a)
                    b0 = system.getTop()-system.getRotator().derotate(a.getDrawMean().reshape((1,2)))[0,0]
                    b1 = system.getBottom()-system.getRotator().derotate(a.getDrawMean().reshape((1,2)))[0,0]
                    #self.ap.drawAgent(a,-300,300,system.getRotator())
                    self.ap.drawAgent(a,int(b0),int(b1),system.getRotator())
        bfname = os.path.join('/tmp/',os.path.splitext(os.path.basename(self.fn))[0]+'-barfeatures.txt')
        nu.savetxt(bfname,nu.array(bf),fmt='%d')
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
