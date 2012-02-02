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

import os
import logging
import numpy as nu
from utilities import cachedProperty
from agent import assignToAgents, mergeAgents, AgentConfig
from imageUtil import normalize, selectColumns
from staff import assessStaffLineAgents

def identifyNonStaffSegments(vertSegments,N,M):
    """Identify vertical segments that are unlikely to contain staffs
    """
    vhsums = nu.array([vs.getVHSums() for vs in vertSegments],nu.int)
    meds = nu.median(vhsums,0)
    ref = nu.array((N/vhsums.shape[0],M))
    nvhsums = vhsums-meds
    nvhsums[nvhsums>0] = 0
    dref = nu.sum(ref**2)**.5
    ndists = (nu.sum(nvhsums**2,1)**.5)/dref
    nonStaff = nu.nonzero(nu.logical_not(ndists < .2))[0]
    #nvhsums = nu.column_stack((nvhsums,vhsums[:,-1]))
    #nvhsums = nu.vstack((nvhsums,nu.append(-ref,-1)))
    #nu.savetxt('/tmp/ns.txt',nvhsums,fmt='%d')
    log = logging.getLogger(__name__)
    log.info('Of {0} segments, items {1} were identified as non-staff'.format(len(vertSegments),nonStaff))
    return nonStaff

def getOffset(v1,v2,dx,maxAngle):
    #kmax = min(70,int(1.5*nu.ceil(dx*nu.tan(maxAngle*nu.pi))))
    kmax = 50
    N = len(v1)
    dotproducts = []
    rng = []
    for i in range(-kmax,kmax+1):
        b = min(max(0,i),N)
        e = max(0,min(N,N-i))
        if e > 0:
            dotproducts.append(nu.dot(v1[:e],v2[b:]))
            rng.append(i)
    ndp = normalize(nu.array(dotproducts))
    return ndp, -rng[nu.argmax(ndp)]

class VerticalSegment(object):
    def __init__(self,scoreImage,top,bottom,colGroups=11,
                 maxAngle=2/180.,nAngleBins=300):
        self.scrImage = scoreImage
        self.top = top
        self.bottom = bottom
        self.maxAngle = maxAngle
        self.nAngleBins = nAngleBins
        self.colGroups = colGroups
        self.nPerStaff = 5
        self.containsStaff = True
        self.draw = False

    def getVHSums(self):
        vsum = len(nu.nonzero(self.vSums)[0])
        hsum = len(nu.nonzero(self.hSums)[0])
        return vsum, hsum

    def flagNonStaff(self):
        self.containsStaff = False

    def hasStaff(self):
        return self.containsStaff
        
    @cachedProperty
    def staffLines(self):
        agents = []
        defAngle = self.angle
        cols = selectColumns(self.vSums,self.colGroups)[0]
        staffAgentConfig = AgentConfig(targetAngle=defAngle,
                                       maxAngleDev=2/180.,
                                       minScore=-2,
                                       maxError=self.scrImage.getWidth()/2000.,
                                       offset=self.top)
        f0 = os.path.splitext(self.scrImage.fn)[0]
        log = logging.getLogger(__name__)
        log.info('Default angle for this staff: {0:.5f} rad/PI'.format(defAngle))
        stop = False
        finalStage = False
        nFinalRuns = 20
        draw = self.draw
        for i,c in enumerate(cols):
            if nFinalRuns == 0:
                break
            agentsnew,died = assignToAgents(self.getImgSegment()[:,c],agents,staffAgentConfig,
                                            self.scrImage.getWidth(),horz=c,fixAgents=finalStage)
            if len(agentsnew) > 3:
                agentsnew,d = mergeAgents(agentsnew)
            agents = agentsnew
            agents.sort(key=lambda x: -x.score)

            if len(agents) > 4 and i > 50 and not finalStage:
                finalStage,selection = assessStaffLineAgents(agents,self.scrImage.getWidth(),
                                                             self.nPerStaff)
                if finalStage:
                    agents = selection
            
            if finalStage:
                nFinalRuns -= 1

            if draw:
                self.scrImage.ap.reset()
                self.scrImage.ap.paintVLine(c)
                for a in agents:
                    if not self.scrImage.ap.isRegistered(a):
                        self.scrImage.ap.register(a)
                    self.scrImage.ap.drawAgentGood(a,-3000,3000)
                self.scrImage.ap.writeImage(f0+'-{0:04d}-c{1}'.format(i,c)+'.png')
        agents.sort(key=lambda x: x.getMiddle(self.scrImage.getWidth()))
        return [agents[k*self.nPerStaff:(k+1)*self.nPerStaff] 
                for k in range(len(agents)/self.nPerStaff)]

    def getImgSegment(self):
        return self.scrImage.img[self.top:self.bottom,:]

    @property
    def hSums(self):
        return self.scrImage.hSums[self.top:self.bottom]

    @cachedProperty
    def vSums(self):
        return nu.sum(self.getImgSegment(),0)

    @cachedProperty
    def angleHistogram(self):
        #self.vSums = nu.sum(self.scrImage.img[self.top:self.bottom,:],0)
        hparts = 3
        cols = selectColumns(self.vSums,hparts)[0]
        angles = []
        nColsToProcess = int(2*len(cols)/10)
        for j in range(nColsToProcess):
            dx = cols[j]-cols[j+1]
            dps,dy = getOffset(self.scrImage.img[self.top:self.bottom,cols[j]],
                               self.scrImage.img[self.top:self.bottom,cols[j+1]],
                               nu.abs(dx),self.maxAngle)
            if nu.min(dps) < 0.5: # min is only > 0 when dps has zero range
                angles.append((nu.arctan2(dy,dx)/nu.pi+.5)%1-.5)
        histrange = (-self.maxAngle,self.maxAngle)
        #bins,lims = nu.histogram(angles,bins=self.nbins,range=histrange)
        return nu.histogram(angles,bins=self.nAngleBins,range=histrange)[0]

    @cachedProperty
    def angle(self):
        """estimate the angle by taking the argmax of the angle histogram,
        weighted by the global angle histogram of the page.
        """
        i = nu.argmax(self.angleHistogram*self.scrImage.weights)
        #nu.savetxt('/tmp/h{0}'.format(i),self.angleHistogram*weights)
        return (float(self.maxAngle)/self.nAngleBins)-\
            self.maxAngle+i*2.0*self.maxAngle/self.nAngleBins


if __name__ == '__main__':
    pass
