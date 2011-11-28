#!/usr/bin/env python

import sys,os
import numpy as nu
from utilities import getter
from agent import assignToAgents, mergeAgents, makeAgentClass
from imageUtil import normalize
from utils import selectColumns
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
    print('of {0} segments, items {1} were identified as non-staff'.format(len(vertSegments),nonStaff))
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

    def getVHSums(self):
        vsum = len(nu.nonzero(self.getVSums())[0])
        hsum = len(nu.nonzero(self.getHSums())[0])
        return vsum, hsum

    def flagNonStaff(self):
        self.containsStaff = False

    def hasStaff(self):
        return self.containsStaff
        
    @getter
    def getStaffLines(self):
        agents = []
        defAngle = self.getAngle()
        cols = selectColumns(self.getVSums(),self.colGroups)[0]
        StaffAgent = makeAgentClass(targetAngle=defAngle,
                                    maxAngleDev=2/180.,
                                    maxError=3,
                                    minScore=-2,
                                    offset=self.top)
        draw = False
        f0 = os.path.splitext(self.scrImage.fn)[0]
        print('default angle for this staff',defAngle)
        stop = False
        finalStage = False
        nFinalRuns = 10
        for i,c in enumerate(cols):
            if nFinalRuns == 0:
                break
            agentsnew,died = assignToAgents(self.getImgSegment()[:,c],agents,StaffAgent,
                                            self.scrImage.getWidth(),horz=c,fixAgents=finalStage)
            if len(agentsnew) > 3:
                agentsnew,d = mergeAgents(agentsnew)
            agents = agentsnew
            agents.sort(key=lambda x: -x.score)

            if len(agents) > 5 and i > 50 and not finalStage:
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
        return self.scrImage.getImg()[self.top:self.bottom,:]

    def getHSums(self):
        return self.scrImage.getHSums()[self.top:self.bottom]

    @getter
    def getVSums(self):
        return nu.sum(self.getImgSegment(),0)

    @getter
    def getAngleHistogram(self):
        #self.vSums = nu.sum(self.scrImage.getImg()[self.top:self.bottom,:],0)
        hparts = 3
        cols = selectColumns(self.getVSums(),hparts)[0]
        angles = []
        nColsToProcess = int(2*len(cols)/10)
        for j in range(nColsToProcess):
            dx = cols[j]-cols[j+1]
            dps,dy = getOffset(self.scrImage.getImg()[self.top:self.bottom,cols[j]],
                               self.scrImage.getImg()[self.top:self.bottom,cols[j+1]],
                               nu.abs(dx),self.maxAngle)
            if nu.min(dps) < 0.5: # min is only > 0 when dps has zero range
                angles.append((nu.arctan2(dy,dx)/nu.pi+.5)%1-.5)
        histrange = (-self.maxAngle,self.maxAngle)
        #bins,lims = nu.histogram(angles,bins=self.nbins,range=histrange)
        return nu.histogram(angles,bins=self.nAngleBins,range=histrange)[0]

    @getter
    def getAngle(self):
        """estimate the angle by taking the argmax of the angle histogram,
        weighted by the global angle histogram of the page.
        """
        i = nu.argmax(self.getAngleHistogram()*self.scrImage.getWeights())
        #nu.savetxt('/tmp/h{0}'.format(i),self.getAngleHistogram()*weights)
        return (float(self.maxAngle)/self.nAngleBins)-\
            self.maxAngle+i*2.0*self.maxAngle/self.nAngleBins


if __name__ == '__main__':
    pass
