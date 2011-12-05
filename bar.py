#!/usr/bin/env python

import sys
import numpy as nu
from itertools import chain,product
from utilities import getter
from imageUtil import getAntiAliasedImg
from utils import Rotator
from agent import AgentPainter

def getHCenterAndWidth(bimg):
    crossSection = nu.median(bimg,0)
    avg = nu.average(nu.arange(len(crossSection)),weights=crossSection)
    center = int(nu.round(avg))
    total = nu.sum(crossSection)
    half = (total-crossSection[center])/2.0
    thr = .9
    for left in xrange(center,0,-1):
        if nu.sum(crossSection[left:center])/half > thr or crossSection[left] == 0:
            left +=1
            break
    for right in xrange(center+1,len(crossSection)):
        if nu.sum(crossSection[center+1:right])/half > thr or crossSection[right] == 0:
            break
    return nu.array(left,center,right)

def getVCenterOfBarline(bimg,kRange,save=False):
    """
    This function finds the vertical center of bar lines, by
    computing the product of the upper and lower half around
    the hypothetical center. When the hypothetical center is 
    correct, the product will be maximal/minimal (depending on
    the semantics of the color values).
    """
    hhalf = 0
    N,M = bimg.shape
    print(nu.max(bimg),nu.min(bimg))
    rsums = nu.sum(bimg.astype(nu.float),1)
    m = (nu.max(rsums)-nu.min(rsums))/2.0
    rsums -= m
    kMin = int(N*(1-kRange)/2.0)
    kMax = int(N*(1+kRange)/2.0)
    scores = []
    krng = range(kMin,kMax+1)
    for k in krng:
        w = min(k,N-k-1)
        #scores.append(nu.sum(((bimg[k-w:k,:]-127)*(bimg[k+w:k:-1,:]-127)))/w)
        scores.append(nu.sum(rsums[k-w:k]*rsums[k+w:k:-1])/w)
    if save:
        nu.savetxt('/tmp/s.txt',nu.column_stack((nu.array(krng)-hhalf,scores)))
        nu.savetxt('/tmp/img.txt',(rsums+m)/255.)
        b,c = nu.histogram(bimg.reshape((-1,1)))
        nu.savetxt('/tmp/h.txt',nu.column_stack((b,(c[:-1]+c[1:])/2.0)))
        nu.savetxt('/tmp/ssums.txt',nu.sort((rsums+m)/255.))
        nu.savetxt('/tmp/simg.txt',bimg[nu.argsort(rsums),:])
        nu.savetxt('/tmp/med.txt',nu.median(bimg,0))
    return krng[nu.argmax(scores)]

class Bar(object):
    def __init__(self,system,agent):
        self.agent = agent
        self.system = system

        # proportion of the system img segment height that the
        # barline maybe shifted vertically to find the best fit
        self.kRange = .1
        self.widthFactor = 8 # neighbourhood is widthFactor times barlineWidth wide
        self.heightFactor = 1.2 # neighbourhood is heightFactor times systemHeight high

    @getter
    def getFeatureIdx(self):
        #w = self.agent.getLineWidth()
        h1,h2 = self.getBarHCoords()
        w = h2-h1
        print('w',w)
        h0 = h1-w
        h3 = h2+w
        staffTops = self.getVerticalStaffLinePositions()
        staffBots = staffTops+self.system.getStaffLineWidth()
        staffTops = nu.round(staffTops).astype(nu.int)
        staffBots = nu.round(staffBots).astype(nu.int)
        staffIdx = nu.array([x for x in 
                             chain.from_iterable([range(staffTops[i],staffBots[i]) 
                                                  for i in range(len(staffTops))])])
        interStaffIdx0 = nu.array([x for x in 
                                   chain.from_iterable([range(staffBots[i]+1,staffTops[i+1]-1) 
                                                        for i in range(4)])])
        interStaffIdx1 = nu.array([x for x in 
                                   chain.from_iterable([range(staffBots[i]+1,staffTops[i+1]-1) 
                                                        for i in range(5,9)])])
        interStaffIdx = nu.append(interStaffIdx0,interStaffIdx1)
        return {'h0':h0,'h1':h1,'h2':h2,'h3':h3,
                'staffTops':staffTops,'staffBots':staffBots,
                'staff':staffIdx,'interStaff':interStaffIdx}

    def checkInterStaffSymmetry(self,n=None):
        h0 = self.getFeatureIdx()['h0']
        h1 = self.getFeatureIdx()['h1']
        h2 = self.getFeatureIdx()['h2']
        h3 = self.getFeatureIdx()['h3']
        interStaff = self.getFeatureIdx()['interStaff']
        nh = self.getNeighbourhood()
        if n != None:
            ap = AgentPainter(nh)
            ml = nu.array([x for x in product(interStaff,nu.arange(h0,h1-1))])
            mr = nu.array([x for x in product(interStaff,nu.arange(h2+1,h3))])
            ap.paintRav(ml,color=(255,0,0),alpha=.4)
            ap.paintRav(mr,color=(255,0,0),alpha=.4)
            ap.writeImage(n)
        return nu.mean(nu.abs(nh.astype(nu.int)[interStaff,h0:h1-1]-nh[interStaff,h2+1:h3]))

    @getter
    def checkStaffSymmetry(self):
        h0 = self.getFeatureIdx()['h0']
        h1 = self.getFeatureIdx()['h1']
        h2 = self.getFeatureIdx()['h2']
        h3 = self.getFeatureIdx()['h3']
        staff = self.getFeatureIdx()['staff']
        nh = self.getNeighbourhood()
        print('nh',nh.shape,h0,h1,h2,h3)
        #return nu.mean(nh.astype(nu.int)[staff,h0:h1-1]-nh[staff,h2+1:h3])
        return nu.mean(nh.astype(nu.int)[staff,:h1-1]-nh[staff,h2+1:])
        
    def checkStaffLines(self):
        """Return the max staff line blackness on both sides of the bar,
        after subtracting the blackness above and below the staff lines.
        This yields low values for lines that are not on a staff (e.g. the accolade)
        """
        h0 = self.getFeatureIdx()['h0']
        h1 = self.getFeatureIdx()['h1']
        h2 = self.getFeatureIdx()['h2']
        h3 = self.getFeatureIdx()['h3']
        staffTops = self.getFeatureIdx()['staffTops']
        staffBots = self.getFeatureIdx()['staffBots']
        staffIdx = self.getFeatureIdx()['staff']
        w = int(nu.ceil(self.system.getStaffLineWidth()))
        aboveStaffIdx = nu.array([x for x in 
                                   chain.from_iterable([range(staffTops[i]-w,staffTops[i]) 
                                                        for i in range(len(staffTops))])])
        belowStaffIdx = nu.array([x for x in 
                                   chain.from_iterable([range(staffBots[i],staffBots[i]+w) 
                                                        for i in range(len(staffTops))])])
        interStaffIdx = self.getFeatureIdx()['interStaff']
        nh = self.getNeighbourhood()
        staffLeft = nu.mean(nh[staffIdx,h0:h1])
        staffRight = nu.mean(nh[staffIdx,h2:h3])

        aboveStaffLeft = nu.mean(nh[aboveStaffIdx,h0:h1])
        belowStaffLeft = nu.mean(nh[belowStaffIdx,h0:h1])
        staffLeft = nu.mean(nh[staffIdx,h0:h1])
        aboveStaffRight = nu.mean(nh[aboveStaffIdx,h2:h3])
        belowStaffRight = nu.mean(nh[belowStaffIdx,h2:h3])
        staffRight = nu.mean(nh[staffIdx,h2:h3])
        #print('sl',staffLeft,aboveStaffLeft,belowStaffLeft)
        #print('sr',staffRight,aboveStaffRight,belowStaffRight)
        #print('r',max(staffLeft-max(aboveStaffLeft,belowStaffLeft),
        #              staffRight-max(aboveStaffRight,belowStaffRight)))
        return max(staffLeft-max(aboveStaffLeft,belowStaffLeft),
                   staffRight-max(aboveStaffRight,belowStaffRight))

    @getter
    def getFeatures(self):
        sd = int(nu.round(self.system.getStaffLineDistance()))
        h0 = self.getFeatureIdx()['h0']
        h1 = self.getFeatureIdx()['h1']
        h2 = self.getFeatureIdx()['h2']
        h3 = self.getFeatureIdx()['h3']
        staffTops = self.getFeatureIdx()['staffTops']
        staffBots = self.getFeatureIdx()['staffBots']
        staffIdx = self.getFeatureIdx()['staff']
        interStaffIdx = self.getFeatureIdx()['interStaff']
        nh = self.getNeighbourhood()
        
        barStaff0 = nu.mean(nh[staffTops[0]:staffBots[4],h1:h2])
        barStaff1 = nu.mean(nh[staffTops[5]:staffBots[9],h1:h2])
        interStaff = nu.mean(nh[staffBots[4]:staffTops[5],h1:h2])
        aboveBar = nu.mean(nh[staffTops[0]-sd:staffTops[0],h1:h2])
        belowBar = nu.mean(nh[staffBots[9]:staffBots[9]+sd,h1:h2])
        if nu.isnan(aboveBar):
            aboveBar = 255.
        if nu.isnan(belowBar):
            belowBar = 255.
        staffLeft = nu.mean(nh[staffIdx,h0:h1])
        staffRight = nu.mean(nh[staffIdx,h2:h3])
        interStaffLeft = nu.mean(nh[interStaffIdx,h0:h1])
        interStaffRight = nu.mean(nh[interStaffIdx,h2:h3])
        # print('bar upper staff',barStaff0)
        # print('bar lower staff',barStaff1)
        # print('above bar',aboveBar)
        # print('below bar',belowBar)
        # print('inter staffline',interStaff)
        # print('mean staffline left from staffs',staffLeft)
        # print('mean staffline right from staffs',staffRight)
        # print('mean interstaffline left from staffs',interStaffLeft)
        # print('mean interstaffline right from staffs',interStaffRight)
        return (barStaff0,barStaff1,aboveBar,belowBar,staffLeft,staffRight,interStaffLeft,interStaffRight,interStaff)

    # todo:
    # * staff border detection: take larger margins around bar to look for absent stafflines at one side
    # * bars that are close: adapt margins to avoid interference?
    # * take apart wider bars: redo bar detection to separate double bars?
    # * increase impact of white in barStaff0 and barStaff1
    

    @getter
    def getEstimates(self):
        return nu.array((self.getBarEstimate(),self.getLeftMostBarEstimate(),self.getRightMostBarEstimate()))

    @getter
    def getBarEstimate(self):
        f = self.getFeatures()
        black = (f[0]+f[1]+f[5]+f[4])/(4.*255)
        white = (f[2]+f[3]+f[6]+f[7])/(4.*255)
        return black-white

    @getter
    def getLeftMostBarEstimate(self):
        f = self.getFeatures()
        white = f[4]/255.
        return 1-white

    @getter
    def getRightMostBarEstimate(self):
        f = self.getFeatures()
        white = f[5]/255.
        return 1-white
        
        
    @getter
    def getBarHCoords(self):
        M = self.getNeighbourhood().shape[1]
        #return nu.round(nu.array(((M-self.agent.getLineWidth())/2.,(M+self.agent.getLineWidth())/2.))).astype(nu.int)
        return nu.array((nu.ceil((M-self.agent.getLineWidth())/2.),nu.floor((M+self.agent.getLineWidth())/2.))).astype(nu.int)
        #b,e = (nu.array((-.5,.5))+self.widthFactor/2.0)*self.agent.getLineWidth()
        #print('be',b,e)
        #return nu.array((nu.floor(b),nu.ceil(e))).astype(nu.int)
        #return nu.array((self.h1,self.h2)).astype(nu.int)

    @getter
    def getVerticalStaffLinePositions(self):
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

    @getter
    def getNeighbourhood(self):

        # approximate system top and bottom (based on global staff detection)
        system0Top = self.system.getRotator().rotate(self.system.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.system.getRotator().rotate(self.system.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        sysHeight = (system1Bot-system0Top)
        #print(system0Top)
        #print(system1Bot)

        yTop = self.agent.getDrawMean()[1]+(system0Top-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.getAngle())
        yBot = self.agent.getDrawMean()[1]+(system1Bot-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.getAngle())
        middle = (nu.array((system0Top,yTop))+nu.array((system1Bot,yBot)))/2.0

        h1 = nu.round(middle[1]-.5*self.agent.getLineWidth())
        h2 = nu.round(middle[1]+.5*self.agent.getLineWidth())
        w = h2-h1
        whalf = int(nu.round(w*self.widthFactor/2.))
        hhalf = int(sysHeight*self.heightFactor/2.)
        
        r = Rotator(self.agent.getAngle()-.5,middle,nu.array((0,0.)))
        self.h1 = nu.round(r.rotate(nu.array([[middle[0],h1]]))[0,1]+whalf)
        self.h2 = nu.round(r.rotate(nu.array([[middle[0],h2]]))[0,1]+whalf)
        #mh1 = nu.round(r.derotate(nu.array([[0,-.5*self.agent.getLineWidth()]]))[0,1]+whalf)
        xx,yy = nu.mgrid[-hhalf:hhalf,-whalf:whalf]
        xxr,yyr = r.derotate(xx,yy)
        # check if derotated neighbourhood is inside image
        minx,maxx,miny,maxy = nu.min(xxr),nu.max(xxr),nu.min(yyr),nu.max(yyr)
        M,N = self.system.getCorrectedImgSegment().shape
        #print('w,h',middle)
        #print('mm',minx,maxx,M,miny,maxy,N)
        if minx < 0 or miny < 0 or maxx >= M or maxy >= N:
            return None

        cimg = getAntiAliasedImg(self.system.getCorrectedImgSegment(),xxr,yyr)
        h0 = max(0,h1-w)
        h3 = min(N,h2+w)
        vCorrection = getVCenterOfBarline(cimg,self.kRange)-hhalf
        # TODO: check if barneighbourhood is inside corrected system segment
        cimg = getAntiAliasedImg(self.system.getCorrectedImgSegment(),xxr+vCorrection,yyr)
        #print(cimg.shape)
        return cimg
    
    @getter
    def getNeighbourhoodNew(self):
        cimg = self._getNeighbourhood()
        hhalf = int(cimg.shape[1]/2.0)
        left,hCorrection,right = getHCenterAndWidth(cimg)-hhalf
        vCorrection = getVCenterOfBarline(cimg,self.kRange)-hhalf
        cimg = self._getNeighbourhood(nu.array((vCorrection,hCorrection)))
        
    def _getNeighbourhood(self,midCorrection=None):
        # approximate system top and bottom (based on global staff detection)
        system0Top = self.system.getRotator().rotate(self.system.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.system.getRotator().rotate(self.system.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        sysHeight = (system1Bot-system0Top)
        yTop = self.agent.getDrawMean()[1]+(system0Top-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.getAngle())
        yBot = self.agent.getDrawMean()[1]+(system1Bot-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.getAngle())
        middle = (nu.array((system0Top,yTop))+nu.array((system1Bot,yBot)))/2.0
        if midCorrection != None:
            middle += midCorrection
        r = Rotator(self.agent.getAngle()-.5,middle,nu.array((0,0.)))
        hhalf = int(sysHeight*self.heightFactor/2.)
        whalf = int(2*self.system.getStaffLineDistance())
        xx,yy = nu.mgrid[-hhalf:hhalf,-whalf:whalf]
        xxr,yyr = r.derotate(xx,yy)
        # check if derotated neighbourhood is inside image
        minx,maxx,miny,maxy = nu.min(xxr),nu.max(xxr),nu.min(yyr),nu.max(yyr)
        M,N = self.system.getCorrectedImgSegment().shape
        if minx < 0 or miny < 0 or maxx >= M or maxy >= N:
            return None
        cimg = getAntiAliasedImg(self.system.getCorrectedImgSegment(),xxr,yyr)
        return cimg

    def write(self):
        getVCenterOfBarline(self._getNeighbourhood(),self.kRange,True)

if __name__ == '__main__':
    pass
