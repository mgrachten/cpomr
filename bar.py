#!/usr/bin/env python

import sys
import numpy as nu
from itertools import chain,product
from utilities import getter,cachedProperty
from imageUtil import getAntiAliasedImg,findValleys,findPeaks,smooth
from utils import Rotator
from agent import AgentPainter

def getHCenterAndWidth(bimg):
    crossSection = nu.median(bimg,0)
    if nu.sum(crossSection) <= 0:
        return (None,None,None)
    avg = nu.average(nu.arange(len(crossSection)),weights=crossSection)
    center = int(nu.round(avg))
    total = nu.sum(crossSection)
    half = (total-crossSection[center])/2.0
    thr = .95
    for left in xrange(center,0,-1):
        if crossSection[left] == 0:
            left +=1
            break
        if nu.sum(crossSection[left:center])/half > thr:
            break
    for right in xrange(center+1,len(crossSection)):
        if nu.sum(crossSection[center+1:right])/half > thr or crossSection[right] == 0:
            break
    return nu.array((left,center,right))

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
    def __init__(self,i,j,barCandidate):
        self.bc = barCandidate
        self.i = i
        self.j = j
        #nu.savetxt('/tmp/bar-{0:03d}-{1:03d}.txt'.format(self.i,self.j),self.bc.diffSums)

class LeftBar(Bar): 
    pass

class RightBar(Bar):
    pass

class MiddleBar(Bar):
    pass

class DoubleBar(Bar):
    pass

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
        
    @cachedProperty
    def diffSums(self):
        return nu.sum(nu.diff(self.neighbourhood.astype(nu.float),axis=1),0)

    @cachedProperty
    def barInfo(self):
        peaks = findPeaks(self.diffSums)
        valleys = findValleys(self.diffSums)
        #mid = self.neighbourhood.shape[1]/2.0
        mid = self.diffSums.shape[0]/2.0
        lpeaks = peaks[peaks <= mid]
        rvalleys = valleys[valleys >= mid]
        hi = lpeaks[nu.argmax(self.diffSums[lpeaks])]
        lo = rvalleys[nu.argmin(self.diffSums[rvalleys])]
        npeaks = peaks[nu.logical_and(peaks>hi,peaks<lo)] 
        nvalleys = valleys[nu.logical_and(valleys>hi,valleys<lo)] 
        assert self.rotator != None
        leftMid,rightMid = self.rotator.derotate(nu.array([[0.0,hi],[0.0,lo+1]]))
        l,r = int(nu.round(leftMid[1])),int(round(rightMid[1]+1))
        print('surface ratio',nu.sum(self.vSums[hi:lo+1])/((lo+1-hi)*nu.max(self.vSums)))
        if len(nvalleys) == len(npeaks):
            if len(nvalleys) == 0:
                return self.BAR,leftMid,rightMid
            elif len(nvalleys) == 1:
                return self.DOUBLE_BAR,leftMid,rightMid
        return self.INVALID,leftMid,rightMid

    @cachedProperty
    def vSums(self):
        return nu.sum(self.approximateNeighbourhood.astype(nu.float),0)

    @cachedProperty
    def featureIdx(self):
        #w = self.agent.getLineWidth()
        h1,h2 = self.getBarHCoords()
        w = h2-h1
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
        nh = self.neighbourhood
        if n != None:
            ap = AgentPainter(nh)
            ml = nu.array([x for x in product(interStaff,nu.arange(h0,h1-1))])
            mr = nu.array([x for x in product(interStaff,nu.arange(h2+1,h3))])
            ap.paintRav(ml,color=(255,0,0),alpha=.4)
            ap.paintRav(mr,color=(255,0,0),alpha=.4)
            ap.writeImage(n)
        return nu.mean(nu.abs(nh.astype(nu.int)[interStaff,h0:h1-1]-nh[interStaff,h2+1:h3]))

    @cachedProperty
    def checkStaffSymmetry(self):
        h0 = self.getFeatureIdx()['h0']
        h1 = self.getFeatureIdx()['h1']
        h2 = self.getFeatureIdx()['h2']
        h3 = self.getFeatureIdx()['h3']
        staff = self.getFeatureIdx()['staff']
        nh = self.neighbourhood
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
        nh = self.neighbourhood
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

    @cachedProperty
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
        nh = self.neighbourhood
        
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
    

    @cachedProperty
    def getVerticalStaffLinePositions(self):
        # approximate system top and bottom (based on global staff detection)
        system0Top = self.system.rotator.rotate(self.system.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.system.rotator.rotate(self.system.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        sysHeight = (system1Bot-system0Top)
        hhalf = int(sysHeight*self.heightFactor/2.)
        sld = self.system.staffs[0].getStaffLineDistance()
        krng = range(-int(nu.ceil(.5*sld)),int(nu.ceil(.5*sld)))
        hsums = nu.sum(self.neighbourhood,1)
        hslw = int(nu.floor(.5*self.system.getStaffLineWidth()))

        inStaff = nu.arange(nu.round(self.system.getStaffLineWidth()))

        staff0Tops = hhalf-.5*sysHeight-hslw+nu.arange(5)*sld
        staff1Tops = hhalf+.5*sysHeight-hslw-nu.arange(4,-1,-1)*sld
        staff0Rows = nu.round(nu.array([s+inStaff for s in staff0Tops]).flat[:]).astype(nu.int)
        staff1Rows = nu.round(nu.array([s+inStaff for s in staff1Tops]).flat[:]).astype(nu.int)
        f0 = krng[nu.argmax([nu.sum(hsums[staff0Rows+k]) for k in krng])]
        f1 = krng[nu.argmax([nu.sum(hsums[staff1Rows+k]) for k in krng])]
        return nu.append(staff0Tops+f0,staff1Tops+f1)

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
                                       for system in self.system.scrImage.getSystems()] for x in y]),0)
        statistic = nu.array(self.leftRightAbsDiffSums)
        statistic[statistic != 0] /= lrmedian[statistic != 0]
        if statistic[0] < .5:
            if statistic[1] < .5:
                return None
            else:
                return LeftBar
        else:
            if statistic[1] < .5:
                return RightBar
            else:
                return MiddleBar
    
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
        w = .3*self.system.getStaffLineDistance()
        N = int(nu.floor(W/2.0))
        lr = int(nu.round(max(0,N-.5*self.agent.getLineWidth())))
        ll = int(nu.round(max(0,lr-w)))
        rl = int(nu.round(min(W,N+.5*self.agent.getLineWidth())))
        rr = int(nu.round(min(W,rl+w)))
        return self._leftRightAbsDiffSums((ll,lr,rl,rr))

    @cachedProperty
    def neighbourhood(self):
        print(self.estimatedType)
        return self.approximateNeighbourhood

    @cachedProperty
    def points(self):
        cimg = self.approximateNeighbourhood
        if cimg == None:
            return None, None
        vhalf = int(cimg.shape[0]/2.0)
        hhalf = int(cimg.shape[1]/2.0)
        left,hCenter,right = getHCenterAndWidth(cimg)
        if left == None:
            return None, None
        hCorrection = hCenter - hhalf
        w = right-left
        print('w,left,hCenter,right',w,left,hCenter,right,hCorrection,cimg.shape)
        h0 = max(0,left-hCorrection-w)
        h1 = left-hCorrection
        h2 = right-hCorrection
        h3 = min(cimg.shape[1],right-hCorrection+w)
        vCorrection = getVCenterOfBarline(cimg[:,h0:h3],self.kRange)-vhalf
        return nu.array((vCorrection,hCorrection)),nu.array((h0,h1,h2,h3))

    @cachedProperty
    def approximateNeighbourhood(self):
        return self._getNeighbourhood()

    @cachedProperty
    def rotator(self):
        return self.rotator
        
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
        #r = Rotator(self.agent.angle-.5,middle,nu.array((0,0.)))
        self.rotator = Rotator(self.agent.angle-.5,middle,nu.array((0,0.)))
        hhalf = int(sysHeight*self.heightFactor/2.)
        whalf = int(2*self.system.getStaffLineDistance())
        xx,yy = nu.mgrid[-hhalf:hhalf,-whalf:whalf]
        xxr,yyr = self.rotator.derotate(xx,yy)
        # check if derotated neighbourhood is inside image
        minx,maxx,miny,maxy = nu.min(xxr),nu.max(xxr),nu.min(yyr),nu.max(yyr)
        M,N = self.system.correctedImgSegment.shape
        if minx < 0 or miny < 0 or maxx >= M or maxy >= N:
            return None
        cimg = getAntiAliasedImg(self.system.correctedImgSegment,xxr,yyr)
        return cimg

    def write(self):
        getVCenterOfBarline(self.approximateNeighbourhood,self.kRange,True)

    # @cachedProperty
    # def getNeighbourhoodNew(self):
    #     """obsolete"""
    #     center,hPoints = self.points
    #     if center == None:
    #         return None
    #     print('c, hp',center,hPoints)
    #     cimg = self._getNeighbourhood(center)
    #     if cimg == None:
    #         return None
    #     print('cimg1',cimg.shape)
    #     return cimg

    # @cachedProperty
    # def getNeighbourhoodOld(self):
    #     # approximate system top and bottom (based on global staff detection)
    #     system0Top = self.system.rotator.rotate(self.system.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
    #     system1Bot = self.system.rotator.rotate(self.system.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
    #     sysHeight = (system1Bot-system0Top)

    #     yTop = self.agent.getDrawMean()[1]+(system0Top-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.angle)
    #     yBot = self.agent.getDrawMean()[1]+(system1Bot-self.agent.getDrawMean()[0])*nu.cos(nu.pi*self.agent.angle)
    #     middle = (nu.array((system0Top,yTop))+nu.array((system1Bot,yBot)))/2.0

    #     h1 = nu.round(middle[1]-.5*self.agent.getLineWidth())
    #     h2 = nu.round(middle[1]+.5*self.agent.getLineWidth())
    #     w = h2-h1
    #     whalf = int(nu.round(w*self.widthFactor/2.))
    #     hhalf = int(sysHeight*self.heightFactor/2.)
        
    #     r = Rotator(self.agent.angle-.5,middle,nu.array((0,0.)))
    #     self.h1 = nu.round(r.rotate(nu.array([[middle[0],h1]]))[0,1]+whalf)
    #     self.h2 = nu.round(r.rotate(nu.array([[middle[0],h2]]))[0,1]+whalf)
    #     #mh1 = nu.round(r.derotate(nu.array([[0,-.5*self.agent.getLineWidth()]]))[0,1]+whalf)
    #     xx,yy = nu.mgrid[-hhalf:hhalf,-whalf:whalf]
    #     xxr,yyr = r.derotate(xx,yy)
    #     # check if derotated neighbourhood is inside image
    #     minx,maxx,miny,maxy = nu.min(xxr),nu.max(xxr),nu.min(yyr),nu.max(yyr)
    #     M,N = self.system.correctedImgSegment.shape
    #     #print('w,h',middle)
    #     #print('mm',minx,maxx,M,miny,maxy,N)
    #     if minx < 0 or miny < 0 or maxx >= M or maxy >= N:
    #         return None

    #     cimg = getAntiAliasedImg(self.system.correctedImgSegment,xxr,yyr)
    #     h0 = max(0,h1-w)
    #     h3 = min(N,h2+w)
    #     vCorrection = getVCenterOfBarline(cimg,self.kRange)-hhalf
    #     # TODO: check if barneighbourhood is inside corrected system segment
    #     cimg = getAntiAliasedImg(self.system.correctedImgSegment,xxr+vCorrection,yyr)
    #     #print(cimg.shape)
    #     return cimg

    # @cachedProperty
    # def getBarHCoordsOld(self):
    #     M = self.getNeighbourhood().shape[1]
    #     #return nu.round(nu.array(((M-self.agent.getLineWidth())/2.,(M+self.agent.getLineWidth())/2.))).astype(nu.int)
    #     return nu.array((nu.ceil((M-self.agent.getLineWidth())/2.),nu.floor((M+self.agent.getLineWidth())/2.))).astype(nu.int)


    # @cachedProperty
    # def getEstimates(self):
    #     return nu.array((self.getBarEstimate(),self.getLeftMostBarEstimate(),self.getRightMostBarEstimate()))

    # @cachedProperty
    # def getBarEstimate(self):
    #     f = self.getFeatures()
    #     black = (f[0]+f[1]+f[5]+f[4])/(4.*255)
    #     white = (f[2]+f[3]+f[6]+f[7])/(4.*255)
    #     return black-white

    # @cachedProperty
    # def getLeftMostBarEstimate(self):
    #     f = self.getFeatures()
    #     white = f[4]/255.
    #     return 1-white

    # @cachedProperty
    # def getRightMostBarEstimate(self):
    #     f = self.getFeatures()
    #     white = f[5]/255.
    #     return 1-white

    
if __name__ == '__main__':
    pass
