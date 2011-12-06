#!/usr/bin/env python

import sys,os
import numpy as nu
from utils import Rotator
from utilities import getter
from agent import makeAgentClass, AgentPainter, assignToAgents, mergeAgents
from utils import selectColumns
from imageUtil import getAntiAliasedImg
from bar import Bar

def sortBarAgents(agents):
    agents.sort(key=lambda x: -x.score)
    scores = nu.append(nu.array([x.score for x in agents if x.score > 1]),0)
    hyp = 0
    if len(scores) > 1:
        hyp = nu.argmin(nu.diff(scores))
        #print(scores)
        print('guessing:',hyp+1)
    return hyp+1

class System(object):
    def __init__(self,scoreImage,staffs,n=0):
        self.scrImage = scoreImage
        self.staffs = staffs
        self.barPoints = []
        self.dodraw = False
        # system counter
        self.n = n

    def getTop(self):
        return self.staffs[0].top
    def getBottom(self):
        return self.staffs[1].bottom

    def addBarPoint(self,xy):
        self.barPoints.append(xy)

    def getLowerLeft(self):
        return self.getSystemPoints()[2]

    def getUpperLeft(self):
        return self.getSystemPoints()[0]

    def getLowerMid(self):
        return self.getSystemPoints()[4]

    def getLowerMidLocal(self):
        return nu.array((self.getSystemHeight()-1,int((self.getSystemWidth()-1)/2)))

    @getter
    def getSystemPoints(self):
        # returns topleft, topright, botleft, botright, and lower hmid
        # of tilted rectangle, such that all above coordinates fall inside the img
        hMid = int(self.scrImage.getWidth()/2.)
        # sometimes the staff is close to the border of the segment
        extra = nu.array((int(nu.ceil(self.getStaffLineDistance())),0))

        top = self.staffs[0].top-extra[0]
        bot = self.staffs[1].bottom+extra[0]

        dyl = -hMid
        dyr = self.scrImage.getWidth()-hMid

        botLeft = bot+dyl*nu.tan(nu.pi*self.getStaffAngle())
        botRight = bot+dyr*nu.tan(nu.pi*self.getStaffAngle())
        topLeft = top+dyl*nu.tan(nu.pi*self.getStaffAngle())
        topRight = top+dyr*nu.tan(nu.pi*self.getStaffAngle())

        botCorrection = min(0,nu.floor(self.scrImage.getHeight()-max(botLeft,botRight)-1))
        topCorrection = min(0,nu.floor(min(topLeft,topRight))-1)
        return (nu.array((topLeft-topCorrection,0)),
             nu.array((topRight-topCorrection,self.scrImage.getWidth())),
             nu.array((botLeft+botCorrection,0)),
             nu.array((botRight+botCorrection,self.scrImage.getWidth())),
             nu.array((bot+botCorrection,hMid)))


    def draw(self):
        for staff in self.staffs:
            staff.draw()
        #self.scrImage.ap.drawText('s{0:02d}'.format(self.n),
        #                          self.getUpperLeft(),size=30,color=(255,0,0),alpha=.8)
        self.drawBarPoints()

    def drawBarPoints(self):
        lower = int(self.getLowerLeft()[0])
        upper = int(self.getUpperLeft()[0])
        r = lower-upper
        c = nu.array((255,0,0))
        for p in self.barPoints:
            self.scrImage.ap.paintRav(nu.column_stack((nu.arange(upper,lower),(nu.zeros(r)+p[0]).astype(nu.int))),
                                      c)
        
    def getStaffAngle(self):
        return nu.mean([s.getAngle() for s in self.staffs])

    @getter
    def getHSums(self):
        return nu.sum(self.getCorrectedImgSegment(),1)

    @getter
    def getStaffLineWidth(self):
        return nu.mean([a.getLineWidth() for a in self.staffs[0].staffLineAgents]+
                       [a.getLineWidth() for a in self.staffs[1].staffLineAgents])

    @getter
    def getStaffLineDistance(self):
        return (self.staffs[0].getStaffLineDistance()+self.staffs[1].getStaffLineDistance())/2.0

    def getImgHParts(self,hbins,overlap):
        M,N = self.getCorrectedImgSegment().shape
        overlapPix = int(overlap*N/2.)
        breaks = nu.linspace(0,N,hbins+1)
        lefts = breaks[:-1].copy()
        lefts[1:] -= overlapPix
        rights = breaks[1:].copy()
        rights[:-1] += overlapPix
        return [self.getCorrectedImgSegment()[:,lefts[i]:rights[i]]
                for i in range(len(lefts))],lefts,rights
            
    @getter
    def getBarLines(self):
        """
        strategy:
        * run barline detection for different (slightly overlapping) segements
        * join agents (+merge)
        """
        vbins = 5
        hbins = 3
        overlap = .05 # of width
        hparts,lefts,rights = self.getImgHParts(hbins,overlap)
        agents = []
        for i,hpart in enumerate(hparts):
            agents.extend(self.getBarLinesPart(hpart,vbins,lefts[i],rights[i],i))
        agents,died = mergeAgents(agents)
        agents.sort(key=lambda x: -x.score)
        for a in agents:
            print(a)
        agents.sort(key=lambda x: x.getDrawMean()[1])

        return agents
        
    def getBarLinesPart(self,img,vbins,yoffset,rightBorder,j):
        BarAgent = makeAgentClass(targetAngle=.5,
                                  maxAngleDev=4/180.,
                                  maxError=self.getStaffLineWidth()/7.0,
                                  minScore=-2,
                                  yoffset=yoffset)
        agents = []
        systemTopL = self.getRotator().rotate(self.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        systemBotL = self.getRotator().rotate(self.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        hsums = nu.sum(img,1)[int(systemTopL):int(systemBotL)]
        rows = selectColumns(hsums,vbins)[0]+int(systemTopL) # sounds funny, change name of function       

        ap = AgentPainter(self.getCorrectedImgSegment())
        ap.paintVLine(yoffset,step=4,color=(50,150,50))
        ap.paintVLine(rightBorder,step=4,color=(50,150,50))
        #draw = self.dodraw
        draw = False
        K = int(.1*len(rows))
        for i,r in enumerate(rows[:K]):
            died = []
            agentsnew,d = assignToAgents(img[r,:],agents,BarAgent,
                                         self.getCorrectedImgSegment().shape[1],
                                         vert=r,fixAgents=False)
            died.extend(d)

            if len(agents) > 2:
                agentsnew,d = mergeAgents(agentsnew)
                died.extend(d)
            agents = agentsnew
            draw = False#self.n==0 and j==4
            if draw:
                ap.reset()
                ap.paintHLine(r,step=2,color=(50,50,50))
                for a in died:
                    ap.unregister(a)
                for a in agents:
                    ap.register(a)
                    ap.drawAgent(a,-400,400)
                ap.writeImage('system{0:04d}-part{1:04d}-{2:04d}-r{3:04d}.png'.format(self.n,j,i,r))
            if len(agents) > 1:
                agents.sort(key=lambda x: -x.score)
        return [a for a in agents if a.score > 1 and a.age > .1*K]

    def getNonTerminatingBarCandidates(self):
        bs = nu.array([b.checkStaffSymmetry() for b in self.getBarCandidates()])
        sidx = nu.argsort(bs)
        if len(bs) > 2:
            return list(nu.array(self.getBarCandidates())[sidx[1:-1]])
        else:
            return []
        
    def selectOpeningClosingBars(self,bars):
        # obsolete
        bs = nu.array([b.checkStaffSymmetry() for b in bars])
        opener = None
        closer = None
        if len(bs) < 4:
            return (0,1)
        sidx = nu.argsort(bs)
        m = nu.mean(bs[sidx[1:-1]])
        std = nu.std(bs[sidx[1:-1]])
        if bs[sidx[0]]-m < 4*std:
            opener = sidx[0]
        if bs[sidx[-1]]-m > 4*std:
            closer = sidx[-1]
        return (opener,closer)

    @getter
    def getBars(self):
        bars = [Bar(self,x) for x in self.getBarLines()]
        print('bars',len(bars))
        for b in bars:
            pass #print(b.getNeighbourhood())
        bars = [x for x in bars if x.getNeighbourhood() != None and x.checkStaffLines() > 50]
        #and x.checkStaffLines() > 50]# and x.checkInterStaffSymmetry()<200 ]
        #for i,b in enumerate(bars):
        #    print(i,b.checkInterStaffSymmetry('s{0:03d}b{1:03d}.png'.format(self.n,i)),b.checkStaffSymmetry())
        print('bars nonempty neighbourhood',len(bars))
        return bars

    @getter
    def getBarCandidates(self):
        bars = [Bar(self,x) for x in self.getBarLines()]
        return [x for x in bars if x.getNeighbourhoodNew() != None]

    def getSystemWidth(self):
        # this gets cut off from the width, to fit in the page rotated
        cutOff = nu.abs(self.getSystemHeight()*nu.tan(nu.pi*self.getStaffAngle()))
        systemWidth = self.scrImage.getWidth()/nu.cos(nu.pi*self.getStaffAngle()) - 2*cutOff
        systemWidth = int((nu.floor(systemWidth/2.0)-1)*2+1)
        return systemWidth

    def getSystemHeight(self):
        return self.getLowerLeft()[0]-self.getUpperLeft()[0]
        
    @getter
    def getRotator(self):
        return Rotator(self.getStaffAngle(),self.getLowerMid(),self.getLowerMidLocal())

    @getter
    def getCorrectedImgSegment(self):
        halfSystemWidth = int((self.getSystemWidth()-1)/2)
        r = self.getRotator()
        #xx,yy = nu.mgrid[0:self.getSystemHeight(),-halfSystemWidth:halfSystemWidth]
        xx,yy = nu.mgrid[0:self.getSystemHeight(),-halfSystemWidth:halfSystemWidth]
        yy += self.getLowerMidLocal()[1]
        xxr,yyr = r.derotate(xx,yy)
        return getAntiAliasedImg(self.scrImage.getImg(),xxr,yyr)

if __name__ == '__main__':
    pass

