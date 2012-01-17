#!/usr/bin/env python

import sys,os
import numpy as nu
from utils import Rotator
from utilities import cachedProperty, getter
from agent import makeAgentClass, AgentPainter, assignToAgents, mergeAgents
from utils import selectColumns
from imageUtil import getAntiAliasedImg, smooth
from bar import BarCandidate

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
        return nu.mean([s.angle for s in self.staffs])

    @cachedProperty
    def hSums(self):
        return nu.sum(self.correctedImgSegment,1)

    @getter
    def getStaffLineWidth(self):
        return nu.mean([a.getLineWidth() for a in self.staffs[0].staffLineAgents]+
                       [a.getLineWidth() for a in self.staffs[1].staffLineAgents])

    @getter
    def getStaffLineDistance(self):
        return (self.staffs[0].getStaffLineDistance()+self.staffs[1].getStaffLineDistance())/2.0

    def getImgHParts(self,hbins,overlap):
        M,N = self.correctedImgSegment.shape
        overlapPix = int(overlap*N/2.)
        breaks = nu.linspace(0,N,hbins+1)
        lefts = breaks[:-1].copy()
        lefts[1:] -= overlapPix
        rights = breaks[1:].copy()
        rights[:-1] += overlapPix
        return [self.correctedImgSegment[:,lefts[i]:rights[i]]
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
        systemTopL = self.rotator.rotate(self.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        systemBotL = self.rotator.rotate(self.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        hsums = nu.sum(img,1)[int(systemTopL):int(systemBotL)]
        rows = selectColumns(hsums,vbins)[0]+int(systemTopL) # sounds funny, change name of function       

        ap = AgentPainter(self.correctedImgSegment)
        ap.paintVLine(yoffset,step=4,color=(50,150,50))
        ap.paintVLine(rightBorder,step=4,color=(50,150,50))
        #draw = self.dodraw
        draw = False
        K = int(.1*len(rows))
        for i,r in enumerate(rows[:K]):
            died = []
            agentsnew,d = assignToAgents(img[r,:],agents,BarAgent,
                                         self.correctedImgSegment.shape[1],
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

    #@cachedProperty
    def getBars(self):
        # get barcandidates (excluding those without a valid neighbourhood)
        barCandidates = [bc for bc in self.barCandidates if bc.estimatedType != None]
        #info = [bc.barInfo for bc in barCandidates]
        # estimated bar type (bar, double bar, invalid)
        #btypes = [x[0] for x in info]
        print('barCandidates')
        for i,b in enumerate(barCandidates):
            #print('type: ({0}{1}), lr: ({2:02f},{3:02f})'.format('x','x',b[0].barInfo[1][1],b[-1].barInfo[2][1]))
            #l,r = b.rotator.rotate(nu.array((b.barInfo[1],b.barInfo[2])))
            barHCoords = b.rotator.rotate(b.barInfo)[:,1]
            print('system/bar',self.n,i)
            print('barinfo',b.estimatedType)
            print('vcorr',b.vCorrection)
            
            ap1 = AgentPainter(b.neighbourhood)
            for bhc in barHCoords:
                ap1.paintVLine(nu.round(bhc),step=2,color=(255,0,0))
                #ap1.paintVLine(nu.round(r[1]),step=2,color=(255,0,0))
            ap1.writeImage('bar-{0:03d}-{1:03d}.png'.format(self.n,i))
            nu.savetxt('/tmp/bar-{0:03d}-{1:03d}.txt'.format(self.n,i),
                       nu.column_stack((b.diffSums,b.weights*b.diffSums,
                                        b.curve,
                                        nu.sum(b.approximateNeighbourhood.astype(nu.float),0)[:-1])))

        if True:
            return None
        #vidx = nu.array([x != BarCandidate.INVALID for x in btypes])
        vidx = nu.array([True for x in range(len(barcandidates))])
        idx = nu.arange(len(info))
        valid = idx[vidx]

        # detect if left or right end of consecutive candidates coincide
        leftMid = nu.array([info[x][1] for x in valid])
        rightMid = nu.array([info[x][2] for x in valid])
        leftDiff = nu.diff(leftMid,axis=0)
        rightDiff = nu.diff(rightMid,axis=0)
        lDist = nu.sum(leftDiff**2,1)**.5
        rDist = nu.sum(rightDiff**2,1)**.5
        
        lw = nu.mean([bc.agent.getLineWidth() for bc in barCandidates])
        lwFactor = 2
        linked = nu.logical_or(lDist < lwFactor*lw,rDist < lwFactor*lw)
        # aggregate groups of barcandidates that belong together
        bcs = []
        if len(valid) == 1:
            bcs.append((barCandidates[valid[0]],))
        elif len(valid) >= 1:
            i = 0
            #for i,j in enumerate(valid[:-1]):
            while i < len(valid)-1:
                j = valid[i]
                if linked[i]:
                    l = info[j][1]
                    r = info[valid[i+1]][2]
                    double = info[j][0] == BarCandidate.DOUBLE_BAR or \
                        info[valid[i+1]][0] == BarCandidate.DOUBLE_BAR
                    bcs.append((barCandidates[j],barCandidates[valid[i+1]]))
                    i += 2
                else:
                    l,r = info[j][1:]
                    double = info[j][0] == BarCandidate.DOUBLE_BAR
                    bcs.append((barCandidates[j],))
                    i += 1
            print(len(valid),valid,len(barCandidates),i)
            if i == len(valid)-1:
                print('jo',valid[i])
                bcs.append((barCandidates[valid[i]],))
                
        for i,b in enumerate([]): #enumerate(bcs):
            #print('type: ({0}{1}), lr: ({2:02f},{3:02f})'.format('x','x',b[0].barInfo[1][1],b[-1].barInfo[2][1]))
            l,r = b[0].rotator.rotate(nu.array((b[0].barInfo[1],b[-1].barInfo[2])))
            print('system/bar',self.n,i)
            for bc in b:
                print('barinfo',bc.barInfo,bc.estimatedType)
            ap1 = AgentPainter(b[0].neighbourhood)
            ap1.paintVLine(nu.round(l[1]),step=2,color=(255,0,0))
            ap1.paintVLine(nu.round(r[1]),step=2,color=(255,0,0))
            ap1.writeImage('bar-{0:03d}-{1:03d}.png'.format(self.n,i))
            nu.savetxt('/tmp/bar-{0:03d}-{1:03d}.txt'.format(self.n,i),b[0].diffSums)
        

    @cachedProperty
    def barCandidates(self):
        bars = [BarCandidate(self,x) for x in self.getBarLines()]
        return bars#[x for x in bars if x.neighbourhood != None]

    def getSystemWidth(self):
        # this gets cut off from the width, to fit in the page rotated
        cutOff = nu.abs(self.getSystemHeight()*nu.tan(nu.pi*self.getStaffAngle()))
        systemWidth = self.scrImage.getWidth()/nu.cos(nu.pi*self.getStaffAngle()) - 2*cutOff
        systemWidth = int((nu.floor(systemWidth/2.0)-1)*2+1)
        return systemWidth

    def getSystemHeight(self):
        return self.getLowerLeft()[0]-self.getUpperLeft()[0]
        
    @cachedProperty
    def rotator(self):
        return Rotator(self.getStaffAngle(),self.getLowerMid(),self.getLowerMidLocal())

    @cachedProperty
    def correctedImgSegment(self):
        halfSystemWidth = int((self.getSystemWidth()-1)/2)
        #xx,yy = nu.mgrid[0:self.getSystemHeight(),-halfSystemWidth:halfSystemWidth]
        xx,yy = nu.mgrid[0:self.getSystemHeight(),-halfSystemWidth:halfSystemWidth]
        yy += self.getLowerMidLocal()[1]
        xxr,yyr = self.rotator.derotate(xx,yy)
        return getAntiAliasedImg(self.scrImage.getImg(),xxr,yyr)

if __name__ == '__main__':
    pass

