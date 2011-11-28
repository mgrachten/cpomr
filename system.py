#!/usr/bin/env python

import sys
import numpy as nu
from utils import Rotator
from utilities import getter
from agent import makeAgentClass, AgentPainter, assignToAgents, mergeAgents
from utils import selectColumns

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
    def __init__(self,scoreImage,staffs):
        self.scrImage = scoreImage
        self.staffs = staffs
        self.barPoints = []
        # self.Qg, the point in the middle of the lower border of the system 
        # in the coordinate system of the original picture (global)
        # this is the point around which the rotation is done
        #self.Qg = nu.array((self.getBottom()-1,int(self.scrImage.getWidth()/2.)))
        
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
        top = self.staffs[0].top
        bot = self.staffs[1].bottom

        dyl = -hMid
        dyr = self.scrImage.getWidth()-hMid

        botLeft = bot+dyl*nu.tan(nu.pi*self.getStaffAngle())
        botRight = bot+dyr*nu.tan(nu.pi*self.getStaffAngle())
        topLeft = top+dyl*nu.tan(nu.pi*self.getStaffAngle())
        topRight = top+dyr*nu.tan(nu.pi*self.getStaffAngle())

        botCorrection = min(0,nu.floor(self.scrImage.getHeight()-max(botLeft,botRight)-1))
        topCorrection = min(0,nu.floor(min(topLeft,topRight))-1)
        print('tcor bcor',topCorrection,botCorrection)
        r = (nu.array((topLeft-topCorrection,0)),
             nu.array((topRight-topCorrection,self.scrImage.getWidth())),
             nu.array((botLeft+botCorrection,0)),
             nu.array((botRight+botCorrection,self.scrImage.getWidth())),
             nu.array((bot+botCorrection,hMid)))
        print('tl,tr,bl,br,bm')
        for p in r:
            print(p)
        return r

    def draw(self):
        for staff in self.staffs:
            staff.draw()
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

    def getBarLines(self):
        agents = []
        defBarAngle = .5 #(self.getStaffAngle()+.5)%1
        print('default staff angle for this system',self.getStaffAngle())
        #print('default bar angle for this system',defBarAngle)
        #assert defBarAngle >= 0
        BarAgent = makeAgentClass(targetAngle=defBarAngle,
                                  maxAngleDev=4/180.,
                                  maxError=3,
                                  minScore=-5,
                                  offset=0)
        maxWidth = nu.inf#3*self.getStaffLineWidth()
        systemTopL = self.getRotator().rotate(self.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        systemBotL = self.getRotator().rotate(self.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        bins = 9
        rows = [p for p in selectColumns(self.getHSums(),bins)[0] if systemTopL <= p <= systemBotL] # sounds funny, change name of function
        
        finalStage = False
        k = 0
        ap = AgentPainter(self.getCorrectedImgSegment())
        draw = True
        draw = False
        for i,r in enumerate(rows[:int(.1*len(rows))]):
            died = []
            agentsnew,d = assignToAgents(self.getCorrectedImgSegment()[r,:],agents,BarAgent,
                                            self.getCorrectedImgSegment().shape[1],vert=r,fixAgents=finalStage)
            died.extend(d)

            if len(agents) > 2:
                agentsnew,d = mergeAgents(agentsnew)
                died.extend(d)
            agents = agentsnew
            #assert len(set(agents).intersection(set(died))) == 0
            print('row',i)
            if len(agents) > 1:
                agents.sort(key=lambda x: -x.score)
                #k = sortBarAgents(agents)
            if draw:
                ap.reset()
                ap.paintHLine(r)
                for a in died:
                    #ap.drawAgent(a,-300,300)
                    ap.unregister(a)
                for j,a in enumerate(agents):
                    print('{0} {1}'.format(j,a))
                    ap.register(a)
                    ap.drawAgent(a,-300,300)
                f0,ext = os.path.splitext(fn)
                print(f0,ext)
                ap.writeImage(f0+'-{0:04d}-r{1}'.format(i,r)+'.png')
        k = sortBarAgents(agents)
        bAgents = agents[:k]
        meanScore = nu.mean([a.score for a in bAgents])
        meanAge = nu.mean([a.age for a in bAgents])
        for j,a in enumerate(agents):
            print('{0} {1}'.format(j,a))
        agents = [a for a in agents if a.score > .4*meanScore and a.age > .4*meanAge]
        print('chose {0} agents'.format(len(agents)))
        draw = False
        if draw:
            ap.reset()
            for a in agents:
                print(a)
                ap.register(a)
                ap.drawAgentGood(a,-300,300)
            f0,ext = os.path.splitext(fn)
            print(f0,ext)
            ap.writeImage(f0+'-sys{0:04d}.png'.format(int(self.getLowerLeft()[0])))
        agents.sort(key=lambda x: x.getDrawMean()[1])
        for j,a in enumerate(agents):
            self.barlineTest(a,self.getTop(),j)
            #self.assessBarLine(a)
        #agents = self.selectBarLines(agents)
        return agents

    def getStaffLinesAroundBar(self,cimg,w,sysHeight,staffLineWidth,staffDistance):
        
        #template = nu.array((-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1))
        #stafftemp = [-1,1,-1,1,-1,1,-1,1,-1,1,-1]
        d = .5
        stafftemp = (int(staffLineWidth)*[1]+int(staffDistance-staffLineWidth)*[-1])*4+int(staffLineWidth)*[1]
        w1 = [-d]*int(sysHeight-8*staffDistance)
        margin = [-d]*int((cimg.shape[0]-sysHeight)/2.)
        #template = nu.array([-.1]*2+stafftemp+[-.1]*2*len(stafftemp)+stafftemp+[-.1]*2)
        template = nu.array(margin+stafftemp+w1+stafftemp+margin)
        stafflineIdx = nu.where(template==1)[0]
        hsum = ((nu.mean(cimg[:,0:int(.25*w)],1)+nu.mean(cimg[:,int(.75*w):],1))/2.0-127.0)/127.0
        l1 = float(len(template))
        l2 = float(len(hsum))
        print(l1,l2)
        def c(x,y,i,j):
            return (x[i]-y[j])**2+10*nu.abs(i/l1-j/l2)**2-.9*max(0,y[j])*(x[i]-y[j])**2
            #return (x[i]-y[j])**2+10*nu.abs(i/l1-j/l2)
            #return -x[i]*y[j]+(float(i)/len(x)-float(j)/len(y))
        apath = nu.column_stack((nu.linspace(0,l1-1,int(max(l1,l2))),nu.linspace(0,l2-1,int(max(l1,l2))))).astype(nu.int)
        path,tcost = dtw(template,hsum,K=int(.2*l1),L=int(.2*l2),cost=c,returnCost=True,apath=apath)
        result = []
        for i in stafflineIdx:
            x = path[path[:,0] == i,1]
            result.append((x[0],x[-1]))
        return result
        
    def barlineTest(self,agent,s,i):
        system0Top = self.getRotator().rotate(self.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system0Bot = self.getRotator().rotate(self.staffs[0].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        system1Top = self.getRotator().rotate(self.staffs[1].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.getRotator().rotate(self.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        w = agent.getLineWidth()*4
        whalf = int(w/2.)
        sysHeight = (system1Bot-system0Top)
        h = sysHeight*1.2
        hhalf = int(h/2.)
        yTop = agent.getDrawMean()[1]+(system0Top-agent.getDrawMean()[0])*nu.cos(nu.pi*agent.getAngle())
        yBot = agent.getDrawMean()[1]+(system1Bot-agent.getDrawMean()[0])*nu.cos(nu.pi*agent.getAngle())
        print(agent)
        middle = (nu.array((system0Top,yTop))+nu.array((system1Bot,yBot)))/2.0
        r = Rotator(agent.getAngle()-.5,middle,nu.array((0,0.)))
        xx,yy = nu.mgrid[-hhalf:hhalf,-whalf:whalf]
        print(middle,whalf,hhalf)
        xxr,yyr = r.derotate(xx,yy)
        minx,maxx,miny,maxy = nu.min(xxr),nu.max(xxr),nu.min(yyr),nu.max(yyr)
        M,N = self.getCorrectedImgSegment().shape

        if minx < 0 or miny < 0 or maxx >= M or maxy >= N:
            return False

        cimg = getAntiAliasedImg(self.getCorrectedImgSegment(),xxr,yyr)
        kRange = .1
        vCorrection = self.findVCenterOfBarline(cimg,kRange)-hhalf
        cimg = getAntiAliasedImg(self.getCorrectedImgSegment(),xxr+vCorrection,yyr)

        ap = AgentPainter(cimg)
        ap.paintVLine(int(1.5*agent.getLineWidth()),step=2,color=(255,0,0))
        ap.paintVLine(int(2.5*agent.getLineWidth()),step=2,color=(255,0,0))
        #print('lw',self.staffs[0].staffLineAgents[0].getLineWidth())
        #hhll = self.getStaffLinesAroundBar(cimg,w,sysHeight,self.getStaffLineWidth(),self.getStaffLineDistance())

        sld = self.staffs[0].getStaffLineDistance()
        color = (200,0,200)
        krng = range(-int(nu.ceil(.5*sld)),int(nu.ceil(.5*sld)))
        ksums = []
        #slw = self.staffs[0].staffLineAgents[j].getLineWidth()
        slw = self.getStaffLineWidth()
        for k in krng:
            ksum = 0
            for j in range(5):
                hi = int(hhalf+1-(system1Bot-system0Top)/2.0+j*sld-nu.ceil(.5*slw))
                lo = int(hhalf+1-(system1Bot-system0Top)/2.0+j*sld+nu.ceil(.5*slw))
                ksum += nu.sum(cimg[hi+k:lo+k,:])
            ksums.append(ksum)
        f = krng[nu.argmax(ksums)]
        for j in range(5):
            #slw = self.staffs[0].staffLineAgents[j].getLineWidth()
            hi = int(hhalf+1-(system1Bot-system0Top)/2.0+j*sld-nu.ceil(.5*slw))+f
            lo = int(hhalf+1-(system1Bot-system0Top)/2.0+j*sld+nu.ceil(.5*slw))+f
            ap.paintHLine(hi,step=2,alpha=.8,color=color)
            ap.paintHLine(lo,step=2,alpha=.8,color=color)

        sld = self.staffs[1].getStaffLineDistance()
        krng = range(-int(nu.ceil(.5*sld)),int(nu.ceil(.5*sld)))
        ksums = []
        for k in krng:
            ksum = 0
            for j in range(5):
                #slw = self.staffs[1].staffLineAgents[j].getLineWidth()
                hi = int(hhalf+1+(system1Bot-system0Top)/2.0-(4-j)*sld-nu.ceil(.5*slw))
                lo = int(hhalf+1+(system1Bot-system0Top)/2.0-(4-j)*sld+nu.ceil(.5*slw))
                ksum += nu.sum(cimg[hi+k:lo+k,:])
            ksums.append(ksum)
        f = krng[nu.argmax(ksums)]

        for j in range(5):
            #slw = self.staffs[1].staffLineAgents[j].getLineWidth()
            hi = int(hhalf+1+(system1Bot-system0Top)/2.0-(4-j)*sld-nu.ceil(.5*slw))+f
            lo = int(hhalf+1+(system1Bot-system0Top)/2.0-(4-j)*sld+nu.ceil(.5*slw))+f
            ap.paintHLine(hi,step=2,alpha=.8,color=color)
            ap.paintHLine(lo,step=2,alpha=.8,color=color)

        ap.writeImage('bar-{0:04d}-{1:03d}.png'.format(s,i))

    def findVCenterOfBarline(self,bimg,kRange):
        N,M = bimg.shape
        kMin = int(N*(1-kRange)/2.0)
        kMax = int(N*(1+kRange)/2.0)
        scores = []
        krng = range(kMin,kMax+1)
        for k in krng:
            w = min(k,N-k-1)
            scores.append(nu.sum(((bimg[k-w:k,:]-127)*(bimg[k+w:k:-1,:]-127)))/w)
        return krng[nu.argmin(scores)]
        
    def selectBarLines(self,agents):
        scores = nu.array([self.assessBarLine(a) for a in agents])
        m = nu.median(scores,0)
        scores[scores[:,0] > m[0],0] = m[0]
        scores[scores[:,1] > m[1],1] = m[1]
        scores[scores[:,2] > m[2],2] = m[2]
        scores[:,0] /= m[0]
        scores[:,1] /= m[1]
        scores[:,2] /= m[2]
        print(scores)
        print(nu.mean(scores,1))
        idx = nu.mean(scores,1)>.8
        return [agents[i] for i in range(len(agents)) if idx[i]]

    def assessBarLineHorzNeighbourhood(self,agent,system0Top,system0Bot,system1Top,system1Bot):
        horz = agent.mean[1]
        l = horz - nu.round(agent.getLineWidth())
        r = horz + nu.round(agent.getLineWidth())
        assert 0 < l-1
        assert r+1 < self.getCorrectedImgSegment().shape[1]-1
        okBlack = 255*(nu.sum([a.getLineWidth() for a in self.staffs[0].staffLineAgents])+
                       nu.sum([a.getLineWidth() for a in self.staffs[1].staffLineAgents]))
        actualBlack0 = nu.sum(nu.mean(self.getCorrectedImgSegment()[system0Top:system0Bot,l-1:l+1],1))
        actualBlack1 = nu.sum(nu.mean(self.getCorrectedImgSegment()[system1Top:system1Bot,r-1:r+1],1))
        return 1.0/(1+nu.abs(1-(.5*okBlack/float(actualBlack0+actualBlack1))))

    def assessBarLineContinuity(self,agent,system0Top,system0Bot,system1Top,system1Bot):
        horz = agent.mean[1]
        assert 0 < horz - 1
        assert horz + 1 < self.getCorrectedImgSegment().shape[1]-1
        w = system0Bot-system0Top + system1Bot-system1Top
        l0 = nu.sum(nu.max(self.getCorrectedImgSegment()[system0Top:system0Bot,horz-1:horz+1],1))
        l1 = nu.sum(nu.max(self.getCorrectedImgSegment()[system1Top:system1Bot,horz-1:horz+1],1))
        return (l0+l1)/float(255*w)

    def assessBarLineEndings(self,agent,system0Top,system0Bot,system1Top,system1Bot):
        horz = agent.mean[1]
        assert 0 < horz - 1
        assert horz + 1 < self.getCorrectedImgSegment().shape[1]-1
        # todo: check vertical ranges
        w1 = int(.1*(system0Bot-system0Top))
        w2 = int(.2*(system0Bot-system0Top))
        l0 = nu.sum(nu.mean(self.getCorrectedImgSegment()[system0Top-w2:system0Top-w1,horz-1:horz+1],1))
        l1 = nu.sum(nu.mean(self.getCorrectedImgSegment()[system1Bot+w1:system1Bot+w2,horz-1:horz+1],1))
        return 1.0/(1+(l0+l1)/255.)

    def assessBarLine(self,agent):
        system0Top = self.getRotator().rotate(self.staffs[0].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system0Bot = self.getRotator().rotate(self.staffs[0].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        system1Top = self.getRotator().rotate(self.staffs[1].staffLineAgents[0].getDrawMean().reshape((1,2)))[0,0]
        system1Bot = self.getRotator().rotate(self.staffs[1].staffLineAgents[-1].getDrawMean().reshape((1,2)))[0,0]
        # staff 0:
        bc = self.assessBarLineContinuity(agent,system0Top,system0Bot,system1Top,system1Bot)
        be = self.assessBarLineEndings(agent,system0Top,system0Bot,system1Top,system1Bot)
        bhn = self.assessBarLineHorzNeighbourhood(agent,system0Top,system0Bot,system1Top,system1Bot)
        return [bc,be,bhn]
        
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
        #r = Rotator(self.getStaffAngle(),self.getLowerMid(),self.getLowerMidLocal())
        r = self.getRotator()
        xx,yy = nu.mgrid[0:self.getSystemHeight(),-halfSystemWidth:halfSystemWidth]
        yy += self.getLowerMidLocal()[1]
        xxr,yyr = r.derotate(xx,yy)
        if True:
            getAntiAliasedImg(self.scrImage.getImg(),xxr,yyr)

        xf = nu.floor(xxr).astype(nu.int)
        xc = nu.ceil(xxr).astype(nu.int)
        yf = nu.floor(yyr).astype(nu.int)
        yc = nu.ceil(yyr).astype(nu.int)
        wxc = xxr%1
        wxf = 1-wxc
        wyc = yyr%1
        wyf = 1-wyc

        #cimg = self.scrImage.getImg()[nu.round(xxr).astype(nu.int),nu.round(yyr).astype(nu.int)]
        cimg = (((wxf+wyf)*self.scrImage.getImg()[xf,yf] + \
                (wxf+wyc)*self.scrImage.getImg()[xf,yc] + \
                (wxc+wyf)*self.scrImage.getImg()[xc,yf] + \
                (wxc+wyc)*self.scrImage.getImg()[xc,yc])/4.0).astype(nu.uint8)
        
        return cimg

def getAntiAliasedImg(img,xx,yy):
        xf = nu.floor(xx).astype(nu.int)
        xc = nu.ceil(xx).astype(nu.int)
        yf = nu.floor(yy).astype(nu.int)
        yc = nu.ceil(yy).astype(nu.int)
        wxc = xx%1
        wxf = 1-wxc
        wyc = yy%1
        wyf = 1-wyc
        return (((wxf+wyf)*img[xf,yf] + 
                 (wxf+wyc)*img[xf,yc] + 
                 (wxc+wyf)*img[xc,yf] + 
                 (wxc+wyc)*img[xc,yc])/4.0).astype(nu.uint8)

if __name__ == '__main__':
    pass

