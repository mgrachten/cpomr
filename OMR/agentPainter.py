#!/usr/bin/env python

import os,logging
import numpy as nu
from PIL import ImageDraw,ImageFont,Image
from imageUtil import getImageData, writeImageData, makeMask, normalize, jitterImageEdges,getPattern
from utilities import makeColors

class AgentPainter(object):
    def __init__(self,img):
        self.img = nu.array((255-img,255-img,255-img))
        self.imgOrig = nu.array((255-img,255-img,255-img))
        self.maxAgents = 300
        self.colors = makeColors(self.maxAgents)
        self.paintSlots = nu.zeros(self.maxAgents,nu.bool)
        self.agents = {}
        self.log = logging.getLogger(__name__)
    def writeImage(self,fn,absolute=False):
        #print(nu.min(img),nu.max(img))
        self.img = self.img.astype(nu.uint8)
        if absolute:
            fn = fn
        else:
            fn = os.path.join('/tmp',os.path.splitext(os.path.basename(fn))[0]+'.png')
        self.log.info('Writing image to file: {0}'.format(fn))
        writeImageData(fn,self.img.shape[1:],self.img[0,:,:],self.img[1,:,:],self.img[2,:,:])

    def isRegistered(self,agent):
        return self.agents.has_key(agent)
        
    def register(self,agent):
        if self.isRegistered(agent):
            return True
        available = nu.where(self.paintSlots==0)[0]
        if len(available) < 1:
            self.log.warn('No paint slots available')
            return False
        #print('registring {0}'.format(agent.id))
        #print(agent.__hash__())
        self.agents[agent] = available[0]
        self.paintSlots[available[0]] = True
        #self.paintStart(agent.point,self.colors[self.agents[agent]])

    def unregister(self,agent):
        if self.agents.has_key(agent):
            self.paintSlots[self.agents[agent]] = False
            #print('unregistring {0}'.format(agent.id))
            del self.agents[agent]

        else:
            self.log.warn('Unknown agent\n')
        
    def reset(self):
        self.img = self.imgOrig.copy()

    def drawText(self, text, pos, size=30, color=(100,100,100), alpha=.5):
        font = ImageFont.truetype('/usr/share/fonts/truetype/ttf-ubuntu-title/Ubuntu-Title.ttf', 
                                  size)
        size = font.getsize(text) # Returns the width and height of the given text, as a 2-tuple.
        im = Image.new('L', size, 255) # Create a blank image with the given size
        draw = ImageDraw.Draw(im)
        draw.text((0,0), text, font=font, fill=None) #Draw text
        d = 255-nu.array(im.getdata(),nu.uint8).reshape(size[::-1])
        maxX = min(self.img.shape[1]-1,d.shape[0]+pos[0])-pos[0]
        maxY = min(self.img.shape[2]-1,d.shape[1]+pos[1])-pos[1]
        d = d[:maxX,:maxY]
        xx,yy = nu.nonzero(d)
        
        vv = d[xx,yy]/255.
        xx += pos[0]
        yy += pos[1]
        alpha = vv*alpha
        self.img[0,xx,yy] = ((1-alpha)*self.img[0,xx,yy] + alpha*color[0]).astype(nu.uint8)
        self.img[1,xx,yy] = ((1-alpha)*self.img[1,xx,yy] + alpha*color[1]).astype(nu.uint8)
        self.img[2,xx,yy] = ((1-alpha)*self.img[2,xx,yy] + alpha*color[2]).astype(nu.uint8)

    def drawAgentGood(self,agent,rmin=-100,rmax=100):
        if self.agents.has_key(agent):
            #print('drawing')
            #print(agent)
            c = self.colors[self.agents[agent]]
            c1 = nu.minimum(255,c+50)
            c2 = nu.maximum(0,c-100)
            M,N = self.img.shape[1:]
            rng = nu.arange(rmin,rmax)
            xy = nu.round(nu.column_stack((rng*nu.sin(agent.angle*nu.pi)+agent.getDrawMean()[0],
                                           rng*nu.cos(agent.angle*nu.pi)+agent.getDrawMean()[1])))
            idx = nu.logical_and(nu.logical_and(xy[:,0]>=0,xy[:,0]<M),
                                 nu.logical_and(xy[:,1]>=0,xy[:,1]<N))
            alpha = min(.8,max(.1,.5+float(agent.score)/max(1,agent.age)))
            xy = xy[idx,:]
            if xy.shape[0] > 0:
                self.paintRav(xy,c2,alpha)
            #for r in range(rmin,rmax):
            #    x = r*nu.sin(agent.angle*nu.pi)+agent.getDrawMean()[0]
            #    y = r*nu.cos(agent.angle*nu.pi)+agent.getDrawMean()[1]
            #    #print(r,agent.angle,agent.getDrawMean(),x,y)
            #    #print(x,y)
            #    if 0 <= x < M and 0 <= y < N:
            #        self.paint(nu.array((x,y)),c2,alpha)

            self.paintRect(agent.getDrawPoints()[0][0],agent.getDrawPoints()[0][0],
                           agent.getDrawPoints()[0][1],agent.getDrawPoints()[0][1],c)
            #self.paintRect(agent.getDrawMean()[0]+2,agent.getDrawMean()[0]-2,
            #               agent.getDrawMean()[1]+2,agent.getDrawMean()[1]-2,c)

            self.paintRav(agent.getDrawPoints(),c1)
            #for p in agent.getDrawPoints():
            #    self.paint(p,c1)

    def drawAgent(self,agent,rmin=-100,rmax=100,rotator=None):
        if self.agents.has_key(agent):
            c = self.colors[self.agents[agent]]
            c1 = nu.minimum(255,c+50)
            c2 = nu.maximum(0,c-100)
            M,N = self.img.shape[1:]
            #rng = nu.arange(rmin,rmax)
            rng = nu.arange(rmin,rmax,.95)
            xy = nu.round(nu.column_stack((rng*nu.sin(agent.angle*nu.pi)+agent.getDrawMean()[0],
                                           rng*nu.cos(agent.angle*nu.pi)+agent.getDrawMean()[1])))
            if rotator:
                xy = rotator.derotate(xy)
            idx = nu.logical_and(nu.logical_and(xy[:,0]>=0,xy[:,0]<M),
                                 nu.logical_and(xy[:,1]>=0,xy[:,1]<N))
            alpha = min(.8,max(.1,.5+float(agent.score)/max(1,agent.age)))
            xy = xy[idx,:].astype(nu.int)
            if xy.shape[0] > 0:
                self.paintRav(xy,c2,alpha)

            # first point
            if agent.getDrawPoints().shape[0] > 0:
                fPoint = agent.getDrawPoints()[0].reshape((1,2))
                if rotator:
                    fPoint = rotator.derotate(fPoint)[0,:]
                else:
                    fPoint = fPoint[0,:]
                self.paintRect(fPoint[0],fPoint[0],fPoint[1],fPoint[1],c)

            drp = agent.getDrawPoints()
            if rotator:
                drp = rotator.derotate(drp)
            self.paintRav(drp,c1)

    def paintLineSegment(self,coord1,coord2,color,alpha=1):
        #x1=30.5;x2=10;r=int(nu.ceil(nu.abs(x2-x1))); 
        l = nu.sum((coord1-coord2)**2)**.5
        coords = nu.column_stack((nu.linspace(coord1[0],coord2[0],l),nu.linspace(coord1[1],coord2[1],l)))
        self.paintRav(coords,color,alpha)

    def paintRav(self,coords,color,alpha=1):
        idx = (self.img.shape[2]*nu.round(coords[:,0])+nu.round(coords[:,1])).astype(nu.int64)
        self.img[0,:,:].flat[idx] = nu.minimum(255,nu.maximum(0,(1-alpha)*self.img[0,:,:].flat[idx]+alpha*color[0])).astype(nu.uint8)
        self.img[1,:,:].flat[idx] = nu.minimum(255,nu.maximum(0,(1-alpha)*self.img[1,:,:].flat[idx]+alpha*color[1])).astype(nu.uint8)
        self.img[2,:,:].flat[idx] = nu.minimum(255,nu.maximum(0,(1-alpha)*self.img[2,:,:].flat[idx]+alpha*color[2])).astype(nu.uint8)


    def paint(self,coord,color,alpha=1):
        #print('point',coord,img.shape)
        self.img[:,int(coord[0]),int(coord[1])] = (1-alpha)*self.img[:,int(coord[0]),int(coord[1])]+alpha*color

    def paintVLine(self,y,alpha=.5,step=1,color=(100,0,100)):
        if 0 <= y < self.img.shape[2]:
            self.img[0,::step,y] = nu.minimum(255,nu.maximum(0,((1-alpha)*self.img[0,::step,y]+alpha*color[0]))).astype(nu.uint8)
            self.img[1,::step,y] = nu.minimum(255,nu.maximum(0,((1-alpha)*self.img[1,::step,y]+alpha*color[1]))).astype(nu.uint8)
            self.img[2,::step,y] = nu.minimum(255,nu.maximum(0,((1-alpha)*self.img[2,::step,y]+alpha*color[2]))).astype(nu.uint8)

    def paintHLine(self,x,alpha=.5,step=1,color=(0,255,255)):
        if 0 <= x < self.img.shape[1]:
            self.img[0,x,::step] = nu.minimum(255,nu.maximum(0,((1-alpha)*self.img[0,x,::step]+alpha*color[0]))).astype(nu.uint8)
            self.img[1,x,::step] = nu.minimum(255,nu.maximum(0,((1-alpha)*self.img[1,x,::step]+alpha*color[1]))).astype(nu.uint8)
            self.img[2,x,::step] = nu.minimum(255,nu.maximum(0,((1-alpha)*self.img[2,x,::step]+alpha*color[2]))).astype(nu.uint8)


    def paintRect(self,xmin,xmax,ymin,ymax,color,alpha=.5):
        rectSize = 10
        N,M = self.img.shape[1:]
        t = int(max(0,xmin-nu.floor(rectSize/2.)))
        b = int(min(N-1,xmax+nu.floor(rectSize/2.)))
        l = int(max(0,ymin-nu.floor(rectSize/2.)))
        r = int(min(M-1,ymax+nu.floor(rectSize/2.)))
        for i,c in enumerate(color):
            self.img[i,t:b,l] = c
            self.img[i,t:b,r] = c
            self.img[i,t,l:r] = c
            self.img[i,b,l:r+1] = c


if __name__ == '__main__':
    pass
