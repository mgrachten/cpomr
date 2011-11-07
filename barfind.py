#!/usr/bin/env python

import sys,os,pickle
import numpy as nu
from stafffind import findStaffLines,selectColumns,getCrossings
from imageUtil import getPattern
from agent import Agent,AgentPainter

class BarLineAgent(Agent):
    targetAngle = .5 # in rad/(2*pi), for example .5 is vertical
    # maxError should depend on imageSize
    # good values (empirically established):
    # maxError=5 for images of approx 826x1169; seems to work also for images of 2550x3510
    # larger resolutions may need a higher value of maxError
    maxError = 5 # mean perpendicular distance of points to line (in pixels)
    maxAngleDev = 2/180. # 
    minScore = -5

def findBarsInSystem(img,staffAgents):
    N,M = img.shape
    top = int(staffAgents[0].mean[0])
    bottom = int(staffAgents[-1].mean[0])
    bins = 3
    hsums = nu.sum(img[top:bottom,:],1)
    columns,colBins = selectColumns(hsums,bins)
    columns += top
    agents = []
    ap = AgentPainter(img)
    taprev= []
    draw = True
    #draw = False
    for i,r in enumerate(columns[:30]):
        print('row',i)
        agentsnew = getCrossings(img[r,:],agents,BarLineAgent,N,vert=r)
        #agents = agentsnew[:]
        #agents.sort(key=lambda x: -x.score)
        if len(agentsnew)> 40:
            ta = sortAgents(agentsnew)
        else:
            ta = agentsnew[:]

        if draw:
            sagentsnew = set(ta)
            setagents = set(taprev)
            born = sagentsnew.difference(setagents)
            died = setagents.difference(sagentsnew)
            ap.reset()
            for a in born:
                ap.register(a)
            for a in died:
                ap.unregister(a)
            for a in ta:
                print(a)
                print(a.points)
                ap.drawAgent(a)
            print('drew agents',len(ta))
            ap.paintHLine(r)
            f0,ext = os.path.splitext(fn)
            print(f0,ext)
            #ap.writeImage(fn.replace('.png','-{0:04d}-c{1}.png'.format(i,c)))
            ap.writeImage(f0+'-{0:04d}-r{1}'.format(i,r)+'.png')

        # DON'T DELETE
        agents = agentsnew[:]
        taprev = ta[:]

if __name__ == '__main__':
    fn = sys.argv[1]
    print('Loading image...'),
    sys.stdout.flush()
    try:
        img = 255-getPattern(fn,False,False)
    except IOError as e: 
        print('problem')
        raise e
        sys.exit()#pass
    print('Done')
    bgThreshold = 10
    img[img< bgThreshold] = 0
    agentfn = os.path.join('/tmp/',os.path.splitext(os.path.basename(fn))[0]+'.agents')
    try:
        with open(agentfn,'r') as f:
            agents = pickle.load(f)
    except Exception as e:
        print(e)
        agents = findStaffLines(img,fn)
        with open(agentfn,'w') as f:
            pickle.dump(agents,f)
    sys.exit()
    for ba in agents:
        print('system')
        findBarsInSystem(img,ba)
        sys.exit()
