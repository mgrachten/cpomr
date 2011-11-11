#!/usr/bin/env python

import sys,os,pickle
import numpy as nu
from stafffind import findStaffLines,selectColumns,getCrossings,mergeAgents
from imageUtil import getPattern,normalize
from agent import Agent,AgentPainter
from angleEstimation import angleEstimator

class BarLineAgent(Agent):
    targetAngle = .5 # in rad/(2*pi), for example .5 is vertical
    # maxError should depend on imageSize
    # good values (empirically established):
    # maxError=5 for images of approx 826x1169; seems to work also for images of 2550x3510
    # larger resolutions may need a higher value of maxError
    maxError = 5 # mean perpendicular distance of points to line (in pixels)
    maxAngleDev = 3/180. # 
    minScore = -2

class BarLinesEvaluator(object):
    def __init__(self,system_l,system_r,staffLinewidth=1):
        self.staffLinewidth = staffLinewidth
        self.system_l = system_l
        self.system_r = system_r
        self.breakpoints = []
    def sortBarAgents(self,agentsold):
        agents = [a for a in agentsold if a.getLineThickness() <= 10*self.staffLinewidth
                  and self.system_l < a.mean[1] < self.system_r ]
        agents.sort(key=lambda x: -x.score)
        scores = nu.array([x.score for x in agents])
        self.breakpoints.append(nu.argmin(nu.diff(scores))+1)
        print('seeing bar lines',self.breakpoints)
        #print('seeing bar lines',nu.array(self.breakpoints))
        return agents
    

def assessBarlines(agents,img,topPoints,botPoints,staffDistance):
    return agents,False,5
    
def findBarsInSystem(img,staffAgents,ap):
    N,M = img.shape
    top = int(staffAgents[0].mean[0])
    bottom = int(staffAgents[-1].mean[0])
    staffDistance = staffAgents[1].mean[0]-staffAgents[0].mean[0]
    staffLinewidth = nu.mean([a.getLineThickness() for a in staffAgents])
    system_l,system_r = guessSystemLR(img,staffAgents)
    topPoints = (staffAgents[0].mean,nu.array((staffAgents[0].mean[0]-staffAgents[0].mean[1]*nu.tan(staffAgents[0].getAngle()*nu.pi),0)))
    botPoints = (staffAgents[-1].mean,nu.array((staffAgents[-1].mean[0]-staffAgents[-1].mean[1]*nu.tan(staffAgents[-1].getAngle()*nu.pi),0)))
    bins = 5
    hsums = nu.sum(img[top:bottom,:],1)
    columns,colBins = selectColumns(hsums,bins)
    columns += top
    agents = []

    ap.register(staffAgents[0])
    ap.register(staffAgents[-1])

    #taprev= []
    draw = True
    #draw = False
    stop = False
    nbars = 0
    ble = BarLinesEvaluator(system_l,system_r,staffLinewidth)
    for i,r in enumerate(columns[:53]):
        print('row',i)
        agentsnew = getCrossings(img[r,:],agents,BarLineAgent,N,vert=r)
        #agents = agentsnew[:]
        #agents.sort(key=lambda x: -x.score)
        print('before sort')
        for a in sorted(agentsnew,key=lambda x: -x.score):
                print(a)
        if len(agentsnew)> 1:
            agentsnew = ble.sortBarAgents(agentsnew)
        print('after sort')
        for a in agentsnew:
                print(a)
        if len(agentsnew)> 2:
            agentsnew = mergeAgents(agentsnew)
            #verifyBarLine(a,img,topPoints,botPoints,staffDistance)
            if False: #i > 10:
                ta,stop,nbars = assessBarlines(ta,img,topPoints,botPoints,staffDistance)
                if stop:
                    agents = ta[:]
                    break
        else:
            pass #agentsnew = agentsnew[:]

        print('after merge')
        for a in sorted(agentsnew,key=lambda x: -x.score):
            print(a)

        if draw:
            sagentsnew = set(agentsnew)
            setagents = set(agents)
            born = sagentsnew.difference(setagents)
            died = setagents.difference(sagentsnew)
            ap.reset()
            ap.drawAgent(staffAgents[0])
            ap.drawAgent(staffAgents[-1])
            for a in born:
                ap.register(a)
            for a in died:
                ap.unregister(a)
            for a in agentsnew:
                #print(a)
                #print(a.lineThickness),
                #print(a.points)
                if a.age > 1:
                    ap.drawAgentGood(a,-300,300)
                #print(verifyBarLine(a,img,topPoints,botPoints,staffDistance,ap))
            print('drew agents',len(agentsnew))
            ap.paintHLine(r)
            f0,ext = os.path.splitext(fn)
            print(f0,ext)
            #ap.writeImage(fn.replace('.png','-{0:04d}-c{1}.png'.format(i,c)))
            ap.writeImage(f0+'-{0:04d}-r{1}'.format(i,r)+'.png')

        # DON'T DELETE
        agents = agentsnew[:]
        #taprev = ta[:]
        
    for a in agents:
        print(a)
        #print(verifyBarLine(a,img,topPoints,botPoints,staffDistance,ap))
    #print('####################################################################################')
    #print('found {0} bars:'.format(nbars))
    return agents[:nbars]

def guessSystemLR(img,sagents):
    """use vsums to find white spaces around system
    """
    # el cheapo: do vsums independent of staffline angles
    # better: sum in the direction perpendicular to the stafflines
    vsums = nu.sum(img[sagents[0].mean[0]:sagents[-1].mean[0],:],0)
    zindex = nu.where(vsums == 0)[0]
    N = img.shape[1]
    lz = zindex < N/2
    lzindex = zindex[lz]
    rzindex = zindex[nu.logical_not(lz)]
    assert len(lzindex) > 0
    assert len(rzindex) > 0
    lm = nu.median(lzindex)
    rm = nu.median(rzindex)
    return lm,rm

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
    bgThreshold = 20
    #img = img[:825,:]
    img[img< bgThreshold] = 0
    #img[img>= bgThreshold] = 255
    angleEstimator(img)
    sys.exit()
    agentfn = os.path.join('/tmp/',os.path.splitext(os.path.basename(fn))[0]+'.agents')
    try:
        with open(agentfn,'r') as f:
            agents = pickle.load(f)
    except Exception as e:
        print(e)
        agents = findStaffLines(img,fn)
        with open(agentfn,'w') as f:
            pickle.dump(agents,f)
    ap = AgentPainter(img)
    # select system; -1 to do all
    s = -1
    bars = []
    for i,ba in enumerate(agents):
        if i == s:
            bars.append(findBarsInSystem(img,ba,ap))
        else:
            bars.append([])
    ap.reset()
    for i,ba in enumerate(agents):
        for a in ba:
            ap.register(a)
            print('staff',a.mean,a.getLineThickness())
            ap.drawAgentGood(a,-2000,2000)
        if i == s:
            #ap.register(ba[0])
            #print([a.getLineThickness() for a in ba])
            #ap.register(ba[-1])
            #ap.drawAgentGood(ba[-1],-2000,2000)
            #bars = findBarsInSystem(img,ba,ap)
            print('system')
            for b in bars[i]:
                ap.register(b)
                ap.drawAgentGood(b,-300,300)
            #print('that was for system',i)
    ap.writeImage(fn)
