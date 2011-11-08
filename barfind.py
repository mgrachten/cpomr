#!/usr/bin/env python

import sys,os,pickle
import numpy as nu
from stafffind import findStaffLines,selectColumns,getCrossings
from imageUtil import getPattern,normalize
from agent import Agent,AgentPainter

class BarLineAgent(Agent):
    targetAngle = .5 # in rad/(2*pi), for example .5 is vertical
    # maxError should depend on imageSize
    # good values (empirically established):
    # maxError=5 for images of approx 826x1169; seems to work also for images of 2550x3510
    # larger resolutions may need a higher value of maxError
    maxError = 2 # mean perpendicular distance of points to line (in pixels)
    maxAngleDev = 3/180. # 
    minScore = -2

def sortBarAgents(agents):
    agents.sort(key=lambda x: -x.score)
    scores = nu.array([x.score for x in agents])
    print('seeing bar lines',nu.argmin(nu.diff(scores))+1)
    return agents

def verifyEndpoint(img,endpoint,pv,ap):
    if ap:
        red = nu.array((255,0,0))
    results = []
    # top
    for i in range(pv.shape[0]):
        p = endpoint+pv[i,:2]
        results.append(img[p[0],p[1]].astype(nu.bool) == pv[i,2].astype(nu.bool))
        if ap:
            ap.paint(p,red,.5)
    return all(results)

def verifyTop(img,endpoint,vm,hm,ap):
    if ap:
        red = nu.array((255,0,0))
    results = []
    #print(endpoint)
    # top
    #print('top')
    try:
        ri = nu.round(normalize(img[int(endpoint[0]-vm),int(endpoint[1]-hm):int(endpoint[1]+hm)]))
        ri -= nu.min(ri)
        results.append(len(nu.nonzero(ri)[0]) == 0)
        #print(ri)
        i = 3
        ri = nu.round(normalize(img[int(endpoint[0]+i*vm),int(endpoint[1]-hm):int(endpoint[1]+hm)]))
        nz = nu.nonzero(ri-nu.min(ri))[0]
        results.append(len(nz) > 0 and nu.all(nz > 0) and nu.all(nz < len(ri)-1 ))
        i = 5
        ri = nu.round(normalize(img[int(endpoint[0]+i*vm),int(endpoint[1]-hm):int(endpoint[1]+hm)]))
        nz = nu.nonzero(ri-nu.min(ri))[0]
        results.append(len(nz) > 0 and nu.all(nz > 0) and nu.all(nz < len(ri)-1 ))
        i = 7
        ri = nu.round(normalize(img[int(endpoint[0]+i*vm),int(endpoint[1]-hm):int(endpoint[1]+hm)]))
        nz = nu.nonzero(ri-nu.min(ri))[0]
        results.append(len(nz) > 0 and nu.all(nz > 0) and nu.all(nz < len(ri)-1 ))
        #print(x0)
        #print(x1)
        #if ap:
        #    #ap.paint(img[int(endpoint-vm),int(endpoint-hm):int(endpoint+hm)]
        return all(results)
    except ValueError:
        return False

def verifyBot(img,endpoint,vm,hm,ap):
    if ap:
        red = nu.array((255,0,0))
    results = []
    # bot
    try:
        ri = nu.round(normalize(img[int(endpoint[0]+vm),int(endpoint[1]-hm):int(endpoint[1]+hm)]))
        ri -= nu.min(ri)
        results.append(len(nu.nonzero(ri)[0]) == 0)
        i = 3
        ri = nu.round(normalize(img[int(endpoint[0]-i*vm),int(endpoint[1]-hm):int(endpoint[1]+hm)]))
        nz = nu.nonzero(ri-nu.min(ri))[0]
        results.append(len(nz) > 0 and nu.all(nz > 0) and nu.all(nz < len(ri)-1 ))
        i = 5
        ri = nu.round(normalize(img[int(endpoint[0]-i*vm),int(endpoint[1]-hm):int(endpoint[1]+hm)]))
        nz = nu.nonzero(ri-nu.min(ri))[0]
        results.append(len(nz) > 0 and nu.all(nz > 0) and nu.all(nz < len(ri)-1 ))
        i = 7
        ri = nu.round(normalize(img[int(endpoint[0]-i*vm),int(endpoint[1]-hm):int(endpoint[1]+hm)]))
        nz = nu.nonzero(ri-nu.min(ri))[0]
        results.append(len(nz) > 0 and nu.all(nz > 0) and nu.all(nz < len(ri)-1 ))
        return all(results)
    except ValueError:
        return False

def verifyBarLine(agent,img,topPoints,botPoints,staffDistance,ap=None):
    #print('intersection',agent.mean,agent.getAngle())
    #print('topPoints',topPoints)
    vmargin = staffDistance/2.0
    hmargin = .5*vmargin
    checkPointsTop = nu.array([#[0,0,1], # end point
                               [-1,0,0], # above
                               [3,0,1], # below
                               [3,-1,0],
                               [3,1,0]])
    checkPointsTop = nu.dot(checkPointsTop,nu.diag((vmargin,hmargin,1)))
    checkPointsBot = nu.array([#[0,0,1], # end point
                               [-3,0,1], # above
                               [1,0,0], # below
                               [-3,-1,0],
                               [-3,1,0]])
    checkPointsBot = nu.dot(checkPointsBot,nu.diag((vmargin,hmargin,1)))
    top = agent.getIntersection(*topPoints)
    #x0 = verifyEndpoint(img,top,checkPointsTop,ap)
    x0 = verifyTop(img,top,vmargin,hmargin,ap)
    bot = agent.getIntersection(*botPoints)
    #x1 = verifyEndpoint(img,bot,checkPointsBot,ap)
    x1 = verifyBot(img,bot,vmargin,hmargin,ap)
    #print('top',x0)
    #print('bot',x1)
    return x0 and x1
    # print('top',img[int(top[0]),int(top[1])],1)
    # print('above top',img[int(top[0]-vmargin),int(top[1])],0)
    # print('below top',img[int(top[0]+vmargin),int(top[1])],1)
    # print('left below top',img[int(top[0]+vmargin),int(top[1]-hmargin)],0)
    # print('right below top',img[int(top[0]+vmargin),int(top[1]+hmargin)],0)
    #agent.getAngle()*nu.pi


def assessBarlinesOld(agents,img,topPoints,botPoints,staffDistance):
    passed = nu.array([verifyBarLine(a,img,topPoints,botPoints,staffDistance) for a in agents]).astype(nu.int)
    #print(len(agents),passed)
    nz = nu.nonzero(passed)[0]
    for i,a in enumerate(agents):
        a.score += 1 if passed[i] else -1
        
    if len(nz) > 1:
        if nz[0] == 0 and nu.max(nu.diff(nz)) == 1:
            return agents,True,len(nz)
        else:
            return agents,False,len(nz)
    else:
        return agents,False,len(nz)

def assessBarlines(agents,img,topPoints,botPoints,staffDistance):
    return agents,False,5
    


def findBarsInSystem(img,staffAgents,ap):
    N,M = img.shape
    top = int(staffAgents[0].mean[0])
    staffDistance = staffAgents[1].mean[0]-staffAgents[0].mean[0]
    #topPoints = (staffAgents[0].points[0,:],staffAgents[0].points[1])
    #print('xx',staffAgents[0].mean[0],staffAgents[0].mean[1],nu.tan(staffAgents[0].getAngle()*nu.pi))
    topPoints = (staffAgents[0].mean,nu.array((staffAgents[0].mean[0]-staffAgents[0].mean[1]*nu.tan(staffAgents[0].getAngle()*nu.pi),0)))
    #print(topPoints)
    #botPoints = (staffAgents[-1].points[0,:],staffAgents[-1].points[1])
    botPoints = (staffAgents[-1].mean,nu.array((staffAgents[-1].mean[0]-staffAgents[-1].mean[1]*nu.tan(staffAgents[-1].getAngle()*nu.pi),0)))
    bottom = int(staffAgents[-1].mean[0])
    bins = 5
    hsums = nu.sum(img[top:bottom,:],1)
    columns,colBins = selectColumns(hsums,bins)
    columns += top
    agents = []

    ap.register(staffAgents[0])
    ap.register(staffAgents[-1])

    taprev= []
    #draw = True
    draw = False
    stop = False
    nbars = 0
    for i,r in enumerate(columns[:50]):
        print('row',i)
        agentsnew = getCrossings(img[r,:],agents,BarLineAgent,N,vert=r)
        #agents = agentsnew[:]
        #agents.sort(key=lambda x: -x.score)
        if len(agentsnew)> 2:
            ta = sortBarAgents(agentsnew)
            #verifyBarLine(a,img,topPoints,botPoints,staffDistance)
            if i > 10:
                ta,stop,nbars = assessBarlines(ta,img,topPoints,botPoints,staffDistance)
                if stop:
                    agents = ta[:]
                    break
        else:
            ta = agentsnew[:]

        if draw:
            sagentsnew = set(ta)
            setagents = set(taprev)
            born = sagentsnew.difference(setagents)
            died = setagents.difference(sagentsnew)
            ap.reset()
            ap.drawAgent(staffAgents[0])
            ap.drawAgent(staffAgents[-1])
            for a in born:
                ap.register(a)
            for a in died:
                ap.unregister(a)
            for a in ta:
                print(a)
                #print(a.lineThickness),
                #print(a.points)
                ap.drawAgentGood(a,-300,300)
                #print(verifyBarLine(a,img,topPoints,botPoints,staffDistance,ap))
            print('drew agents',len(ta))
            ap.paintHLine(r)
            f0,ext = os.path.splitext(fn)
            print(f0,ext)
            #ap.writeImage(fn.replace('.png','-{0:04d}-c{1}.png'.format(i,c)))
            ap.writeImage(f0+'-{0:04d}-r{1}'.format(i,r)+'.png')

        # DON'T DELETE
        agents = agentsnew[:]
        taprev = ta[:]
        
    for a in agents:
        print(a),
        print(verifyBarLine(a,img,topPoints,botPoints,staffDistance,ap))
    #print('####################################################################################')
    #print('found {0} bars:'.format(nbars))
    return agents[:nbars]


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
    img[img< bgThreshold] = 0
    #img[img>= bgThreshold] = 255
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
    s = 3
    bars = []
    for i,ba in enumerate(agents):
        if i == s:
            bars.append(findBarsInSystem(img,ba,ap))
        else:
            bars.append([])
    ap.reset()
    for i,ba in enumerate(agents):
        if i == s:
            ap.register(ba[0])
            ap.register(ba[-1])
            ap.drawAgentGood(ba[0],-2000,2000)
            ap.drawAgentGood(ba[-1],-2000,2000)
            #bars = findBarsInSystem(img,ba,ap)
            print('system')
            for b in bars[i]:
                ap.register(b)
                ap.drawAgentGood(b,-300,300)
            #print('that was for system',i)
    ap.writeImage(fn)
