#!/usr/bin/env python

import sys,pickle,os
import numpy as nu
import scipy.stats
from arffHandling import ArffObject

def getCrit1ab(info):
    """
    """
    # y's of top and bot staffs
    t0,b0,t1,b1 = nu.round(info['staffTops'][nu.array((0,4,5,9))])
    w = nu.round(1*info['staffWidth'])
    margin = 2
    marg2 = 1
    intraStaffXa = nu.append(nu.arange(t0+marg2,b0-marg2),nu.arange(t1+marg2,b1-marg2)).astype(nu.int)
    intraStaffXb = nu.arange(b0+marg2,t1-marg2).astype(nu.int)
    if len(info['hcoords']) == 2:
        l,r = info['hcoords'].astype(nu.int)
        intraStaffY = nu.arange(l+marg2,r-marg2)
    else:
        l0,r0,l1,r1 = info['hcoords'].astype(nu.int)
        intraStaffY = nu.append(nu.arange(l0+marg2,r0-marg2),nu.arange(l1+marg2,r1-marg2))
    crit1a = nu.mean(info['img'][intraStaffXa,:][:,intraStaffY])/255.
    crit1b = nu.mean(info['img'][intraStaffXb,:][:,intraStaffY])/255.
    crit1c = nu.mean(info['img'][nu.arange(t0-w-margin,t0-margin).astype(nu.int),:][:,intraStaffY])/255.
    crit1d = nu.mean(info['img'][nu.arange(b1+margin,b1+margin+w).astype(nu.int),:][:,intraStaffY])/255.
    return [crit1a,crit1b,crit1c,crit1d]

    
def evalBar(name,info):
    img = info['img']
    margin = 2
    #print(nu.column_stack((nu.floor(info['staffTops']),nu.ceil(info['staffTops']+info['staffWidth']))))
    be = nu.column_stack((nu.floor(info['staffTops'])[1:]-margin,
                          margin+nu.ceil(info['staffTops']+info['staffWidth'])[:-1]))
    be = nu.vstack((be[:4],be[5:]))
    mb,me = be[4,:]
    interStaffX = nu.arange(me,mb).astype(nu.int)
    intraStaffX = nu.hstack([nu.arange(e,b) for b,e in be]).astype(nu.int)
    #w = nu.ceil(info['hcoords'][1]-info['hcoords'][0])
    w = nu.round(1*info['staffWidth'])
    b,e = int(info['hcoords'][0]),int(info['hcoords'][-1])
    intraStaffY = nu.append(nu.arange(b-w,b)-margin,nu.arange(e,e+w)+margin).astype(nu.int)
    N = len(intraStaffY)
    mid = (N-1)/2.0
    weights = scipy.stats.norm.pdf(nu.arange(N),mid,N/8.0)
    weights = weights/nu.sum(weights)
    sums = nu.sum(img[intraStaffX,:],0)[intraStaffY]
    sums = sums/(255*len(intraStaffX))
    sums3 = nu.sum(img[interStaffX,:],0)[intraStaffY]
    sums3 = sums3/(255*len(interStaffX))
    crit = nu.sum(weights*sums)
    crit3 = nu.sum(weights*sums3)
    #nu.savetxt('/tmp/b/{0}'.format(name),img[intraStaffX,:][:,intraStaffY])
    #nu.savetxt('/tmp/l/{0}'.format(name),nu.column_stack((sums,weights,sums*weights)))
    return [crit,crit3]+getCrit1ab(info)
    #nu.array([x for x in product(intraStaff,nu.arange(h0,h1-1))])

def getnn(d,names,k=1):
    for i,x in enumerate(d):
        nidx = nu.argsort(nu.sum((d[:,1:]-x[1:])**2,1))
        if any(d[nidx[1:1+k],0]!=x[0]):
            print(i,names[i])
def main(ff):
    acc = []
    for fn in ff:
        with open(fn,'r') as f:
            acc.extend(list(pickle.load(f).items()))
    acc = dict(acc)
    s = []
    keys = acc.keys()
    keys.sort()
    for i,k in enumerate(keys):
        v = acc[k]
        #print(k)
        #print(v['bar'])
        #print(v['position'])
        parts = os.path.splitext(k)[0].split('-')
        eb = evalBar(k,v)
        s.append(eb+[1 if v['bar'] else 0])
        print(i+1,k,s[-1][-1],logReg(eb))
    #s = nu.array(s)
    #s = s[s[:,1]>=.05,:]
    #names = nu.array(keys)[s[:,1]>=.05]
    attributes = [('crit1','numeric'),
                  ('crit3','numeric'),
                  ('crit2','numeric'),
                  ('crit4','numeric'),
                  ('crit5','numeric'),
                  ('crit6','numeric'),
                  ('bar',(1,0))]
    arff = ArffObject(relation='bars',
                      attributes=attributes,
                      instances=s)
    arff.writeToFile('/tmp/b.arff',mode='arff')
    #GETNN(s,names)
    nu.savetxt('/tmp/s.txt',s)

def logReg(instance):
    c0intercept = -16.2
    class0 = nu.array([-9.08,-0.93,17.34,7.94,-1.42,-3.49])
    return nu.dot(nu.array(instance),class0)+c0intercept

def isBar(name,info):
    eb = evalBar(name,info)
    return logReg(eb) > 0

if __name__ == '__main__':
    datfiles = sys.argv[1:]
    try:
        os.mkdir('/tmp/b')
    except:
        pass
    try:
        os.mkdir('/tmp/l')
    except:
        pass
    main(datfiles)
