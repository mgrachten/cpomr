#!/usr/bin/env python

import sys,pickle,os
import numpy as nu
import scipy.stats

def getCrit1ab(info):
    """
    """
    # y's of top and bot staffs
    t0,b0,t1,b1 = nu.round(info['staffTops'][nu.array((0,4,5,9))])
    intraStaffXa = nu.append(nu.arange(t0,b0),nu.arange(t1,b1)).astype(nu.int)
    intraStaffXb = nu.arange(b0,t1).astype(nu.int)
    if len(info['hcoords']) == 2:
        l,r = info['hcoords'].astype(nu.int)
        intraStaffY = nu.arange(l,r)
    else:
        l0,r0,l1,r1 = info['hcoords'].astype(nu.int)
        intraStaffY = nu.append(nu.arange(l0,r0),nu.arange(l1,r1))
    crit1a = nu.mean(info['img'][intraStaffXa,:][:,intraStaffY])/255.
    crit1b = nu.mean(info['img'][intraStaffXb,:][:,intraStaffY])/255.
    return [crit1a,crit1b]

def getCrit2(info):
    """
    """
    # y's of top and bot staffs
    t0,b0,t1,b1 = nu.round(info['staffTops'][nu.array((0,4,5,9))])
    intraStaffX = nu.append(nu.arange(t0,b0),nu.arange(t1,b1)).astype(nu.int)
    if len(info['hcoords']) == 2:
        l,r = info['hcoords'].astype(nu.int)
        intraStaffY = nu.arange(l,r)
    else:
        l0,r0,l1,r1 = info['hcoords'].astype(nu.int)
        intraStaffY = nu.append(nu.arange(l0,r0),nu.arange(l1,r1))
    crit1 = nu.mean(info['img'][intraStaffX,:][:,intraStaffY])/255.
    return crit1

    
def evalBar(name,info):
    img = info['img']
    #print(nu.column_stack((nu.floor(info['staffTops']),nu.ceil(info['staffTops']+info['staffWidth']))))
    be = nu.column_stack((nu.floor(info['staffTops'])[1:]-1,
                          1+nu.ceil(info['staffTops']+info['staffWidth'])[:-1]))
    be = nu.vstack((be[:4],be[5:]))
    interStaffX = nu.hstack([nu.arange(e,b) for b,e in be]).astype(nu.int)
    #w = nu.ceil(info['hcoords'][1]-info['hcoords'][0])
    w = nu.round(1.5*info['staffWidth'])
    b,e = int(info['hcoords'][0]),int(info['hcoords'][-1])
    interStaffY = nu.append(nu.arange(b-w,b)-1,nu.arange(e,e+w)+1).astype(nu.int)
    N = len(interStaffY)
    mid = (N-1)/2.0
    weights = scipy.stats.norm.pdf(nu.arange(N),mid,N/6.0)
    weights = weights/nu.sum(weights)
    sums = nu.sum(img[interStaffX,:],0)[interStaffY]
    sums = sums/(255*len(interStaffX))
    crit = nu.sum(weights*sums)
    #print('w',w)
    if info['bar'] == False and crit < .05:
        print(name,crit,getCrit1b(info))
    if info['bar'] == True and crit > .3:
        print(name,crit,getCrit1ab(info))
    nu.savetxt('/tmp/b/{0}'.format(name),img[interStaffX,:][:,interStaffY])
    nu.savetxt('/tmp/l/{0}'.format(name),nu.column_stack((sums,weights,sums*weights)))
    return [crit]+getCrit1ab(info)
    #nu.array([x for x in product(interStaff,nu.arange(h0,h1-1))])

def getnn(d,names,k=3):
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
    for k in keys:
        v = acc[k]
        #print(k)
        #print(v['bar'])
        #print(v['position'])
        parts = os.path.splitext(k)[0].split('-')
        s.append([1 if v['bar'] else 0]+evalBar(k,v))
    s = nu.array(s)
    s = s[s[:,1]>=.05,:]
    names = nu.array(keys)[s[:,1]>=.05]
    getnn(s,names)
    nu.savetxt('/tmp/s.txt',s)

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
