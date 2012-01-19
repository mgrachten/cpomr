#!/usr/bin/env python

import sys,pickle,os
import numpy as nu
import scipy.stats

def evalBar(name,info):
    img = info['img']
    #print(nu.column_stack((nu.floor(info['staffTops']),nu.ceil(info['staffTops']+info['staffWidth']))))
    be = nu.column_stack((nu.floor(info['staffTops'])[1:]-1,
                          1+nu.ceil(info['staffTops']+info['staffWidth'])[:-1]))
    be = nu.vstack((be[:4],be[5:]))
    interStaffX = nu.hstack([nu.arange(e,b) for b,e in be]).astype(nu.int)
    w = nu.ceil(info['hcoords'][1]-info['hcoords'][0])
    b,e = int(info['hcoords'][0]),int(info['hcoords'][-1])
    interStaffY = nu.append(nu.arange(b-w,b)-1,nu.arange(e,e+w)+1).astype(nu.int)
    N = len(interStaffY)
    mid = N/2.0
    weights = scipy.stats.norm.pdf(nu.arange(N),mid,N/8.0)
    weights = weights/nu.sum(weights)
    sums = nu.sum(img[interStaffX,:],0)[interStaffY]
    sums = sums/(255*len(interStaffX))
    nu.savetxt('/tmp/b/{0}'.format(name),img[interStaffX,:][:,interStaffY])
    nu.savetxt('/tmp/l/{0}'.format(name),nu.column_stack((sums,weights,sums*weights)))
    print('w',nu.sum(weights*sums))
    return nu.sum(weights*sums)
    #nu.array([x for x in product(interStaff,nu.arange(h0,h1-1))])

def main(ff):
    acc = []
    for fn in ff:
        with open(fn,'r') as f:
            acc.extend(list(pickle.load(f).items()))
    acc = dict(acc)
    s = []
    for k,v in acc.items():
        print(k)
        print(v['bar'])
        #print(v['position'])
        parts = os.path.splitext(k)[0].split('-')
        s.append((evalBar(k,v),1 if v['bar'] else 0,int(parts[2]),int(parts[4]),int(parts[5])))
    nu.savetxt('/tmp/s.txt',nu.array(s))

if __name__ == '__main__':
    datfiles = sys.argv[1:]
    main(datfiles)
