#!/usr/bin/env python
import numpy as nu
import sys,os
from scipy import interpolate


class IndexOverflowException(Exception):
    def __init__(self,itype):
        self.itype = itype
    def __str__(self):
        return 'The product of the length of the sequences can not exceed {0}\
 in the current index data type of {1} bytes. Choose smaller sequences or a larger index data type.'.format(2**(8*self.itype(0.0).nbytes),self.itype(0.0).nbytes)

def costDeriv(s,t,i,j):
    return nu.sum(nu.abs(s[i,:] - t[j,:])**2)**.5
    #k = 2
    #return nu.sum(nu.abs(s[i,:] - t[j,:])**k)**(k**-1)
    #return nu.abs(s[i,0] - t[j,0]) + nu.abs(s[i,0] - t[j,0])**.5

def costBasic(s,t,i,j):
    # return (s[i] - t[j])**2
    # return abs(s[i] - t[j])
    #return abs(s[i] - t[j])+.0001*abs(i-j)
    #c = 10
    return abs(s[i] - t[j])#+c*abs(i/float(s.shape[0])-j/float(t.shape[0]))

# choose which cost function to use
cost = costDeriv
    
class CircMatrix:
    def __init__(self,K,L,dtype=nu.float,initVal=0):
        self.K = K
        self.L = L
        self.data = nu.zeros((K,L),dtype)+initVal
        self.current = (0,0)
    def __getitem__(self,(i,j)):
        return self.data[i%self.K,j%self.L]

    def __setitem__(self,(i,j),v):
        self.current = (i%self.K,j%self.L)
        self.data[i%self.K,j%self.L] = v

    def getBuffer(self):
        return self.data

class DTWDataManager:
    # direction tokens
    ROW = 0
    DIAG = 1
    COL = 2
    def __init__(self,M,N,K,L):
        # M -- length of source seq
        # N -- length of target seq
        # K -- source size of window
        # L -- target size of window
        self.M = M
        self.N = N
        self.K = min(K,M)
        self.L = min(L,N)
        
        # Data type to use for the cost matrix
        self.dtype = nu.double
        # Data type to use for the matrix indices, 
        # should be a numpy type, for overflow check
        self.itype = nu.uint32

        # check for sufficient byte size to contain indices
        if (self.M*self.N) > 2**(8*self.itype(0.0).nbytes):
            raise IndexOverflowException(self.itype)

        # rotating data structure for storing the costs
        self.m = CircMatrix(self.K+1,self.L+1,dtype=self.dtype,initVal=nu.inf)
        # start from cost 0
        self.setCost(-1,-1,0.0)

        # Determine the upperlimit of the band surface
        # (for keeping track of the path):
        # We will not need to calculate more than this amount of cells
        Z = self.M*self.L+self.N*self.K-self.K*self.L 
        # This is for storing child-parent cell relations per row: (cell,parent cell) 
        # The index z is factored into row and column indices as follows:
        # i=z%M; j=z/M
        self.predecessors = nu.zeros((Z,2),self.itype)
        self.pred_i = 0
        # optimization: make persistent storage for parentCosts
        self.parentCosts = nu.zeros(3,dtype=self.dtype)

    def getPath(self):
        p = self.predecessors[nu.argsort(self.predecessors[:,0]),:]
        path = nu.zeros(p.shape[0],nu.uint32)
        j = 0
        path[j] = p[-1,0]
        i = p.shape[0]
        while path[j] > 0:
            i = nu.searchsorted(p[:i,0],path[j])
            j = j+1
            path[j] = p[i,1] 
        return nu.array([(z%self.M, z/self.M) for z in path[j::-1]])

    def addPredecessor(self,i,parent_i):
        self.predecessors[self.pred_i] = (i,parent_i)
        self.pred_i = self.pred_i +1

    def addParent(self,cell,parent):
        self.parents[cell] = parent

    def setCost(self,i,j,c):
        """
        Set the cell (i,j) to value c
        """
        self.m[i,j] = c

    def rowMin(self,i):
        l = nu.argmin(self.m.getBuffer()[i%self.m.K,:])
        return ( (self.L+i%self.L-l)%self.L ,self.m[i,l] )

    def colMin(self,j):
        k = nu.argmin(self.m.getBuffer()[:,j%self.m.L])
        return ( (self.K+j%self.K-k)%self.K ,self.m[k,j] )

    def getCost(self):
        return self.m[-2,-2]

    def getDirection(self,i,j):
        """
        Deterimine the most promising direction to proceed, 
        based the results up to cell i,j
        
        """
        k,m = self.rowMin(i)
        l,n = self.colMin(j)
        if k < self.K/2 and l < self.L/2:
            return self.DIAG
        if m < n:
            return self.ROW
        else:
            return self.COL

    def addCost(self,i,j,c):
        """
        Set the m(i,j) to value c + min(m(i-1,j),m(i,j-1),m(i-1,j-1));
        Also store which is the minimal predecessor of (i,j)
        """
        self.parentCosts = (self.m[i-1,j  ],
                            self.m[i  ,j-1],
                            self.m[i-1,j-1])

        k = nu.argmin(self.parentCosts)
        self.m[i,j] = self.parentCosts[k] + c
        #print(i,j,self.m[i,j])
        #print(self.m.getBuffer())
        # i=z%M; j=z/M
        z = self.M*j+i
        if k == 0: 
            zParent = self.M*j+(i-1)
            # parent = (i-1,j)
        elif k == 1:
            zParent = self.M*(j-1)+i
            # parent = (i,j-1)
        else:
            zParent = self.M*(j-1)+(i-1)
            # parent = (i-1,j-1)
        self.addPredecessor(z,zParent)
        #self.addParent((i,j),parent)

class Navigator:
    """Steer the window along a previous path,
    possibly with smoothing the path
    """
    def __init__(self,path,memory=1):
        self.i = 0
        self.memory = memory
        self.directions = nu.array([{(1,0):DTWDataManager.ROW, 
                                     (1,1):DTWDataManager.DIAG, 
                                     (0,1):DTWDataManager.COL}[tuple(x)]
                                    for x in nu.diff(path,axis=0)])
    def getDirection(self):
        self.i = self.i+1
        return int(round(nu.mean(self.directions[max(0,self.i-self.memory):self.i])))
        
def dtw(s,t,K=None,L=None,cost=None,apath=None,memory=1,returnCost=False):
    """
    Find the optimal alignment of s and t through cost
    Arguments:
    s -- source sequence
    t -- target sequence
    K,L -- the dimensions of the search window
    cost -- the cost function
    
    Return:

    """

    if cost == None:
        cost = costBasic
    M = len(s)
    N = len(t)

    if K == None:
        K = M
    if L == None:
        L = N
    K = min(M,K)
    L = min(N,L)

    dm = DTWDataManager(M,N,K,L)

    # fill up first part
    v = int(nu.ceil(K/2.0))
    w = int(nu.ceil(L/2.0))
    #print('traversing distance matrix')
    for i in range(0,v):
        for j in range(0,w):
            dm.addCost(i,j,cost(s,t,i,j))
    
    # use either the approximate path to guide the search
    # or search greedily
    def getDirGreedy(i,j):
        return dm.getDirection(i,j)

    if apath == None:
        getDir = getDirGreedy
    else:
        nav = Navigator(apath,memory)
        def getDir(i,j):
            return nav.getDirection()
    i,j = (v-1,w-1)
    
    while i < M-1 or j < N-1:
        r = getDir(i,j)
        #rg = getDirGreedy(i,j)
        #print(r,rg)
        # check if a row can be added
        canAddRow = i < M-1
        # check if a col can be added
        canAddCol = j < N-1
        # check whether (when possible) a row should be added
        willAddRow = canAddRow and (r in (DTWDataManager.ROW,DTWDataManager.DIAG) 
                                    or not canAddCol)
        # check whether (when possible) a row should be added
        willAddCol = canAddCol and (r in (DTWDataManager.COL,DTWDataManager.DIAG) 
                                    or not canAddRow)

        if willAddRow:
            i = i + 1
        if willAddCol:
            j = j + 1

        if willAddRow:
            j_start = max(0,j-L)
            dm.setCost(i, j_start, nu.inf)
            if not willAddCol:
                dm.setCost(i-1, j_start, nu.inf)
            for jj in range(j_start+1,j):
                dm.addCost(i,jj,cost(s,t,i,jj))
        if willAddCol:
            i_start = max(0,i-K)
            dm.setCost(i_start, j, nu.inf)
            if not willAddRow:
                dm.setCost(i_start, j-1, nu.inf)
            for ii in range(i_start+1,i):
                dm.addCost(ii,j,cost(s,t,ii,j))

        dm.addCost(i,j,cost(s,t,i,j))

    #print('computing path')
    if returnCost:
        #print(dm.m.getBuffer())
        return dm.getPath(),dm.getCost()
    else:
        return dm.getPath()
  
def smooth(x,k):
    smoothingkernel = (nu.ones(k)*k)**-1
    return nu.convolve(x,smoothingkernel,'same')

def downSample(x,k):
    """Downsample x by smoothing plus linear interpolation
    x -- a Nx2 matrix, 1st column is time, 2nd is value; 
         samples should be approx. equidistant
    k -- downsampling parameter: the result will have N/2**k points
    """
    xsmooth = x.copy()
    for i in range(1,x.shape[1]):
        xsmooth[:,i] = smooth(x[:,i],2**k)
    assert(xsmooth.shape[0] > 1)
    xsmooth[0,1:] = xsmooth[1,1:]
    xsmooth[-1,1:] = xsmooth[-2,1:]
    idx = nu.linspace(xsmooth[0,0],xsmooth[-1,0],int(xsmooth.shape[0]/float(2**k)))
    d = [interpolate.interp1d(xsmooth[:,0],xsmooth[:,x])(idx) for x in range(1,x.shape[1])]
    return nu.column_stack([idx]+d)


def addDerivs(x):
    d = nu.zeros((x.shape[0],x.shape[1]-1),nu.float32)
    d[:-1,:] = nu.diff(x[:,1:],axis=0)
    d[1:,:] = (d[1:,:] + d[:-1,:])/2.
    for i in range(d.shape[1]):
        d[:,i] = smooth(d[:,i],10)
    return nu.column_stack((x,d))

def substituteDerivs(x):
    #d = nu.zeros((x.shape[0],x.shape[1]-1),nu.float32)
    for i in range(1,x.shape[1]):
        x[:-1,i] = nu.diff(x[:,i])
        x[1:,i] = (x[1:,i] + x[:-1,i])/2.
        x[1:,i] = smooth(x[1:,i],5)
    return x

def normalizePath(xx,begin=None,end=None):
    if not begin:
        begin = xx[0,:]
    if not end:
        end = xx[-1,:]
    x = nu.float32(xx.copy())
    x[:,0] = (x[:,0]-begin[0])/(end[0]-begin[0])
    x[:,1] = (x[:,1]-begin[1])/(end[1]-begin[1])
    assert((x >= 0.).all())
    assert((x <= 1.).all())
    if x[0,:].any() > 0.0:
        x = nu.vstack((nu.array([[0.,0.]],), x))
    if x[-1,:].any() < 1.0:
        x = nu.vstack((x,nu.array([[1.,1.]],)))
    return x

def resamplePath(xx,end):
    x = nu.float32(xx.copy())
    x[:,0] = end[0]*x[:,0]/x[-1,0]
    x[:,1] = end[1]*x[:,1]/x[-1,1]
    N = x.shape[0]
    y = [nu.array((0,0),nu.float32)]
    i = 1
    while i < N:
        #print('next x',x[i,:])
        d = x[i,:] - x[i-1,:]
        #print('d',d)
        K = int(nu.sum(d)+2)
        z = [y[-1]]
        #print('start y',z[0])
        for j in range(1,K+1):
            ynew = (nu.float32(y[-1]) + j*nu.float32(d)/K)
            #if not all(ynew == z[-1]):
            z.append(ynew)
        #print('end y',z[-1])
        y.extend(z[1:])
        i = i + 1
    y = nu.array(nu.round(y),nu.uint32)
    y = nu.vstack((y[0,:],y[1:][nu.sum(nu.diff(y,axis=0),axis=1) > 0]))
    # the steps should not be bigger than 1
    assert(all(nu.max(nu.diff(y,axis=0),axis=0)==1))
    return y

def transformBeats(path,beats):
    """Transform beat times through the alignment path in both directions
    """
    nbeats = [(0,0)]
    N = path.shape[0]
    for i,j in beats:
        last_i,last_j = nbeats[-1]
        nbeats.append((min(N-1,nu.searchsorted(path[last_i:,0],i)+last_i),
                       min(N-1,nu.searchsorted(path[last_j:,1],j)+last_j)))
    nbeats = nu.array(nbeats[1:])
    return nu.column_stack((path[nbeats[:,1],0],path[nbeats[:,0],1]))
    
if __name__ == '__main__':
    # calibration values as established by measurement 
    # of sensors
    sensor_offset = 340.
    sensor_scale = 67. 

    W = 10 # window size
    K,L = (W,W) # square window
    memory = 1 # smoothness of guiding path
    sourceFile = sys.argv[1]
    print('source: {0}'.format(sourceFile))
    targetFile = sys.argv[2]
    print('target: {0}'.format(targetFile))
    beatFile = sys.argv[3]
    #conBeatFile = sys.argv[4]
    #print('conBeat: {0}'.format(conBeatFile))
    #rehBeatFile = sys.argv[3]
    #print('rehBeat: {0}'.format(rehBeatFile))
    beats = nu.loadtxt(beatFile)
    ssong = os.path.splitext(os.path.basename(sourceFile))[0]
    tsong = os.path.splitext(os.path.basename(targetFile))[0]
    nr = ssong[-2:]
    print(ssong)
    s = nu.loadtxt(sourceFile)
    t = nu.loadtxt(targetFile)
    # fix timestamp problems 
    # (evenly space samples over total time range)
    #s[:,0] = nu.linspace(s[0,0],s[-1,0],s.shape[0]) 
    #t[:,0] = nu.linspace(t[0,0],t[-1,0],t.shape[0]) 
    # Convert ADC values to g-values
    s[:,1:] = (s[:,1:]-sensor_offset)/sensor_scale 
    t[:,1:] = (t[:,1:]-sensor_offset)/sensor_scale 
    # Subtract mean to get rid of earth-gravity
    #s[:,1:] = s[:,1:]-nu.mean(s[:,1:],axis=0)
    #t[:,1:] = t[:,1:]-nu.mean(t[:,1:],axis=0)
    
    #s = substituteDerivs(s)
    #t = substituteDerivs(t)
    #s = addDerivs(s)
    #t = addDerivs(t)
    for i in range(1,s.shape[1]):
        s[:,i] = s[:,i]-nu.mean(s[:,i])
        t[:,i] = t[:,i]-nu.mean(t[:,i])
        #s[:,i] = (s[:,i]-nu.mean(s[:,i]))/nu.std(s[:,i])
        #t[:,i] = (t[:,i]-nu.mean(t[:,i]))/nu.std(t[:,i])
    #p = None
    #annotPath = nu.column_stack((cbf[:,0],rbf[:,0]))
    #begin = (nu.min((s[0,0],annotPath[0,0])),
    #         nu.min((t[0,0],annotPath[0,1])))
    #end = (nu.max((s[-1,0],annotPath[-1,0])),
    #       nu.max((t[-1,0],annotPath[-1,1])))
    p = None#normalizePath(annotPath,begin,end)
    
    # determine appropriate downsampling for given input path
    # assume more or less equal lengths of s and t
    N = (s.shape[0]+s.shape[0])/2.
    #B = annotPath.shape[0]
    maxDownSampleFactor = 10#nu.floor(nu.log2(N/B))
    minDownSampleFactor = 2.5
    #p = annotPath#resamplePath(annotPath,(s.shape[0]-1,t.shape[0]-1))
    #for k in (2.5,2.4,2.3,2.2,2.1,2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0):
    # for k in (1.3,1.2,1.1,1.0):

    #for k in nu.linspace(10,1,91):
    for k in nu.linspace(maxDownSampleFactor,minDownSampleFactor,1+(maxDownSampleFactor-minDownSampleFactor)/.5):
        parameterDir = '/tmp/{0}_{1}_{2:.2f}/'.format(W,memory,k)
        try:
            os.makedirs(parameterDir)
            print(parameterDir)
        except Exception as e:
            pass
        #s_d = addDerivs(downSample(s,k))
        #t_d = addDerivs(downSample(t,k))
        s_d = downSample(s,k)
        t_d = downSample(t,k)
        print(k)
        if p is None:
            p = dtw(s_d[:,1:],t_d[:,1:],K,L,cost)
            # s_index t_index s_time t_time s_value t_value
        else:
            p1 = resamplePath(normalizePath(p),(s_d.shape[0]-1,t_d.shape[0]-1))
            p = dtw(s_d[:,1:],t_d[:,1:],K,L,cost,p1,memory)
        path = nu.column_stack((p[:,0],p[:,1],
                                    s_d[p[:,0],0],
                                    t_d[p[:,1],0],
                                    s_d[p[:,0],1:],
                                    t_d[p[:,1],1:]))
        nu.savetxt('{0}/path_{1}.txt'.format(parameterDir,nr),path)
        tbeats = transformBeats(path[:,(2,3)],beats)
        #nu.savetxt('/tmp/out.txt',nu.mean(nu.mean(nu.abs(tbeats-beats),axis=1)),fmt='%.3f')
        #nu.savetxt('{0}/meanBeatGap_{1}.txt'.format(parameterDir,nr),nu.mean(nu.mean(nu.abs(tbeats-beats),axis=1)))
        with open('{0}/meanBeatGap_{1}.txt'.format(parameterDir,nr),'w') as f:
            f.write('{0:f} {1:f}\n'.format(k,nu.mean(nu.mean(nu.abs(tbeats-beats),axis=1))))
        nu.savetxt('{0}/{1}.txt'.format(parameterDir,ssong),s_d)
        nu.savetxt('{0}/{1}.txt'.format(parameterDir,tsong),t_d)
