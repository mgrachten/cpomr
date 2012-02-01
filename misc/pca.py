#!/usr/bin/env python
import numpy
    
def wcov(X,w):
    """Return the covariance of matrix X (M by N),
    weighted by vector w (N);
    w is normalized by dividing by sum(w)
    """
    wn = w/float(numpy.sum(w))
    W = numpy.diag(wn)
    wmeans = numpy.sum(numpy.dot(X,W),axis=1)
    Z = numpy.subtract(numpy.transpose(X),wmeans)
    return numpy.dot(numpy.dot(numpy.transpose(Z),W),Z)/(1.-numpy.sum(wn**2))

def wmean(X,w):
    wn = w/float(numpy.sum(w))
    return numpy.dot(X,wn)

def pca(X,v=None):
    """Principal Component Analysis of
    matrix X (instances are in the columns)
    Return values:
    pc: principal components of X (in the columns)
    z:  transformed X data (X projected on the pc dimensions)
    D:  eigenvalues of cov(X)
    """
    if v is None:
        C = numpy.cov(X)
    else:
        C = wcov(X,v)
    U,D,pc = numpy.linalg.svd(C,full_matrices=1,compute_uv=1)
    pc = numpy.transpose(pc)
    z = numpy.transpose(numpy.dot(numpy.transpose(X)-numpy.mean(numpy.transpose(X),axis=0),pc))
    return (pc,z,D)
    # w = numpy.diag(D)


if __name__ == '__main__':
    X = X = numpy.random.normal(0,3,(100,2))
    v = numpy.ones(X.shape[1],numpy.double)
    for i in pca(X,v):
        print(i)
    print()
    for i in pca(X):
        print(i)


