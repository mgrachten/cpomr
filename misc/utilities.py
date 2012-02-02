#!/usr/bin/env python

#    Copyright 2012, Maarten Grachten.
#
#    This file is part of mg_python_modules.
#
#    mg_python_modules is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mg_python_modules is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with mg_python_modules.  If not, see <http://www.gnu.org/licenses/>.

import re,sys,os
from numpy import array,size,std
import numpy as nu
import subprocess
from math import ceil
from functools import update_wrapper, wraps
import itertools

class MissingModuleError(Exception):
    def __init__(self,modulename):
        Exception.__init__(self)
        self.modulename = modulename
    def __str__(self):
        return 'The function you are trying to use requires module "{0}", which is not found/installed.'.format(self.modulename)


try:
    import simplejson

    def readJSON(fn):
        try:
            with open(fn,'r') as f:
                result = simplejson.load(f)
        except:
            sys.stderr.write('Error, cannot read configuration from {0}\n'.format(fn))

except ImportError:
    sys.stderr.write('Warning: could not load module simplejson, some functions will not be available\n')

    def readJSON(*args):
        raise MissingModuleError('simplejson')

try:
    from scipy import interpolate
except:
    sys.stderr.write('Warning, could not import scipy not all functionality is available')


def cachedProperty (func ,name =None ):
    """
    cachedProperty(func, name=None) -> a descriptor
    This decorator implements an object's property which is computed
    the first time it is accessed, and which value is then stored in
    the object's __dict__ for later use. If the attribute is deleted,
    the value will be recomputed the next time it is accessed.

    Usage:

    class X(object):
        @cachedProperty
        def foo(self):
            return computation()
    """
    if name is None :
        name = func.__name__

    @wraps(func)
    def _get (self ):
        try :
            return self.__dict__[name]
        except KeyError :
            self.__dict__[name] = func(self)
            return self.__dict__[name]

    @wraps(func)
    def _set (self,value):
        self.__dict__[name] = value

    @wraps(func)
    def _del (self ):
        self.__dict__.pop(name,None)

    #update_wrapper(_get, func)
    #update_wrapper(_set, func)
    #update_wrapper(_del, func)
    return property(_get, _set, _del)

def getter(f):
    "Decorator for object methods, Stores the return value inside the object for later calls"
    def _getter(self):
        if not hasattr(self,'valuedict'):
            self.valuedict = {}
        if not self.valuedict.has_key(f):
            self.valuedict[f] = f(self)
        return self.valuedict[f]
    return _getter


def itersubclasses(cls, _seen=None):
    """
    itersubclasses(cls)

    Generator over all subclasses of a given class, in depth first order.

    >>> list(itersubclasses(int)) == [bool]
    True
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>> 
    >>> for cls in itersubclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL (new-style) classes currently defined
    >>> [cls.__name__ for cls in itersubclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """
    
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub

def rescale(v,xmin,xmax):
    vmin = nu.min(v)
    vmax = nu.max(v)
    
    if vmax > vmin:
        v = xmin+(xmax-xmin)*(v-vmin)/(vmax-vmin)
    return v

class VerbosityPrinter:
    def __init__(self,level=5,out=sys.stdout,flushOnWrite=False):
        self.level = level
        self.out = out
        self.flushOnWrite = flushOnWrite
    def setLevel(self,l):
        self.level = l
    def write(self,msg,verbosity=10):
        if verbosity >= self.level:
            self.out.write(msg)
        if self.flushOnWrite: 
            self.out.flush()
    def writeln(self,msg,verbosity=10):
        self.write(msg+'\n',verbosity)

def indir(dirname,f):
    return os.path.join(dirname,f)

class Result(object):
    def __init__(self,v):
        self.v = v
    def get(self):
        return self.v

class FakePool(object):
    def __init__(self):
        pass
    def map(self,f,a):
        return map(f,a)
    def apply_async(self,f,args=(),kwargs={},callback=None):
        if callback:
            callback(f(*args,**kwargs))
        else:
            return Result(f(*args,**kwargs))
    def close(self):
        pass
    def terminate(self):
        pass
    def join(self):
        pass

# def interpolateMap(idx,xy,defVal=0.0):
#     assert xy.shape[0] > 0
#     assert len(idx) > 0
    
#     if xy[0,0] > idx[0]:
#         xy = nu.vstack((nu.array([[idx[0],defVal]]),xy))
#     if xy[-1,0] < idx[-1]:
#         xy = nu.vstack((xy,nu.array([[idx[-1],defVal]])))

#     f = interpolate.interp1d(xy[:,0],xy[:,1])
#     return f(idx)

def interpolateMap(idx,xy):
    assert xy.shape[0] > 0
    assert len(idx) > 0
    vmin = nu.min(idx)
    vmax = nu.max(idx)
    
    if xy[0,0] > vmin:
        xy = nu.vstack((nu.array([[vmin,xy[0,1]]]),xy))
    if xy[-1,0] < vmax:
        xy = nu.vstack((xy,nu.array([[vmax,xy[-1,1]]])))
    f = interpolate.interp1d(xy[:,0],xy[:,1])
    return f(idx)

def makeColors(n,noGrey=True):
    """Return a list of at least n colors (i.e. triples <i,j,k>,
    where 0 <= i,j,k <=255), ordered from saturated to non-saturated.
    If noGrey = True, no grey tones are returned"""
    k = int(ceil(n**(1.0/3)))
    if noGrey and n > (k**3-k):
        k += 1
    basis = array(range(k))*(1.0/(k-1))
    #print(basis)
    #combinations = array([array(i) for i in cartesianProduct(basis,basis,basis)])
    combinations = array([array(i) for i in itertools.product(basis,basis,basis)])
    stddevs = array([std(i) for i in combinations])
    return [(255.0*i).astype(int) for i in combinations[stddevs.argsort()][::-1]]

def powerset(seq):
    if len(seq):
        head = powerset(seq[:-1])
        return head + [item + [seq[-1]] for item in head]
    else:
        return [[]]

def perm(items, n=None):
    """Return all permutations of items."""
    if n is None:
        n = len(items)
    for i in range(len(items)):
        v = items[i:i+1]
        if n == 1:
            yield v
        else:
            rest = items[:i] + items[i+1:]
            for p in perm(rest, n-1):
                yield v + p

def getOutputFromCommand(cmd):
    """Simple wrapper around popen2, to get output from a shell command"""
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                         stdout=subprocess.PIPE, close_fds=True)
    p.stdin.close()
    result = p.stdout.readlines()
    p.stdout.close()
    return result

def partition(func, iterable):
    """Returns a dict result where result[func(i)] = { i | func(i) }"""
    result = {}
    for i in iterable:
        result.setdefault(func(i), []).append(i) 
    return result

def argpartition(func, iterable):
    """Returns a dict result where result[func(i)] = { i | func(i) }"""
    result = {}
    for i,v in enumerate(iterable):
        result.setdefault(func(v), []).append(i)
    return result

## General utility functions for handling data files

def writeFile(filename,data):
    """Write data to filename. If data is a list it writes
    the elements line by line, otherwise the displayed
    object is written. On objects that are not str, int, or
    float, the value of repr(object) is written."""
    if not type(data) == list:
        data = [data]
    f = open(filename,'w')
    for i in data:
        if type(i) in [str, int, float]:
            f.write(i)
        else:
            if '__repr__' in dir(i):
                f.write(repr(i))
            else:
                print("Warning: don't know how to write object of type %s" % type(i))
    f.close()
    return True

def readFile(filename,join=False,strip=False,ignore='',split=False):
    """Read data from a file and return the lines as a list of strings
    (join=False) or as a single string (join=True), possibly stripping
    whitespace characters from begin and end of lines (strip=True).
    Lines starting with any character in `ignore', are....ignored."""
    f = open(filename,'r')
    data = f.readlines()
    f.close()
    return __processStringSeq(data,join,strip,ignore,split)

def __processStringSeq(data,join=False,strip=False,ignore='',split=False):
    """Post-process a list of strings according to arguments specified
    return the lines as a list of strings (join=False) or as a single
    string (join=True), possibly stripping whitespace characters from
    begin and end of lines (strip=True).  Lines starting with any
    character in `ignore', are....ignored."""
    if len(ignore) > 0:
        data = filter(lambda l: l[0] not in ignore, data)
    if strip:
        data = [l.strip() for l in data]
    if join:
        data = ''.join(data)
    elif split:
        data = [l.split() for l in data]
    return data

def readFromStdin(join=False,strip=False,ignore=''):
    """Read data from stdin and return the lines as a list of strings
    (join=False) or as a single string (join=True), possibly stripping
    whitespace characters from begin and end of lines (strip=True).
    Lines starting with any character in `ignore', are....ignored."""
    data = []
    line = sys.stdin.readline()
    while (line):
        data.append(line)
        line = sys.stdin.readline()
    return __processStringSeq(data,join,strip,ignore)

def interpretField(data):
    """Convert data to int, if not possible, to float,
    if not possible return data itself."""
    try:
        return int(data)
    except ValueError:
        try:
            return float(data)
        except ValueError:
            return data

rationalPattern = re.compile('^([0-9]+)/([0-9]+)$')

class ParseRationalException(Exception):
    def __init__(self,string):
        self.string = string
    def __str__(self):
        return 'Could not parse string "{0}"'.format(self.string)
        
    
class Ratio:
    def __init__(self,string):
        try:
            self.numerator,self.denominator = [int(i) for i in string.split('/')]
        except:
            raise ParseRationalException(string)
    def getNumerator(self):
        return self.numerator
    def getDenominator(self):
        return self.denominator

def interpretFieldRationalClass(data,allowAdditions=False):
    """Convert data to int, if not possible, to float, if not possible
    try to interpret as rational number and return it as float, if not
    possible, return data itself."""
    global rationalPattern
    v = interpretField(data)
    if type(v) == str:
        m = rationalPattern.match(v)
        if m:
            #groups = m.groups()
            #return float(groups[0])/float(groups[1])
            return Ratio(v)
        else:
            if allowAdditions:
                parts = v.split('+')
                if len(parts) > 1:
                    iparts = [interpretFieldRational(i,allowAdditions=False) for i in parts]
                    # to be replaced with isinstance(i,numbers.Number)
                    if all(type(i) in (int,float) for i in iparts):
                        return sum(iparts)
                    else:
                        return v
                else:
                    return v
            else:
                return v
    else:
        return v

def interpretFieldRational(data,allowAdditions=False):
    """Convert data to int, if not possible, to float, if not possible
    try to interpret as rational number and return it as float, if not
    possible, return data itself."""
    global rationalPattern
    v = interpretField(data)
    if type(v) == str:
        m = rationalPattern.match(v)
        if m:
            groups = m.groups()
            return float(groups[0])/float(groups[1])
        else:
            if allowAdditions:
                parts = v.split('+')
                if len(parts) > 1:
                    iparts = [interpretFieldRational(i,allowAdditions=False) for i in parts]
                    # to be replaced with isinstance(i,numbers.Number)
                    if all(type(i) in (int,float) for i in iparts):
                        return sum(iparts)
                    else:
                        return v
                else:
                    return v
            else:
                return v
    else:
        return v

def readDataFile(filename,separator=None,ignore='@#%;\n',output='array',fieldInterpreter=interpretField):
    return readData(readFile(filename,ignore=ignore),separator,output,fieldInterpreter)

def readData(data,separator=None,output='array',fieldInterpreter=interpretField):
    """Read a data file, and return the data as a (numpy) array.
    It uses `separator' to split fields. `fieldInterpreter' is
    used to interpret each field. By default, wherever possible,
    fields are converted to int, or float. Lines starting with
    any character in `ignore', are....ignored.

    Beware that when reading in heterogenous data (combinations
    of strings, integers, floats), some values may be truncated
    by the array constructor. Use 'list' output instead.
    """
    if not data:
        return None
    if fieldInterpreter:
        dataM = [ [interpretField(i) for i in line.strip().split(separator)] 
                  for line in data if len(line.strip()) > 0]
    else:
        dataM = [ line.strip().split(separator) for line in data if len(line.strip()) > 0]
    if output == 'array':
        return array(dataM)
    elif output == 'list':
        return dataM
    else:
        print('unknown output type specified: %s' % output)
        return None


def writeDataFile(filename,data,separator=' '):
    """Write `data' (2D array) to `filename', using `separator'
    to delimit fields. 
    """
    if not type(data) == type(array([])):
        print('Error: data should be of type %s' % type(array([])))
        return None
    f = open(filename,'w')
    m = data.shape
    if len(m) == 1:
        M = m[0]
        for i in range(M):
            f.write('%s\n' % (data[i]))
    else:
        if len(m) > 2:
            print('using only first two dimensions')
        M = m[0]
        N = m[1]
        for i in range(M):
            l = []
            for j in range(N):
                l.append('%s' % (data[i,j]))
            f.write(separator.join(l)+'\n')
    ##for row in data:
    ##    print(row)
    ##    row.tofile(f,sep=separator,format='%f')
    ##    f.write('\n')
    f.close()
    print(filename)

