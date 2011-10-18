#!/usr/bin/env python

import os
from imageUtil import getImageData,normalize,makeMask,getPattern,getImageAndMask
from multiprocessing import Lock
import numpy as nu

def colorString2Tuple(s):
    assert len(s) == 7
    return (int(s[1:3],16),int(s[3:5],16),int(s[5:7],16))

class VocabularyItem(object):
    def __init__(self,l,dirname):
        parts = l.strip().split()
        self.label = parts[0]
        self.color = colorString2Tuple(parts[1])
        self.thresholds = [float(x) for x in parts[2::2]]
        self.files = [os.path.join(dirname,x) for x in parts[3::2]]
        self.images = [None for f in self.files]
        self.pureimages = [None for f in self.files]
        self.masks = [None for f in self.files]
        self.minVals = [None for f in self.files]
        self.maxVals = [None for f in self.files]
        #self.loadLock = Lock()

    def getColor(self):
        return self.color
    def setCalibration(self,i,v0,v1):
        self.minVals[i] = v0
        self.maxVals[i] = v1

    def normalize(self,i,array,binary=False,threshold=1):
        """Normalize a data array according to the calibration values of the i-th image
        A value > 1 should imply a match
        """
        assert self.minVals[i] != None
        assert self.maxVals[i] != None
        r = (array-self.minVals[i])/(self.maxVals[i]-self.minVals[i])
        if binary:
            return r >= threshold
        else:
            return r

    def getThresholds(self):
        return self.thresholds

    def getFiles(self):
        return self.files

    def getThreshold(self,i=0):
        assert len(self.thresholds)>i
        return self.thresholds[i]

    def getFile(self,i=0):
        assert len(self.files)>i
        return self.files[i]

    def getImage(self,i=0):
        assert len(self.images)>i
        #self.loadLock.acquire()
        if self.images[i] == None:
            #self.images[i] = normalize(getImageData(self.files[i]))-.5
            #self.images[i] = 255-nu.array(getImageData(self.files[i]),nu.int8)-128
            self.images[i] = getPattern(self.files[i],True,True)
            self.pureimages[i],self.masks[i] = getImageAndMask(self.files[i],True,True)
            #self.images[i] = nu.array(nu.round(makeMask(self.images[i])*self.images[i]),nu.int8)
        #self.loadLock.release()
        return self.images[i]

    def __str__(self):
        return '{0} {1}'.format(self.label,self.files)

class Vocabulary(object):
    def __init__(self):
        self.vocItems = {}
    def addItem(self,vi):
        self.vocItems[vi.label] = vi
    def getItem(self,label):
        return self.vocItems.get(label,None)

class ScoreVocabulary(Vocabulary):
    def __init__(self):
        super(ScoreVocabulary,self).__init__()
        self.bar = None
    def addItem(self,vi):
        if vi.label == 'bar':
            self.bar = vi
        else:
            self.vocItems[vi.label] = vi
    def getItem(self,label):
        return self.vocItems.get(label,None)
    def getLabels(self):
        return self.vocItems.keys()
    def getBar(self):
        return self.bar

def makeVocabulary(vocabularyDir):
    vocfile = os.path.join(vocabularyDir,'vocabulary.txt')
    vocabulary = ScoreVocabulary()
    with open(vocfile,'r') as f:
        for l in f.readlines():
            if l[0] is not '#' and len(l.strip()) > 0:
                vocabulary.addItem(VocabularyItem(l,vocabularyDir))
    return vocabulary

if __name__ == '__main__':
    pass
