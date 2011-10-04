#!/usr/bin/env python

import os
from imageUtil import getImageData,normalize,makeMask
from multiprocessing import Lock
import numpy as nu

class VocabularyItem(object):
    def __init__(self,l,dirname):
        parts = l.strip().split()
        self.label = parts[0]
        self.thresholds = [float(x) for x in parts[1::2]]
        self.files = [os.path.join(dirname,x) for x in parts[2::2]]
        self.images = [None for f in self.files]
        #self.loadLock = Lock()
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
            self.images[i] = 255-nu.array(getImageData(self.files[i]),nu.int8)-128
            self.images[i] = nu.array(nu.round(makeMask(self.images[i])*self.images[i]),nu.int8)
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
            if l[0] is not '#':
                vocabulary.addItem(VocabularyItem(l,vocabularyDir))
    return vocabulary

if __name__ == '__main__':
    pass
