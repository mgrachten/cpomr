#!/usr/bin/env python

import sys,os, pickle, logging
import bar
import numpy as nu
from utilities import cachedProperty
from multiprocessing import Pool
from utilities import FakePool
from scoreImage import ScoreImage

# all this KeyboardInterrupt stuff is a workaround of bug
# http://bugs.python.org/issue8296

class KeyboardInterruptError(Exception): pass

def processPage(imgFile):
    try:
        si = ScoreImage(imgFile)
        si.bars
        return si
    except KeyboardInterrupt:
        raise KeyboardInterruptError()

pool = Pool()

class Piece(object):
    def __init__(self,imgFiles):
        self.imgFiles = imgFiles
    
    @cachedProperty
    def imgs(self):
        log = logging.getLogger(__name__)
        imgs = []
        try:
            imgs = pool.map(processPage,self.imgFiles)
        except KeyboardInterrupt:
            log.info('Got ^C while pool mapping, terminating the pool')
            pool.terminate()
            log.info('Pool is terminated')
            log.info('Joining pool processes')
            pool.close()
            pool.join()
            log.info('Join complete')
        except Exception, e:
            log.info('Got exception: %r, terminating the pool' % (e,))
            pool.terminate()
            log.info('Pool is terminated')
            log.info('Joining pool processes')
            pool.close()
            pool.join()
            log.info('Join complete')
        return imgs

    def drawAnnotatedScores(self,outputDir):
        bar_i = 1
        for img in self.imgs:
            # this draws the annotations on the internally
            # stored image
            img.drawAnnotatedScore(bar_i)
            # this writes the internally stored image to a file
            img.ap.writeImage(img.filenameBase+'.png')
            bar_i += len(img.bars)

    def writeBarCoordinates(self,outputDir):
        bar_i = 0
        bb = []
        for j,img in enumerate(self.imgs):
            bb.extend([[j,bar_i+i]+list(bar.boundingBoxes) for i,bar in enumerate(img.bars)])
            bar_i += len(img.bars)
        img.filenameBase
        return bb
    
if __name__ == '__main__':
    pass
