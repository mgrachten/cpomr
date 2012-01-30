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
        log = logging.getLogger(__name__)
        try:
            self.imgs = pool.map(processPage,self.imgFiles)
        except KeyboardInterrupt:
            log.info('Got ^C while pool mapping, terminating the pool')
            pool.terminate()
            log.info('Pool is terminated')
        except Exception, e:
            log.info('Got exception: %r, terminating the pool' % (e,))
            pool.terminate()
            log.info('Pool is terminated')
        finally:
            log.info('Joining pool processes')
            pool.join()
            log.info('Join complete')
    
    def drawAnnotatedScores(self,outputDir):
        bar_i = 0
        for img in self.imgs:
            img.drawAnnotatedScore(bar_i)
            img.ap.writeImage(img.fn)
            bar_i += len(img.bars)

if __name__ == '__main__':
    pass
