#!/usr/bin/env python

import sys,os, logging
import numpy as nu
import pickle
from OMR.scoreImage import ScoreImage
from OMR.piece import Piece


logging.basicConfig(format='%(levelname)s: [%(name)s] %(message)s',level=logging.INFO)

if False: #__name__ == '__main__':
    fns = sys.argv[1:]
    Piece(fns)

if __name__ == '__main__':
    fns = sys.argv[1:]
    si = ScoreImage(fns[0])
    si.drawImage()
    #with open('/tmp/img.dat','w') as f:
    #    pickle.dump(si,f)
    #si.drawCorrectedImage()
    #for b in si.bars:
    #    pass#print(b.getBBs())

