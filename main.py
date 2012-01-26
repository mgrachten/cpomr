#!/usr/bin/env python

import sys,os, logging
import numpy as nu
from OMR.scoreImage import ScoreImage

logging.basicConfig(format='%(levelname)s: [%(name)s] %(message)s',level=logging.INFO)

if __name__ == '__main__':
    fn = sys.argv[1]
    si = ScoreImage(fn)
    #si.drawImage()
    for b in si.bars:
        pass#print(b.getBBs())

