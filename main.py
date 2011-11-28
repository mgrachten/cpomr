#!/usr/bin/env python

import sys,os
import numpy as nu
from scoreImage import ScoreImage

if __name__ == '__main__':
    fn = sys.argv[1]
    si = ScoreImage(fn)
    si.drawImage()

