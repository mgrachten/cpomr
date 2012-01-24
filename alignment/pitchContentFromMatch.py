#!/usr/bin/env python

import sys,os
import numpy as nu
from operator import attrgetter

from matchUtils import MatchFile,SnoteNoteLine
#from utilities import argpartition
from utilities import partition

def groupBarPitches(snotes):
    snoteGroups = partition(attrgetter('OnsetInBeats'),snotes)
    onsets = snoteGroups.keys()
    onsets.sort()
    data = []
    for i,x in enumerate(onsets):
        data.extend([(i,y.MidiPitch[0]) for y in snoteGroups[x]])
    return nu.array(data)

def getPitchesPerBar(match):
    snotes = [x.snote for x in match.lines if hasattr(x,'snote') and not 'grace' in x.snote.ScoreAttributesList ]
    barnotes = partition(attrgetter('Bar'),snotes)
    bars = barnotes.keys()
    bars.sort()
    d = nu.empty((0,3),nu.int)
    for b in bars:
        bn = groupBarPitches(barnotes[b])
        d = nu.vstack((d,nu.column_stack((nu.zeros(bn.shape[0],nu.int)+b,bn))))
    return d

if __name__ == '__main__':
    mfn = sys.argv[1]
    match = MatchFile(mfn) 
    nu.savetxt(os.path.join('/tmp/',os.path.splitext(os.path.basename(mfn))[0]+'.txt'),
               getPitchesPerBar(match),fmt='%d')
