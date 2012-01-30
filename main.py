#!/usr/bin/env python

import sys, os, logging, argparse
import numpy as nu
import pickle
from OMR.scoreImage import ScoreImage
from OMR.piece import Piece


logging.basicConfig(format='%(levelname)s: [%(name)s] %(message)s',level=logging.INFO)

class CommandLineHandler(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Detect music notation in sheet music')
        # options:
        # * draw annotated images
        # * write coordinates to text file
        self.parser.add_argument('filenames', metavar='FILENAME', type=str, nargs='+',
                                 help='Image filename; multiple filenames will be treated as ' \
                                     'the consecutive pages of a single piece')
        self.parser.add_argument('--output-dir','-o', metavar='OUTPUTDIR', type=str,nargs=1,default='/tmp/',
                                 help='Write output to directory; directory will be created if ' \
                                     'it does not exist (default: %(default)s)',
                                 dest='outputDir')
        self.parser.add_argument('--draw-annotations','-d',action='store_true', dest='draw',default=False,
                                 help='Draw annotated scores; output will be in png format, ' \
                                     'named after input image, and stored in OUTPUTDIR')
        self.parser.add_argument('--write-bar-coordinates','-b',action='store_true', dest='barCoordinates', default=False,
                                 help='Write bar bounding box coordinates to a text file;  output ' \
                                 'will be in txt format, and stored in OUTPUTDIR')
        self.args = self.parser.parse_args()
        self.draw = self.args.draw
        self.outputDir = self.args.outputDir
        self.barCoordinates = self.args.barCoordinates
        self.filenames = self.args.filenames

if __name__ == '__main__':
    clh = CommandLineHandler()
    #imageFilenames = sys.argv[1:]
    piece = Piece(clh.filenames)
    if clh.barCoordinates:
        piece.writeBarCoordinates(clh.outputDir)
    if clh.draw:
        piece.drawAnnotatedScores(clh.outputDir)

if False: #__name__ == '__main__':
    fns = sys.argv[1:]
    si = ScoreImage(fns[0])
    si.drawImage()
