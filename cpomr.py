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
        self.parser.add_argument('filenames', metavar='FILENAME', type=str, nargs='+',
                                 help='Image filename; multiple filenames will be treated as ' \
                                     'the consecutive pages of a single piece')
        self.parser.add_argument('--output-dir','-o', metavar='OUTPUTDIR', 
                                 type=str,default='/tmp/',
                                 help='Write output to directory; directory will be created if ' \
                                     'it does not exist (default: %(default)s)',
                                 dest='outputDir')
        self.parser.add_argument('--draw-annotations','-d',action='store_true', 
                                 dest='draw',default=False,
                                 help='Draw annotated scores (default: %(default)s); ' \
                                     'output will be in png format, ' \
                                     'named after input image, and stored in OUTPUTDIR')
        self.parser.add_argument('--write-bar-coordinates','-b',action='store_true', 
                                 dest='barCoordinates', default=False,
                                 help='Write bar bounding box coordinates to a text file ' \
                                     '(default: %(default)s);  output ' \
                                     'will be in txt format, and stored in OUTPUTDIR')
        self.args = self.parser.parse_args()
        self.canWrite = False
        self.draw = self.args.draw
        self.outputDir = self.args.outputDir
        self.barCoordinates = self.args.barCoordinates
        self.filenames = self.args.filenames

if __name__ == '__main__':
    clh = CommandLineHandler()
    log = logging.getLogger(__name__)
    clh.canWrite = False
    if clh.outputDir:
        try:
            os.makedirs(clh.outputDir)
        except OSError as e:
            if e.errno != 17:
                raise e
        try:
            assert os.access(clh.outputDir,os.W_OK)
            clh.canWrite = True
        except:
            log.error('Can not write to output directory {0}'.format(clh.outputDir))

    piece = Piece(clh.filenames)
    if clh.draw:
        if clh.canWrite:
            piece.drawAnnotatedScores(clh.outputDir)
        else:
            log.warn('Will not draw annotated scores (output directory not writeable)')
    if clh.barCoordinates:
        if clh.canWrite:
            piece.writeBarCoordinates(clh.outputDir)
        else:
            log.warn('Will not write bar coordinates (output directory not writeable)')

