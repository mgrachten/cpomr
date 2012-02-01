#!/usr/bin/env python

import sys,os

## make python find our local modules
RENCON11PATH = os.getenv('RENCON11PATH')
if RENCON11PATH == None:
    sys.stderr.write('Error: couldn\'t get value of RENCON11PATH from shell environment\n')
    sys.exit(1)

sys.path.insert(0,os.path.join(RENCON11PATH,'library/python_modules'))

from utilities import partition

class FilenameGroup(object):
    def __init__(self,fgroup,base=''):
        self.base = base
        if len(fgroup) > 0:
            self.prefix = fgroup[0].prefix
            self.separator = fgroup[0].separator
            self.files = dict([(h.usuffix,h.filename) for h in fgroup])
        else:
            self.prefix = None
            self.separator = None
            self.files = None

class Filename(object):
    def __init__(self,f,separator):
        self.filename = f
        self.basename = os.path.splitext(os.path.basename(f))[0]
        self.separator = separator
        self.parts = self.basename.split(self.separator)
    def getLength(self):
        return len(self.parts)
    def getPart(self,i):
        assert(isinstance(i,int))
        try:
            return self.parts[i]
        except IndexError:
            return False
    def getParts(self):
        return self.parts
    def setUniqueSuffix(self,i):
        self.usuffix = self.separator.join(self.parts[i:])
        self.prefix = self.separator.join(self.parts[:i])

def groupFiles(files,base=None,prefix='',separator='_',ext='.feat'):
    """Group files by their longest prefix 
    """
    #filenames = [Filename(f,'_') for f in files if f.endswith('.feat') and f.startswith(prefix)]
    filenames = [Filename(f,separator) for f in files 
                 if f.endswith(ext) and 
                 f.startswith(prefix)]
    files = _groupFiles(filenames,0)
    return [FilenameGroup(g,base) for g in files]


def _groupFiles(files,i):
    p = partition(lambda x: x.getPart(i),files)
    if len(p.values())==len(files) or any([f.getLength()-1 <= i for f in files]):
        for f in files:
            f.setUniqueSuffix(i)
        return [files]
    else:
        acc = []
        for q in p.values():
            acc.extend(_groupFiles(q,i+1))
        return acc

################

class MatchExpPair:
    def __init__(self,name=False,match=False,exp=False):
        self.name = name
        self.match = match
        self.exp = exp
    def isComplete(self):
        return self.match and self.exp

def pairFiles(expDir,matchDir):
    """
    Pair the match and exp files belonging to the same piece.
    """
    expfiles = os.listdir(expDir)
    matchfiles = os.listdir(matchDir)
    pairDict = {}

    for f in expfiles:
        b,ext = os.path.splitext(os.path.basename(f))
        if ext == '.exp':
            pairDict[b] = MatchExpPair(name=b,exp=os.path.join(expDir,f))

    for f in matchfiles:
        b,ext = os.path.splitext(os.path.basename(f))
        if ext in ('.match','.score'):
            p = pairDict.get(b,MatchExpPair(name=b))
            p.match = os.path.join(matchDir,f)
            pairDict[b] = p

    return [x for x in pairDict.values() if x.isComplete()]



def pairFilesNew(dirDict):
    """
    Pair files
    """
    for k,directory in dirDict.items():
        os.listdir(directory)
    ## not finished yet


if __name__ == '__main__':
    pass
