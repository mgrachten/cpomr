#!/usr/bin/env python

"""
This module is for accessing the data in match files accessible
from within python. It has only read capabilities, and is not
functionally complete in any way. But with the basics here provided
it should be relatively easy to extend the module to get the data
you want.
"""

import os,sys,re,math,operator
import numpy as nu
import utilities

def pitchName2Midi_PC(modifier,name,octave):
    if name == 'r':
        return 0,0
    baseClass = {'c':0,'d':2,'e':4,'f':5,'g':7,'a':9,'b':11}[name.lower()] + {'b':-1,'bb':-2,'#':1,'##':2,'n':0}[modifier]
    mid = (octave+1)* 12 + baseClass
    #for mozartmatch files (in which the octave numbers are off by one)
    #mid = octave*12 + baseClass
    pitchclass = baseClass%12
    return mid,pitchclass

################### NEW ################### NEW ################### NEW

class MatchLine:
    """A class that represents a line in a match file. It is intended
    to be subclassed. It's constructor sets up a list of field names
    as object attributes.
    """
    fieldNames = []
    def __init__(self,matchObj,fieldInterpreter=utilities.interpretFieldRational):
        self.setAttributes(matchObj,fieldInterpreter)
    def setAttributes(self,matchObj,fieldInterpreter=lambda x: x):
        """Set attribute objects using values from matchObj"""
        groups = [fieldInterpreter(i) for i in matchObj.groups()]
        if len(self.fieldNames) == len(groups):
            for (a,v) in zip(self.fieldNames,groups):
                setattr(self,a,v)

class UnknownMatchLine(MatchLine):
    def __init__(self,line):
        self.line = line
    
class Note(MatchLine):
    """Class representing the played note part of a match line."""
    fieldNames = ['Number','NoteName','Modifier','Octave',
                  'Onset','Offset','AdjOffset','Velocity' ]
    pattern = 'note\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)'
    reObj = re.compile(pattern)
    def __init__(self,m):
        MatchLine.__init__(self,m)
        self.MidiPitch = pitchName2Midi_PC(self.Modifier,self.NoteName,self.Octave)

class TrailingNoteLine(MatchLine):
    """Class representing a Trailing Note line."""
    fieldNames = ['Number','NoteName','Modifier','Octave',
                  'Onset','Offset','AdjOffset','Velocity' ]
    pattern = 'note\((.+),\[(.+),(.+)\],(.+),(.+),(.+),(.+),(.+)\)'
    reObj = re.compile(pattern)
    def __init__(self,m):
        self.note = Note(m)

class Snote(MatchLine):
    """Class representing the score note part of a match line."""
    fieldNames = ['Anchor','NoteName','Modifier','Octave',
                  'Bar','Beat','Offset','Duration',
                  'OnsetInBeats','OffsetInBeats','ScoreAttributesList']
    #pattern = 'snote\((.+),\[(.+),(.+)\],(.+),(.+):(.+),(.+),(.+),(.+),(.+),\[(.*)\]\)'
    pattern = 'snote\(([^,]+),\[([^,]+),([^,]+)\],([^,]+),([^,]+):([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),\[(.*)\]\)'
    reObj = re.compile(pattern)
    def __str__(self):
        r = ['Snote']
        for fn in Snote.fieldNames:
            r.append(' {0}: {1}'.format(fn,self.__dict__[fn]))
        return '\n'.join(r)+'\n'
    def __init__(self,m):
        MatchLine.__init__(self,m)
        self.MidiPitch = pitchName2Midi_PC(self.Modifier,self.NoteName,self.Octave)
        self.DurationInBeats = self.OffsetInBeats-self.OnsetInBeats
        self.DurationSymbolic = m.groups()[7]
        self.ScoreAttributesList = self.ScoreAttributesList.split(',')

class InfoLine(MatchLine):
    """Class representing an Info line."""
    fieldNames = ['Attribute','Value']
    pattern = 'info\(\s*([^,]+)\s*,\s*(.+)\s*\)\.'
    reObj = re.compile(pattern)

class MetaLine(MatchLine):
    """Class representing a Meta line."""
    fieldNames = ['Attribute','Value','Bar','TimeInBeats']
    pattern = 'meta\(\s*([^,]*)\s*,\s*([^,]*)\s*,\s*([^,]*)\s*,\s*([^,]*)\s*\)\.'
    reObj = re.compile(pattern)

class SnoteNoteLine(MatchLine):
    """Class representing a "match" (containing snote and note)."""
    pattern = Snote.pattern+'-'+Note.pattern
    reObj = re.compile(pattern)
    def __init__(self,m1,m2):
        self.snote = Snote(m1)
        self.note = Note(m2)

class SnoteDeletionLine(MatchLine):
    fieldNames = Snote.fieldNames
    pattern = Snote.pattern+'-deletion\.' # unused for efficiency reasons
    reObj = re.compile(pattern)
    def __init__(self,m1):
        self.snote = Snote(m1)
        
class InsertionNoteLine(MatchLine):
    fieldNames = Note.fieldNames
    pattern = 'insertion-'+Note.pattern # unused for efficiency reasons
    reObj = re.compile(pattern)
    def __init__(self,m2):
        self.note = Note(m2)

class HammerBounceNoteLine(MatchLine):
    fieldNames = Note.fieldNames
    pattern = 'hammer_bounce-'+Note.pattern # unused for efficiency reasons
    reObj = re.compile(pattern)
    def __init__(self,m2):
        self.note = Note(m2)

class OrnamentNoteLine(MatchLine):
    fieldNames = Note.fieldNames
    pattern = 'ornament\([^\)]*\)-'+Note.pattern # unused for efficiency reasons
    reObj = re.compile(pattern)
    def __init__(self,m2):
        self.note = Note(m2)

class TrillNoteLine(MatchLine):
    fieldNames = Note.fieldNames
    pattern = 'trill\([^\)]*\)-'+Note.pattern # unused for efficiency reasons
    reObj = re.compile(pattern)
    def __init__(self,m2):
        self.note = Note(m2)

class SnoteTrailingLine(MatchLine):
    fieldNames = ['Anchor','NoteName','Modifier','Octave',
                  'Bar','Beat','Offset','Duration',
                  'OnsetInBeats','OffsetInBeats','ScoreAttributesList']
    pattern = 'snote\((.+),\[(.+),(.+)\],(.+),(.+):(.+),(.+),(.+),(.+),(.+),\[(.*)\]\)'
    reObj = re.compile(pattern)
    def __init__(self,m):
        self.snote = Snote(m)

class SnoteOnlyLine(MatchLine):
    fieldNames = ['Anchor','NoteName','Modifier','Octave',
                  'Bar','Beat','Offset','Duration',
                  'OnsetInBeats','OffsetInBeats','ScoreAttributesList']
    pattern = 'snote\((.+),\[(.+),(.+)\],(.+),(.+):(.+),(.+),(.+),(.+),(.+),\[(.*)\]\)'
    reObj = re.compile(pattern)
    def __init__(self,m):
        self.snote = Snote(m)

class MatchFile:
    """This class can be instantiated using a match filename.
    
    """
    def __init__(self,filename):
        fileData = utilities.readFile(filename,strip=True)

        # try to get voice information file
        self.voiceIdxFile = []
        self.lines = nu.array([self.parseMatchLine(l) for l in fileData])

    def getInfo(self,attribute=None):
        info = [i for i in self.lines if isinstance(i,InfoLine)]
        if attribute:
            try:
                idx = [i.Attribute for i in info].index(attribute)
                return info[idx].Value
            except:
                return None
        else:
            return [i for i in self.lines if isinstance(i,InfoLine)]

    def getLinesAtScoreTimes(self,times):
        snoteLines = [l for l in self.lines if hasattr(l,'snote')]
        #nu.savetxt('/tmp/o.txt',[(l.snote.OnsetInBeats,l.snote.OffsetInBeats) for l in snoteLines])
        onoffsets = nu.array([(l.snote.OnsetInBeats,l.snote.OffsetInBeats) for l in snoteLines],
                          dtype=nu.dtype([('onset',nu.float),('offset',nu.float)]))
        lidx = nu.argsort(onoffsets,order=('onset','offset'))
        #idx = nu.arange(0,onoffsets.shape[0])
        #nu.savetxt('/tmp/o.txt',nu.column_stack((idx,onoffsets['onset'][lidx],
        #                                         nu.zeros(onoffsets.shape[0]),
        #                                         onoffsets['offset'][lidx]-onoffsets['onset'][lidx])))

        tidx = nu.argsort(times)
        i = 0
        i_min = 0
        result = []
        for t in times[tidx]:
            r = []
            ii = []
            i = i_min
            while i < len(lidx) and not (onoffsets['onset'][lidx[i]] > t and onoffsets['offset'][lidx[i]] > t):
                if (onoffsets['onset'][lidx[i]] <= t and onoffsets['offset'][lidx[i]] > t):
                    r.append(lidx[i])
                    ii.append(i)
                i += 1
            if len(ii) > 0:
                i_min = ii[0]
            result.append(r)
        return [[snoteLines[x] for x in notes] for notes in result]
  

    def getFirstOnset(self):
        self.getSnoteIdx()
        if len(self.snoteIdx) == 0:
            return None
        else:
            return self.lines[self.snoteIdx[0]].snote.OnsetInBeats

    def getTimeSignatures(self):
        tspat = re.compile('([0-9]+)/([0-9]*)')
        m = [(int(x[0]),int(x[1])) for x in 
             tspat.findall(self.getInfo('timeSignature'))]
        timeSigs = []
        if len(m) > 0:
            timeSigs.append((self.getFirstOnset(),m[0]))
        for l in self.getTimeSigLines():
            timeSigs.append((float(l.TimeInBeats),[(int(x[0]),int(x[1])) for x in tspat.findall(l.Value)][0]))
        timeSigs = list(set(timeSigs))
        timeSigs.sort(key=lambda x: x[0])
        return timeSigs

    def getTimeSigLines(self):
        return [i for i in self.lines if
                isinstance(i,MetaLine) and
                hasattr(i,'Attribute') and
                i.Attribute == 'timeSignature']

    def getSnoteIdx(self):
        """Return the line numbers that have snotes. 
        
        """
        #return [i.snote for i in self.lines if hasattr(i,'snote') and isinstance(i.snote,Snote)]
        if hasattr(self,'snotes'):
            return self.snoteIdx
        else:
            self.snoteIdx = [i for i,l in enumerate(self.lines) if hasattr(l,'snote')]
        return self.snoteIdx

    def getSopranoVoice(self,returnIndices=False):
        """Return the snotes marked as soprano notes
        (excluding those marked as grace notes).
        
        """
        if returnIndices:
            return [i for i,l in enumerate(self.lines) if hasattr(l,'snote') and 's' in l.snote.ScoreAttributesList and \
                        not 'grace' in l.snote.ScoreAttributesList and l.snote.Duration > 0.0 ]
        else:
            return [l for l in self.lines if hasattr(l,'snote') and 's' in l.snote.ScoreAttributesList and \
                        not 'grace' in l.snote.ScoreAttributesList and l.snote.Duration > 0.0 ]
    
      
    def getHighestVoiceWithoutIndexFile(self,excludeGrace=True,returnIndices=False):
        sopr = self.getSopranoVoice(returnIndices)
        if len(sopr) > 0:
            return(sopr)

        def isGrace(note):
            return 'grace' in note.ScoreAttributesList
        def isInLowerStaff(note):
            return 'staff2' in note.ScoreAttributesList

        idx = self.getSnoteIdx()
        
        features = []
        for i,idx in enumerate(self.snoteIdx):
            n = self.lines[idx].snote 
            if not (isInLowerStaff(n) 
                    or (excludeGrace and isGrace(n))
                    or n.Duration == 0.0):
                features.append((n.OnsetInBeats,n.OffsetInBeats,n.MidiPitch[0],i))
        
        features = nu.array(features)
        # sort according to pitch (highest first)
        features = features[nu.argsort(features[:,2])[::-1]]

        # sort according to onset (smallest first)
        features = features[nu.argsort(features[:,0],kind='mergesort')]

        voice = [features[0,:]]
        for f in features:
            # if onset is later_eq than last voice offset, add next note
            if f[0] >= voice[-1][1]:
                voice.append(f)
        
        # indices into the list of snotes
        indices = nu.array(nu.array(voice)[:,3],nu.int)
        if returnIndices:
            return nu.array(self.snoteIdx)[indices]
        else:
            #return [m for i,m in enumerate(self.lines[self.snoteIdx]) if i in indices]
            return [l for l in self.lines[self.snoteIdx][indices]]

    def parseMatchLine(self,l):
        """Returns objects representing the line as:
        * hammer_bounce-PlayedNote.
        * info(Attribute, Value).
        * insertion-PlayedNote.
        * ornament(Anchor)-PlayedNote.
        * ScoreNote-deletion.
        * ScoreNote-PlayedNote.
        * ScoreNote-trailing_score_note.
        * trailing_played_note-PlayedNote.
        * trill(Anchor)-PlayedNote.
        * meta(Attribute,Value,Bar,Beat).

        """      

        snoteMatch = Snote.reObj.search(l)
        noteMatch = Note.reObj.search(l,pos=snoteMatch.end() if snoteMatch else 0)
        if snoteMatch:
            if noteMatch:
                return SnoteNoteLine(snoteMatch,noteMatch)
            else:
                if re.compile('-deletion\.$').search(l,pos=snoteMatch.end()):
                    return SnoteDeletionLine(snoteMatch)
                else:
                    if re.compile('-trailing_score_note\.$').search(l,pos=snoteMatch.end()):
                        return SnoteTrailingLine(snoteMatch)
                    else:
                        return SnoteOnlyLine(snoteMatch)
        else: # no snoteMatch
            if noteMatch:
                if re.compile('^insertion-').search(l,endpos=noteMatch.start()):
                    return InsertionNoteLine(noteMatch)
                elif re.compile('^trill\([^\)]*\)-').search(l,endpos=noteMatch.start()):
                    return TrillNoteLine(noteMatch)
                elif re.compile('^ornament\([^\)]*\)-').search(l,endpos=noteMatch.start()):
                    return OrnamentNoteLine(noteMatch)
                elif re.compile('^trailing_played_note-').search(l,endpos=noteMatch.start()):
                    return TrailingNoteLine(noteMatch)
                elif re.compile('^hammer_bounce-').search(l,endpos=noteMatch.start()):
                    return HammerBounceNoteLine(noteMatch)
                else:
                    return False
            else:
                metaMatch = MetaLine.reObj.search(l)
                if metaMatch:
                    return MetaLine(metaMatch,lambda x: x)
                else:
                    infoMatch = InfoLine.reObj.search(l)
                    if infoMatch:
                        return InfoLine(infoMatch,
                                        fieldInterpreter=utilities.interpretField)
                    else:
                        return UnknownMatchLine(l)
                    #fieldInterpreter=utilities.interpretFieldRationalClass)


    
##m = scorePerformanceMatch(sys.argv[1])
##for k,i in m.getTempoCurveNoteLevel():
##    print k,i
#print m.getSopranoVoice()
#print len(m.matches)




################### OLD ################### OLD ################### OLD
# import bnetUtils, midiUtils

# class ScorePerformanceMatch:
#     def __init__(self,filename):
#         fileData = utilities.readFile(filename,strip=True)

#         # try to get voice information file
#         try:
#             self.voiceIdxFile = os.path.join(RENCON08PATH, 'data/feature_files/', os.path.splitext(os.path.split(filename)[1])[0] + '_voiceIndices.feat')
#         except:
#             failed = True
#         if failed or not(os.path.isfile(self.voiceIdxFile) and os.access(self.voiceIdxFile,os.R_OK)):
#           self.voiceIdxFile = []
#         else:
#           self.vIdxDS = bnetUtils.readFeatureFile(self.voiceIdxFile)
        
#         self.headerInfo = dict([self.parseInfoLine(l) for l in fileData if l[0:4] == 'info'])
#         self.matches = [self.parseMatchLine(l) for l in fileData if l[0:5 ] == 'snote' and \
#                         l.find('-note(') > 0]
#         self.snotes = [self.parseMatchLine(l,needMatch=False) for l in fileData if l[0:5 ] == 'snote']
        

#     def scoreToMidi(self,file,units=480,tempo=1000000):
#         m = midiUtils.MidiFile()
#         m.setHeader(midiUtils.MidiHeader(0,1,units))
#         events = [midiUtils.TempoEvent(0,tempo)]
#         snotes = [i[0] for i in self.snotes if i[0] is not None ]
#         snotes.sort(key = lambda x: x['OnsetInBeats'])
#         firstOn = snotes[0]['OnsetInBeats'] if len(snotes) > 0 else 0
#         vel = 80
#         ch = 1
#         for i in snotes:
#             print('%d on: %d duration: %d' % (i['MidiPitch'][0],int(units*(i['OnsetInBeats']-firstOn)),
#                                               int(units*(i['OffsetInBeats']-firstOn))-int(units*(i['OnsetInBeats']-firstOn))))
#             print('%d off: %d' % (i['MidiPitch'][0],int(units*(i['OffsetInBeats']-firstOn))))
#             print(i)
#             print('')
#             events.append(midiUtils.NoteOnEvent(int(units*(i['OnsetInBeats']-firstOn)),
#                                                 ch,i['MidiPitch'][0],vel))
#             events.append(midiUtils.NoteOffEvent(int(units*(i['OffsetInBeats']-firstOn)),
#                                                  ch,i['MidiPitch'][0],0))
#         events.sort(key = lambda x: x.getTime())
#         track = midiUtils.MidiTrack()
#         track.setEvents(events)
#         m.addTrack(track)
#         m.writeFile(file)

#     def getSopranoVoice(self):
#         return [p for p in self.matches if 's' in p[0]['ScoreAttributesList'] and \
#                 not 'grace' in p[0]['ScoreAttributesList'] and p[0]['Duration'] > 0.0 ]
    
#     def getHighestVoice(self):
#       if self.voiceIdxFile:
#         return self.getHighestVoiceIdxFile()
#       else:
#         return self.getHighestVoiceWithoutIndexFile()
        
#     def getHighestVoiceIdxFile(self):
#       #print("idx file")
#       return [m for i, m in enumerate(self.matches or self.snotes) if i+1 in self.vIdxDS.getDataByFeature('voiceIndices')]
      
#     def getHighestVoiceWithoutIndexFile(self,excludeGrace=True,returnIndices=False):
#       #print("not idx file")
#       sopr = self.getSopranoVoice()
#       if len(sopr) > 0:
#           return(sopr)
#       def isGrace(note):
#           return 'grace' in note['ScoreAttributesList']
#       def isInLowerStaff(note):
#           return 'staff2' in note['ScoreAttributesList']

#       snotes = [(i,x) for i,x in enumerate(self.matches or self.snotes)
#                 if x[0] and not (isInLowerStaff(x[0]) or (excludeGrace and isGrace(x[0]))
#                                  or x[0]['Duration'] == 0.0)]
#       features = nu.array([[v[0]['OnsetInBeats'],v[0]['OffsetInBeats'],v[0]['MidiPitch'][0],i]
#                               for i,v in snotes if v[0]])
#       ## sort according to pitch (highest first)
#       features = nu.flipud(features[nu.argsort(features[:,2]),:])
#       ## sort according to onset (smallest first)
#       features = features[nu.argsort(features[:,0],kind='mergesort'),:]
#       voice = [features[0,:]]
#       for f in features:
#           ## if onset is later_eq than last voice offset, add next note
#           if f[0] >= voice[-1][1]:
#               voice.append(f)
#       indices = list(nu.array(voice)[:,3])
#       if returnIndices:
#           return indices
#       else:
#           #return [m for i,m in enumerate(self.matches) if i in indices]
#           return [m for i,m in enumerate(self.matches or self.snotes) if i in indices]

#     def getHighestVoiceOld(self,excludeGrace=True,returnIndices=False):
#         def isGrace(note):
#             return 'grace' in note['ScoreAttributesList'] and note['Duration'] == 0
#         def isInLowerStaff(note):
#             return 'staff2' in note['ScoreAttributesList']


#         snotes = [(i,x) for i,x in enumerate(self.matches or self.snotes)
#                   if x[0] and (not isInLowerStaff(x[0])) and not (excludeGrace and isGrace(x[0]))]
#         for i in snotes:
#             print(i[0]['ScoreAttributesList'])
#         features = nu.array([[v[0]['OnsetInBeats'],v[0]['OffsetInBeats'],v[0]['MidiPitch'][0],i]
#                                 for i,v in snotes if v[0]])
#         features = nu.flipud(features[nu.argsort(features[:,2]),:])
#         features = features[nu.argsort(features[:,0],kind='mergesort'),:]
#         voice = [features[0,:]]
#         for f in features:
#             if f[0] >= voice[-1][1]:
#                 voice.append(f)
#         indices = list(nu.array(voice)[:,3])
#         if returnIndices:
#             return indices
#         else:
#             #return [m for i,m in enumerate(self.matches) if i in indices]
#             return [m for i,m in enumerate(self.snotes) if i in indices]

#     def printHeaderInfo(self):
#         for k,v in self.headerInfo.items():
#             print k,":",v

#     def getPerformedSopranoIOIratios(self):
#         sopranoVoice = self.getSopranoVoice()
#         sOnsets = [s[0]['OnsetInBeats'] for s in sopranoVoice]
#         # last note has offset instead of next note onset
#         pOnsets = [s[1]['Onset'] for s in sopranoVoice]
#         sIOI = nu.array(sOnsets[1:])-nu.array(sOnsets[:-1])
#         pIOI = nu.array(pOnsets[1:])-nu.array(pOnsets[:-1])
#         #sIOI,pIOI,positions = zip(*[(i,j,k) for i,j,k in zip(list(sIOI),list(pIOI),positions) if i != 0 and j != 0])
#         sIOI = nu.array(sIOI)
#         pIOI = nu.array(pIOI)
#         return nu.array([math.log(i) for i in pIOI/(sIOI*self.headerInfo['midiClockUnits'])])
  
#     def getTempoCurveNoteLevel(self):
#         sopranoVoice = self.getSopranoVoice()
#         sOnsets = [s[0]['OnsetInBeats'] for s in sopranoVoice]
#         # last note has offset instead of next note onset
#         sOnsets.append(sopranoVoice[-1][0]['OffsetInBeats'])
#         pOnsets = [s[1]['Onset'] for s in sopranoVoice]
#         pOnsets.append(sopranoVoice[-1][1]['Offset'])
#         positions = sOnsets[:-1]
#         sIOI = nu.array(sOnsets[1:])-nu.array(sOnsets[:-1])
#         pIOI = nu.array(pOnsets[1:])-nu.array(pOnsets[:-1])
#         sIOI,pIOI,positions = zip(*[(i,j,k) for i,j,k in zip(list(sIOI),list(pIOI),positions) if i != 0 and j != 0])
#         sIOI = nu.array(sIOI)
#         pIOI = nu.array(pIOI)
#         return [(k,i) for i,k in zip(list(self.headerInfo['midiClockUnits']*sIOI*60000000.0/ \
#                                  ( pIOI * self.headerInfo['midiClockRate'] )), positions)
#                 if i != 0.0]

#     def parseMatchLine(self,l,needMatch=True):
#         sn = None
#         n = None
#         snoteREString = 'snote\((.+),\[(.+),(.+)\],(.+),(.+):(.+),(.+),(.+),(.+),(.+),\[(.*)\]\)'
#         noteREString = '[^s]note\((.+),\[(.+),(.+)\],(.+),(.+),(.+),(.+),(.+)\)'
#         timeSigMetaREString = 'meta(timeSignature,([^,]*),([^,]*),([^,]*))\.'
#         keySigMetaREString = 'meta(keySignature,\'([^\']*)\',([^,]*),([^,]*))\.'
#         snPattern = re.compile(snoteREString)
#         tsPattern = re.compile(timeSigMetaREString)
#         ksPattern = re.compile(keySigMetaREString)
#         m = snPattern.search(l)
#         if m:
#             # turn strings into integers or floats whenever possible:
#             groups = [utilities.interpretFieldRational(i) for i in m.groups()]
#             sn = dict(zip(['Anchor','NoteName','Modifier','Octave','Bar','Beat','Offset',\
#                            'Duration','OnsetInBeats','OffsetInBeats','ScoreAttributesList'],
#                           groups))
#             sn['MidiPitch'] = pitchName2Midi_PC(sn['Modifier'],sn['NoteName'],sn['Octave'])
#             # split string into list
#             sn['ScoreAttributesList'] = sn['ScoreAttributesList'].split(',')
#         nPattern = re.compile(noteREString)
#         mm = nPattern.search(l)
#         if mm:
#             groups = [utilities.interpretFieldRational(i) for i in mm.groups()]
#             n = dict(zip(['Number','NoteName','Modifier','Octave','Onset','Offset','AdjOffset','Velocity'],
#                          groups))
#             n['MidiPitch'] = pitchName2Midi_PC(n['Modifier'],n['NoteName'],n['Octave'])
#         else:
#             ##print('oei, note regex didn\'t match...')
#             if needMatch:
#                 return None
#         return (sn,n)

#     def parseInfoLine(self,l):
#         commaIndex = l.index(',')
#         iKey = l[5:commaIndex]
#         iVal = l[commaIndex+1:-2]
#         return (iKey, utilities.interpretField(iVal.strip("'")))



