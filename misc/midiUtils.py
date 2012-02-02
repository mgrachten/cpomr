#!/usr/bin/env python

#    Copyright 2012, Maarten Grachten.
#
#    This file is part of mg_python_modules.
#
#    mg_python_modules is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mg_python_modules is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with mg_python_modules.  If not, see <http://www.gnu.org/licenses/>.

from utilities import partition
from numpy import *
from sys import stderr
from midi.MidiOutStream import MidiOutStream
from midi.MidiInFile import MidiInFile
from midi.MidiOutFile import MidiOutFile

def tic2ms(n,tempo,div):
    return tempo*n/(1000.0*div)

def tic2beat(n,div):
    return n/float(div)

######################### Midi Ontology
class MidiTimeEvent(object):
    def __init__(self,time):
        self.time = time
    #def __str__(self):
    #    return 'asdf'
    def getTime(self):
        return self.time

    def setTime(self,time):
        self.time = time

    def send(self,midi):
        pass

class MidiMetaEvent(MidiTimeEvent):
    def __init__(self,time):
        MidiTimeEvent.__init__(self,time)


class EndOfTrackEvent(MidiMetaEvent):
    def __init__(self,time):
        MidiMetaEvent.__init__(self,time)
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.end_of_track()

class TextEvent(MidiMetaEvent):
    def __init__(self,time,text):
        MidiMetaEvent.__init__(self,time)
        self.text = text
    def getText(self):
        return self.text
    def setText(self,text):
        self.text = text
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.text(self.getText())

class TrackNameEvent(MidiMetaEvent):
    def __init__(self,time,name):
        MidiMetaEvent.__init__(self,time)
        self.name = name
    def getTrackName(self):
        return self.name
    def setTrackName(self,name):
        self.name = name
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.sequence_name(self.getTrackName())

class InstrumentNameEvent(MidiMetaEvent):
    def __init__(self,time,name):
        MidiMetaEvent.__init__(self,time)
        self.name = name
    def getInstrumentName(self):
        return self.name
    def setInstrumentName(self,name):
        self.name = name
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.instrument_name(self.getInstrumentName())

class TempoEvent(MidiMetaEvent):
    def __init__(self,time,val):
        MidiMetaEvent.__init__(self,time)
        self.tempo = val
    def getTempo(self):
        return self.tempo
    def setTempo(self,tempo):
        self.tempo = tempo
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.tempo(self.getTempo())
    
class KeySigEvent(MidiMetaEvent):
    def __init__(self,time,key,scale):
        MidiMetaEvent.__init__(self,time)
        self.key = key
        self.scale = scale
    def getKey(self):
        return self.key
    def getScale(self):
        return self.scale

    def setKey(self,key):
        self.key = key
    def setScale(self,scale):
        self.scale = scale

    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.key_signature(self.getKey(),self.getScale())

class TimeSigEvent(MidiMetaEvent):
    def __init__(self,time, num, den, metro, thirtysecs):
        MidiMetaEvent.__init__(self,time)
        self.num = num
        self.den = den
        self.metro = metro
        self.thirtysecs = thirtysecs
    def getNum(self):
        return self.num
    def getDen(self):
        return self.den
    def getMetro(self):
        return self.metro
    def getThirtyseconds(self):
        return self.thirtysecs

    def setNum(self,num):
        self.num = num
    def setDen(self,den):
        self.den = den
    def setMetro(self,metro):
        self.metro = metro
    def setThirtyseconds(self,thirtysecs):
        self.thirtysecs = thirtysecs

    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.time_signature(self.getNum(),self.getDen(),self.getMetro(),self.getThirtyseconds())
        
class MidiEvent(MidiTimeEvent):
    def __init__(self,time,ch):
        MidiTimeEvent.__init__(self,time)
        self.channel = ch

    def getChannel(self):
        return self.channel

    def setChannel(self,ch):
        self.channel = ch

    def getChannelForSend(self):
        """Ugly workaround for apparent bug in mxm code:
        interprets channels as one lower as supposed to be"""
        return self.channel - 1


class PatchChangeEvent(MidiEvent):
    def __init__(self,time,channel,patch):
        MidiEvent.__init__(self,time,channel)
        self.patch = patch
    def getPatch(self):
        return self.patch
    def setPatch(self,patch):
        self.patch = patch
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.patch_change(self.getChannelForSend(),self.getPatch())



class NoteOnOffEvent(MidiEvent):
    def __init__(self,time,ch,note,velocity):
        MidiEvent.__init__(self,time,ch)
        self.note = note
        self.velocity = velocity
    def __str__(self):
        return '%s On %s %s %s' % (self.getTime(),self.getChannel(),\
                                   self.getNote(),self.getVelocity())
    def getNote(self):
        return self.note
    def getVelocity(self):
        return self.velocity

    def setNote(self,note):
        self.note = note
    def setVelocity(self,vel):
        self.velocity = vel

class NoteOnEvent(NoteOnOffEvent):
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.note_on(channel=self.getChannelForSend(),
                     note=self.getNote(),
                     velocity=self.getVelocity())

class NoteOffEvent(NoteOnOffEvent):
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.note_off(channel=self.getChannelForSend(),
                      note=self.getNote(),
                      velocity=self.getVelocity())

class AftertouchEvent(NoteOnOffEvent):
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.aftertouch(channel=self.getChannelForSend(),
                        note=self.getNote(),
                        velocity=self.getVelocity())

class ControllerEvent(MidiEvent):
    def __init__(self,time,ch,controller,value):
        MidiEvent.__init__(self,time,ch)
        self.controller = controller
        self.value = value
    def __str__(self):
        return '%s Par %s %s %s' % (self.getTime(),self.getChannel(),\
                                   self.getController(),self.getValue())
    def getController(self):
        return self.controller
    def getValue(self):
        return self.value
    def setController(self,controller):
        self.controller = controller
    def setValue(self,value):
        self.value = value
    def send(self,midi):
        midi.update_time(self.getTime(),relative=0)
        midi.continuous_controller(channel=self.getChannelForSend(),
                                   controller=self.getController(),
                                   value=self.getValue())

class MidiNote:
    def __init__(self,onsetEvent,offsetEvent):
        self.onset = onsetEvent
        self.offset = offsetEvent
    def getOnset(self):
        return self.onset.getTime()
    def getOffset(self):
        return self.offset.getTime()
    def getDuration(self):
        return self.getOffset() - self.getOnset()
    def getChannel(self):
        return self.onset.getChannel()
    def getNote(self):
        return self.onset.getNote()
    def getVelocity(self):
        return self.onset.getVelocity()

    def setOnset(self,time):
        self.onset.setTime(time)
    def setOffset(self,time):
        self.offset.setTime()
    def setChannel(self,ch):
        self.onset.setChannel(ch)
        self.offset.setChannel(ch)
    def setNote(self,note):
        self.onset.setNote(note)
        self.offset.setNote(note)
    def setVelocity(self,vel):
        return self.onset.setVelocity(vel)

    def __str__(self):
        return '%s %s %s %s %s' % (self.getOnset(),self.getOffset(),\
                                   self.getChannel(),self.getNote(),self.getVelocity())

## some events still missing, like aftertouch etc
class MidiFile:
    def __init__(self,in_file=None,zeroVelOnIsOff=False):
        self.header = None
        self.tracks = []
        self.zeroVelOnIsOff = zeroVelOnIsOff
        if in_file is not None:
            self.readFile(in_file)
    
    def summarize(self):
        out = [self.getHeader().summarize()]
        for t in self.getTracks():
            out.append(t.summarize())
        return '\n'.join(out)

    def computeTime(self,time,track=0,defaultTempo=1000000):
        """Compute the time in seconds for <time> (in midi units), taking
        into account the tdiv, and tempo events in <track>"""
        try:
            return self.computeTimes([time],track=track,defaultTempo=defaultTempo)[0]
        except IndexError:
            print('No Tempo events found in file, could not compute time.')
            
    def computeTimes(self,times,track=0,defaultTempo=1000000):
        """Compute the time in seconds for <time> (in midi units), taking
        into account the tdiv, and tempo events in <track>"""
        try:
            events = self.getTrack(track).getEvents(TempoEvent)
        except:
            print('midi file has no %d-th track' % track)
            return False
        if len(events) > 0 and events[0].getTime() > 0:
            ## assume default tempo until first tempo event
            events.insert(0,TempoEvent(0,defaultTempo))
        mtime = max(times)
        timeTempo = array([(e.getTime(),e.getTempo()) for e in events if e.getTime() < mtime],double)
        tempoTimes = transpose(array((timeTempo[:,0],
                                      concatenate((array([0]),cumsum(timeTempo[:-1,1]*diff(timeTempo[:,0])))),
                                      timeTempo[:,1]),ndmin=2))
        j = 0
        result = [0]*len(times)
        for i in argsort(array(times)):
            while j < tempoTimes.shape[0] and tempoTimes[j,0] > times[i]:
                j = j+1
            result[i] = (tempoTimes[j-1,1] + (times[i]-tempoTimes[j-1,0])*tempoTimes[j-1,2])/\
                        (10**6*float(self.getHeader().getTimeDivision()))
        return result

    def getHeader(self):
        return self.header
    def setHeader(self,header):
        self.header = header
    def getTracks(self):
        return self.tracks
    def getTrack(self,n=0):
        return self.tracks[n]
    def replaceTrack(self,n,track):
        self.tracks[n] = track
    def addTrack(self,track):
        self.tracks.append(track)
        
    def readFile(self,filename):
        self.midi_in = MidiInFile(MidiHandler(self, self.zeroVelOnIsOff), filename)
        ## header and tracks get instantiated through the midi event handler
        self.midi_in.read()

    def writeFile(self,filename):
        midi = MidiOutFile(filename)
        self.getHeader().send(midi)
        [track.send(midi) for track in self.getTracks()]
        midi.eof()

class MidiHeader:
    def __init__(self,format,numberOfTracks=1,timeDivision=480):
        self.format = format
        self.numberOfTracks = numberOfTracks
        self.timeDivision = timeDivision
    def __str__(self):
        return 'MFile %d %d %d\n' % (self.format,self.numberOfTracks,self.timeDivision)
    def getFormat(self):
        return self.format
    def getNumberOfTracks(self):
        return self.numberOfTracks
    def getTimeDivision(self):
        return self.timeDivision
    def setFormat(self,value):
        self.format = value
    def setNumberOfTracks(self,value):
        self.numberOfTracks = value
    def setTimeDivision(self,value):
        self.timeDivision = value
    def send(self,midi):
        midi.header(format=self.getFormat(),
                    nTracks=self.getNumberOfTracks(),
                    division=self.getTimeDivision())
    def summarize(self):
        out = "Midi Header\n"
        out += "        Format: %s\n" % self.getFormat()
        out += " Nr. of Tracks: %s\n" % self.getNumberOfTracks()
        out += "      Division: %s\n" % self.getTimeDivision()
        return out

    def asText(self):
        """print header in tm
        f representation"""
        "todo"
        pass

class MidiTrack:
    def __init__(self,events = []):
        self.events = events

    def summarize(self):
        events = partition(lambda x: isinstance(x,MidiMetaEvent), self.getEvents())
        ev = list(set([e.__class__.__name__ for e in  events.get(False,[])]))
        mev = list(set([e.__class__.__name__ for e in  events.get(True,[])]))
        midiChannels = list(set([e.getChannel() for e in events.get(False,[])]))
        tname = self.getEvents(TrackNameEvent)
        out = "Midi Track\n"
        if len(tname) > 0:
            out += "        Track Name: %s\n" % tname[0].getTrackName()
        out += "     Nr. of Events: %s\n" % len(events.get(False,[]))
        out += " Nr. of MetaEvents: %s\n" % len(events.get(True,[]))
        out += "    Event Channels: %s\n" % midiChannels
        out += "            Events: %s\n" % ev
        out += "       Meta Events: %s\n" % mev
        return out
    
    def addEvent(self,event):
        """Add an event to the track"""
        self.events.append(event)

    def close(self):
        """Sort events and add an EndOfTrack event to the track if it's missing"""
        self.sortEvents()
        endTime = int(self.getEvents()[-1].getTime() if len(self.getEvents()) > 0 else 0)
        if len(self.getEvents()) == 0 or not isinstance(self.getEvents()[-1],EndOfTrackEvent):
            self.addEvent(EndOfTrackEvent(endTime))
        
    def sortEvents(self):
        self.events.sort(key=lambda x: x.getTime())
        
    def getEvents(self,eventType=None,filters=None):
        """Get all events of type eventType from track that return True on all filter predicates"""
        result = self.events
        if not eventType == None:
            result = [e for e in result if isinstance(e,eventType)]
        if not filters == None:
            result = [e for e in result if all([f(e) for f in filters])]
        return result

    def setEvents(self,events):
        """Replace the event list in track by events"""
        self.events = events

    def send(self,midi):
        midi.start_of_track()
        #events = self.getEvents()
        #events.sort(key=lambda x: x.getTime())
        self.close()
        [e.send(midi) for e in self.getEvents()]

    def getOnOffs(self):
        """Get all NoteOnEvents and NoteOffEvents from track"""
        onoffs = self.getEvents(NoteOffEvent)+self.getEvents(NoteOnEvent)
        onoffs.sort(key=lambda x: x.getTime())
        return onoffs

    def getOnOffEqClasses(self):
        """Get all NoteOnEvents and NoteOffEvents from track,
        grouped by time"""
        onoffs = self.getOnOffs()
        return partition(lambda x: x.getTime(), onoffs)
  
    def getNotes(self):
        """Return a list of MidiNotes"""
        onoffs = self.getOnOffEqClasses()
        eventTimes = onoffs.keys()
        eventTimes.sort()
        acc = []
        sounding = {}
        errors = 0
        for t in eventTimes:
            onOffsByChannel = partition(lambda x: x.getChannel(), onoffs[t])
            for ch,evs in onOffsByChannel.items():
                onOffsByNote = partition(lambda x: x.getNote(), evs)
                for note,v in onOffsByNote.items():
                    isSounding = sounding.has_key((ch,note))
                    onOff = partition(lambda x: isinstance(x,NoteOnEvent),v)
                    ons = onOff.get(True,[])
                    offs = onOff.get(False,[])
                    ons = partition(lambda x: x.getVelocity() == 0,ons)
                    onsZeroVel = ons.get(True,[])
                    ons = ons.get(False,[])
                    nOns = len(ons)
                    nOnsZeroVel = len(onsZeroVel)
                    nOffs = len(offs) - nOnsZeroVel

                    if nOffs >= 0: # there's an off for every 0vel On
                        if nOnsZeroVel > 0:
                            acc.append(MidiNote(onsZeroVel[0],offs[0]))
                        else:
                            pass # there are no ons 0vel
                    else: # there's 0vel ons without off, treat as note off
                        if isSounding:
                            on = sounding[(ch,note)]
                            acc.append(MidiNote(on,onsZeroVel[0])) ## any off will do
                            del sounding[(ch,note)]
                        else:
                            ## Warn spurious note on with 0 vel
                            pass
                    if nOffs > 0:
                        if isSounding:
                            on = sounding[(ch,note)]
                            acc.append(MidiNote(on,offs[0])) ## any off will do
                            del sounding[(ch,note)]
                            isSounding = False
                        else:
                            ## Warn spurious note off
                            pass
                    if nOns > 0:
                        if isSounding:
                            ## Warn implicit note off by new note on
                            on = sounding[(ch,note)]
                            acc.append(MidiNote(on,ons[0])) ## any off will do
                            del sounding[(ch,note)]
                        sounding[(ch,note)] = ons[0]

        acc.sort(key=lambda x: x.getOnset())
        return acc
    def getHomophonicSlices(self):
        onoffs = self.getOnOffEqClasses()
        times = onoffs.keys()
        times.sort()
        sounding = set()
        acc = []
        prev_time = 0
        for t in times:
            if not sounding == {}:
                acc.append((prev_time,t,[i[1] for i in sounding]))
                prev_time = t
            onsOffs = partition(lambda x: isinstance(x,NoteOnEvent),onoffs[t])
            ons = set([(e.getChannel(),e.getNote()) for e in onsOffs.get(True,[])])
            offs = set([(e.getChannel(),e.getNote()) for e in onsOffs.get(False,[])])
            
            on = ons-offs-sounding
            off = offs-ons-sounding
            snd = sounding-ons-offs
            onOff = set.intersection(ons,offs)-sounding
            offSnd = set.intersection(sounding,offs)-ons
            onSnd = set.intersection(ons,sounding)-offs
            #onOffSnd = list(set.intersection(set.intersection(ons,offs),snds))
            onOffSnd = set.intersection(onOff,offSnd)
            for i in list(on):
                sounding.add(i) # add to sounding
            for i in list(offSnd):
                sounding.remove(i) # remove from sounding
            for i in list(onOff):
                pass #print('Warning, ignoring zero duration (grace) note at time %d' % t)
            if len(onSnd) > 1:
                print('Warning, ignoring unexpected note on at time %d' % t)
                print(onSnd)
            if len(off) > 1:
                print('Warning, ignoring unexpected %s note off at time %d' % (off, t))

        acc.sort(key=lambda x: x[0])
        return acc

class MidiHandler(MidiOutStream):
    """Event handler that constructs a MidiFile object
    (with MidiHeader and MidiTrack objects) from a midifile"""

    def __init__(self,midiFile,zeroVelOnIsOff):
        MidiOutStream.__init__(self)
        self.midiFile = midiFile
        self.zeroVelOnIsOff = zeroVelOnIsOff
        self.events = []
        
    def channel_message(self, message_type, channel, data):
        stderr.write('ch msg, type: %s, ch: %s, data: %s' % (message_type, channel, data))

    def start_of_track(self, n_track=0):
        self.events = []
        #self.track = MidiTrack()

    def end_of_track(self):
        self.events.append(EndOfTrackEvent(self.abs_time()))
        #self.track.events = self.events
        #self.midiFile.addTrack(self.track)
        self.midiFile.tracks.append(MidiTrack(self.events))
        #print(len(self.midiFile.getTrack().getEvents(NoteOnEvent)[0].getVelocity()))
        
    def continuous_controller(self, channel, controller, value):
        channel += 1
        self.events.append(ControllerEvent(self.abs_time(), channel, controller, value))

    def note_on(self, channel=1, note=0x40, velocity=0x40):
        channel += 1
        if self.zeroVelOnIsOff and velocity == 0:
            self.note_off(channel, note, velocity)
        else:
            self.events.append(NoteOnEvent(self.abs_time(),channel,note,velocity))

    def note_off(self, channel=1, note=0x40, velocity=0x40):
        channel += 1
        self.events.append(NoteOffEvent(self.abs_time(),channel,note,velocity))

    def header(self, format=0, nTracks=1, division=480):
        self.midiFile.setHeader(MidiHeader(format=format,numberOfTracks=nTracks,timeDivision=division))

    def text(self, text):
        self.events.append(TextEvent(self.abs_time(),text))

    def sequence_name(self, text):
        self.events.append(TrackNameEvent(self.abs_time(),text))

    def instrument_name(self, text):
        self.events.append(InstrumentNameEvent(self.abs_time(),text))

    def key_signature(self, sf, mi):
        self.events.append(KeySigEvent(self.abs_time(),key=sf,scale=mi))

    def time_signature(self, nn, dd, cc, bb):
        self.events.append(TimeSigEvent(self.abs_time(),num=nn,den=dd,metro=cc,thirtysecs=bb))

    def tempo(self, value):
        self.events.append(TempoEvent(self.abs_time(),value))
    def sysex_event(self, data):
        pass
    
    def patch_change(self, channel, patch):
        channel += 1
        self.events.append(PatchChangeEvent(self.abs_time(),channel,patch))

#################################################
## some convenience functions for midi files

def convertMidiToType0(inputMidiFile):
    """Return a new MidiFile with all tracks merged into a single one,
    effectively converting a type 1 midi file into a type 0 file.
    """
    outputMidiFile = MidiFile()
    header = inputMidiFile.getHeader()
    header.setFormat(0)
    header.setNumberOfTracks(1)
    outputMidiFile.setHeader(header)
    track = MidiTrack()
    allEvents = partition(
        lambda x: isinstance(x,EndOfTrackEvent) ,
        reduce(lambda x,y: x+y, [track.getEvents() for track in inputMidiFile.getTracks()]))
    endOfTrackEvents,nonEndOfTrackEvents = [allEvents[i] for i in (True,False)]
    endOfTrackEvents.sort(key=lambda x: x.getTime())
    track.setEvents(nonEndOfTrackEvents+[endOfTrackEvents[-1]])
    outputMidiFile.addTrack(track)
    return outputMidiFile

def convertMidiFileToType0(inputFilename,outputFilename):
    """Convert a type 1 midi file into a type 0 midi file by merging
    all tracks into a single one.
    """
    convertMidiToType0(MidiFile(inputFilename)).writeFile(outputFilename)

if __name__ == '__main__':
    pass
