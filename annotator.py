#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygtk,sys,os
pygtk.require("2.0")
import gtk
import gtk.glade
import numpy as nu

class Annotator(object):

    def __init__(self,files):
        self.files = files
        self.gladefile = "annotator.xml" 
        self.glade = gtk.Builder()
        self.glade.add_from_file(self.gladefile)
        self.glade.connect_signals(self)
        self.glade.get_object("window1").show_all()
        self.i = -1
        self.xy = (0,0)
        self.nextImg()
        self.bars = [[] for x in range(len(self.files))]
        #print(self.glade.get_object("image1"))
    def on_window1_destroy(self, *args):
        self.quit()
    def quit(self, *args):
        gtk.main_quit()
    def on_bpress(self,window,event):
        if event.button == 1:
            self.addPoint((event.x,event.y))
    def addPoint(self,p):
        print('adding point',p,self.i)
        self.bars[self.i].append(p)
    def savePoints(self,i):
        print('file',i)
        print('bars',[len(b) for b in self.bars])
        fn = os.path.join('/tmp/',os.path.splitext(os.path.basename(self.files[i]))[0]+'.txt')
        # useful when annotating scaled down images:
        multiplier = 1.0
        nu.savetxt(fn,(multiplier*nu.array(self.bars[i])).astype(nu.int),fmt='%d')
        print('points saved to {0}'.format(fn))
    def on_motion(self,widget,event):
        self.xy = (event.x,event.y)
    def showPoints(self,i):
        fn = os.path.join('/tmp/',os.path.splitext(os.path.basename(self.files[i]))[0]+'.txt')
        multiplier = 3.5
        print('file {0}'.format(fn))
        print((multiplier*nu.array(self.bars[i])).astype(nu.int))

    def erasePoints(self,i):
        self.bars[i] = []
        print('print erased points')
    def removeLastPoint(self,i):
        if len(self.bars[i]) > 0:
            self.bars[i] = self.bars[i][:-1]
            print('removed last point, points:')
            print(nu.array(self.bars[i]))
    def on_irel(self,window,event):
        print('joi',event)
    def on_kpress(self,window,event):
        keyname = gtk.gdk.keyval_name( event.keyval)
        if event.state & gtk.gdk.CONTROL_MASK:
            if keyname == 'q':
                self.quit()
            if keyname == 'n':
                self.nextImg()
            if keyname == 'p':
                self.prevImg()
            if keyname == 's':
                self.savePoints(self.i)
            if keyname == 'a':
                self.showPoints(self.i)
            if keyname == 'e':
                self.erasePoints(self.i)
            if keyname == 'z':
                self.removeLastPoint(self.i)
        else:
            if keyname == 'space':
                self.addPoint(self.xy)
                
    def prevImg(self):
        self.i -= 1
        if self.i < 0:
            print('first image')
            self.i += 1
        else:
            self.glade.get_object("image2").set_from_file(self.files[self.i])
            print('showing image ',self.files[self.i])

    def nextImg(self):
        self.i += 1
        if self.i >= len(self.files):
            print('last image')
            self.i -= 1
            #gtk.main_quit()
        else:
            self.glade.get_object("image2").set_from_file(self.files[self.i])
            print('showing image ',self.files[self.i])
if __name__ == "__main__":
    files = sys.argv[1:]
    #x= gtk.gdk.pixbuf_new_from_file(fn)
    #print(x)
    try:
        a = Annotator(files)
        gtk.main()
    except KeyboardInterrupt:
        pass
