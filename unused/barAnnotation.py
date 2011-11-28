

def getNumberedBarCoords(barpoints,scrImage):
    systems = si.getSystems()
    assignBarlinesToSystems(systems,barpoints)
    bar = 0
    bars = {}
    for system in systems:
        if system.barPoints.shape[0] == 0:
            tl = (system.staffs[0].staffLineAgents[0].getDrawMean()[0],0)
            br = (system.staffs[1].staffLineAgents[-1].getDrawMean()[0],scrImage.getWidth()-1)
            wh = (br[0]-tl[0],br[1]-tl[1])
            bars[bar] = bars.get(bar,[])+[reversed(tl),reversed(wh)]
        elif system.barPoints.shape[0] == 1:
            tl = (system.staffs[0].staffLineAgents[0].getDrawMean()[0],system.barPoints[0,0])
            br = (system.staffs[1].staffLineAgents[-1].getDrawMean()[0],scrImage.getWidth()-1)
            wh = (br[0]-tl[0],br[1]-tl[1])
            bars[bar] = bars.get(bar,[])+[reversed(tl),reversed(wh)]
        else:
            for i in range(1,system.barPoints.shape[0]):
                tl = (system.staffs[0].staffLineAgents[0].getDrawMean()[0],system.barPoints[i-1,0])
                br = (system.staffs[1].staffLineAgents[-1].getDrawMean()[0],system.barPoints[i,0])
                wh = (br[0]-tl[0],br[1]-tl[1])
                bars[bar] = bars.get(bar,[])+[reversed(tl),reversed(wh)]
                bar += 1
    bs = bars.keys()
    bs.sort()
    fn = os.path.join('/tmp',os.path.splitext(os.path.basename(scrImage.fn))[0]+'.txt')

    imgnr = None
    barnr = None
    imgbarfile = '/tmp/imgbar.txt'

    d = nu.loadtxt(imgbarfile)
    imgnr = int(d[0])
    barnr = int(d[1])

    assert imgnr != None
    assert barnr != None
    with open(fn,'w') as f:
        for b in bs:
            f.write('{0} {1} '.format(b+barnr,imgnr))
            for xy in bars[b]:
                for p in xy:
                    f.write('{0:d} '.format(int(nu.round(p))))
            f.write('\n')
    nu.savetxt(imgbarfile,nu.array([imgnr+1,barnr+len(bs)]))

def assignBarlinesToSystems(systems,barpoints):
    for xy in barpoints:
        assigned = False
        for system in systems:
            if system.getUpperLeft()[0] < xy[1] < system.getLowerLeft()[0]:
                system.addBarPoint(xy)
                assigned = True
                break
        if not assigned:
            print('warning, could not assign point',xy)
    for system in systems:
        system.barPoints = nu.array(system.barPoints).reshape((-1,2))




if __name__ == '__main__':
    # copied from stafffind.py
    fn = sys.argv[1]
    barfile = sys.argv[2]
    si = ScoreImage(fn)
    #si.drawImage()
    # barpoints in format: horz vert
    barpoints = nu.loadtxt(barfile)
    getNumberedBarCoords(barpoints,si)
    si.drawImage()
