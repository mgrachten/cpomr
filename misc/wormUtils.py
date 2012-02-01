from numpy import array
import utilities,re

class WormFile:
    def __init__(self,wormfile=None):
        if wormfile:
            self.readFile(wormfile)
    def readFile(self,filename):
        lines = utilities.readFile(filename)
        headerLinesPat = re.compile('(?P<attribute>[^:]+):(?P<value>.+)$',re.I)
        i=0
        while headerLinesPat.match(lines[i]):
            i+=1 
        self.header = dict([(k.group('attribute').strip(),
                             (j,utilities.interpretFieldRational(k.group('value').strip())))
                       for j,k in zip(range(i),[headerLinesPat.match(l) for l in lines[:i]])])
        self.data = array([[float(l[0]),float(l[1]),float(l[2]),int(l[3])] for l in
                      [k.split() for k in lines[i:]]])
    def writeFile(self,filename):
        result = []
        headerInfo = self.getHeader().items()
        headerInfo.sort(key=lambda x: x[1][0])
        for k,v in headerInfo:
            result.append('%s: %s' % (k,str(v[1])))
        data = self.getData()
        for i in range(data.shape[0]):
            d = data[i,:]
            result.append('%f %f %f %d' % (d[0],d[1],d[2],d[3]))
        utilities.writeFile(filename,'\n'.join(result))
    def getHeader(self):
        return self.header
    def getHeaderField(self,field):
        return self.getHeader()[field][1]
    def getData(self):
        return self.data
    def setHeader(self,header):
        self.header = header
    def setHeaderField(self,field,value):
        self.getHeader()[field] = value
    def setData(self,data):
        self.data = data
    
