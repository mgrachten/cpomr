import utilities
import sys

class ArffFileReader:
    """Object for reading in an Arff file and returning relation, attribute, and instance data"""
    def __init__(self,filename):
        lines = utilities.readFile(filename)
        headerLines,bodyLines = self.splitHeaderAndBody(lines)
        self.instances = []
        self.relation = headerLines[0].split()[1].strip('\'"')
        self.attributes = self.getAttributesFromHeaderLines(headerLines)
        self.readInstances(bodyLines)

    def interpretNUMERIC(self,s):
        try:
            return int(s)
        except:
            try:
                return float(s)
            except:
                print('Warning: cannot interpet %s as NUMERIC' % s)
                sys.exit()

    def readAttributeValueFromLine(self,attribute,line):
        substring = ''
        line = line.lstrip()
        if len(line) > 0 and line[0] == '?':
            return (None, line[line.find(',')+1:])
        elif isinstance(attribute[1],list):
            for value in attribute[1]:
                if line.startswith(value):
                    return (value,line[len(value)+1:])
            print('Warning, instance is not compatible with header')
            sys.exit()
        elif attribute[1].upper() == 'NUMERIC':
            i = line.find(',')
            if i < 0:
                substring = line
            else:
                substring = line[:i]
            #if substring.strip() == '?':
            #    return (None, line[i+1:])
            #else:
            return (self.interpretNUMERIC(substring), line[i+1:])
        elif attribute[1].upper() == 'STRING':
            if len(line) > 0:
                if line[0] == '\'':
                    line = line[1:]
                    i = line.find('\'')
                elif line[0] == '"':
                    line = line[1:]
                    i = line.find('"')
                else:
                    i = line.find(',')
                if i < 0:
                    return (line,'')
                else: 
                    return (line[:i],line[i:])
        else:
            print('Warning, unknown attribute type %s' % attribute[1])
            sys.exit()

    def readInstances(self,lines):
        self.instances = []
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                instance = []
                for a in self.attributes:
                    v,line = self.readAttributeValueFromLine(a,line)
                    instance.append(v)
                self.instances.append(instance)

    def getRelation(self):
        return self.relation

    def getInstances(self):
        return self.instances

    def getAttributes(self):
        return self.attributes

    def splitHeaderAndBody(self,lines):
        """Split lines of ARFF file into header and body"""
        headerLines = []
        separator = ','
        while lines[0].upper().find('@DATA') < 0:
            headerLines.append(lines[0])
            del lines[0]
        del lines[0]
        while lines[0] == '\n':
            del lines[0]
        return [headerLines,lines]

    def readClassValues(self,string):
        assert(string[0] == '{')
        i = string.find('}')
        if i < 0:
            print('Warning, no delimiting "}" found in CLASS description')
            sys.exit()
        else:
            return [x.strip() for x in string[1:i].split(',')]

    def getAttributesFromHeaderLines(self,headerLines):
        """Return the attributes and their types in headerLines
        (as defined by @attribute) as a list of pairs"""
        headerLines = [l.strip() for l in headerLines if l.lstrip().upper().startswith('@ATTRIBUTE')]
        attributes = []
        for line in headerLines:
            line = line[len('@ATTRIBUTE'):].lstrip()
            attName = line.strip().split()[0]
            attType = line[len(attName)+1:].lstrip()
            if attType.upper() in ('NUMERIC','STRING'):
                attributes.append([attName,attType.upper()])
            else:
                values = self.readClassValues(attType)
                attributes.append([attName,values])
        return attributes

class ArffObject:
    def __init__(self,relation='',attributes=None,instances=None):
        self.relation = relation
        if instances is not None:
            self.instances = instances
        else:
            self.instances = []
        if attributes is not None:            
            self.attributes = self.completeAttributes(attributes)
        else: 
            self.attributes = []

    def readFromFile(self,filename):
        afr = ArffFileReader(filename)
        self.relation = afr.getRelation()
        self.attributes = afr.getAttributes()
        self.instances = afr.getInstances()
        del afr

    def swapAttributes(self,a1,a2):
        i1 = self.getAttributeNames().index(a1)
        i2 = self.getAttributeNames().index(a2)
        a3 = self.attributes[i1]
        self.attributes[i1] = self.attributes[i2]
        self.attributes[i2] = a3
        d3 = self.instances[i1]
        self.instances[i1] = self.instances[i2]
        self.instances[i2] = d3

    def removeAttribute(self,attribute):
        i = self.getIndexOfAttribute(attribute)
        if i is not None:
            del self.attributes[i]
            for instance in self.instances:
                del instance[i]

    def appendAttribute(self,attribute,values=[]):
        if len(self.attributes) == 0:
            # no prior attributes
            #print(len(values),len(self.instances))
            for i,v in enumerate(values[:]):
                self.instances.append([v])
        else:
            print(len(values),len(self.instances))
            assert len(values) == len(self.instances), 'warning: nr of instances of new attribute not equal to prior nr of instances'
            for i,v in enumerate(values[:]):
                self.instances[i].append(v)
        assert len(attribute) == 2, 'Warning, attribute description should be (\'NAME\',\'DATATYPE\')'
        self.attributes.append(attribute)
    
    def writeToFile(self,filename,mode='data'):
        utilities.writeFile(filename,self.getFormattedData(mode))

        
    def getRelation(self):
        return self.relation
    
    def getInstance(self,i,attribute=None):
        return self.instances[i]

    def getInstances(self):
        return self.instances

    def getAttributes(self):
        return self.attributes

    def getAttributeNames(self):
        return [i[0] for i in self.getAttributes()]
    
    def getColumnByName(self,attributeName):
        try:
            return self.getColumn(self.getAttributeNames().index(attributeName))
        except IndexError:
            sys.stderr.write("Error: can't find attribute name: %s\n" % attributeName)
            return None
            
    def getColumn(self,i):
        def getValueFromRow(r,i):
            try:
                return r[i]
            except IndexError:
                return None
        #return [getValueFromRow(r,i) for r in self.instances]
        return [r[i] for r in self.instances]

    def getFormattedData(self,mode='arff'):
        return self.formatHeader(mode)+self.formatInstances(mode)

    def formatAttribute(self,attribute):
        result = '@ATTRIBUTE %s' % (attribute[0]) 
        if isinstance(attribute[1],str):
            return result + ' %s\n' % attribute[1]
        else:
            return result + ' {' + ','.join([str(i) for i in attribute[1]]) + '}\n'

    def formatHeader(self,mode='arff'):
        if mode == 'arff':
            return '@RELATION \'%s\'\n\n' % (self.getRelation()) +\
                ''.join([self.formatAttribute(attribute) for
                         attribute in self.getAttributes()]) + '\n'
        if mode == 'data':
            return '# ' + ' '.join([a[0] for a in self.getAttributes()]) + '\n'

    def formatInstances(self,mode='arff'):
        #def formatInstance(instance):
        #    return ','.join(['?' if i == None else str(i) for i in instance])+'\n'
        def formatInstanceArff(instance):
            r = []
            for i,a in enumerate(self.attributes):
                if instance[i] == None:
                    r.append('?')
                elif a[1] == 'STRING':
                    r.append('\'%s\'' % instance[i])
                elif a[1] == 'NUMERIC':
                    r.append(str(instance[i]))
                else:
                    r.append('%s' % instance[i])
            return ','.join(r)+'\n'
        def formatInstanceData(instance,sep=' ',default='X'):
            r = []
            for i,a in enumerate(self.attributes):
                if instance[i] == None:
                    r.append(default)
                elif a[1] == 'STRING':
                    r.append('\'%s\'' % instance[i])
                elif a[1] == 'NUMERIC':
                    r.append(str(instance[i]))
                else:
                    r.append('%s' % instance[i])
            return sep.join(r)+'\n'
        if mode == 'arff':
            return '@DATA\n' +\
                ''.join([formatInstanceArff(i) for i in self.getInstances()])
        if mode == 'data':
            return ''.join([formatInstanceData(i) for i in self.getInstances()])


    def getAttributeOfInstance(self,attributeName,i):
        try:
            return self.instances[i][self.getAttributeNames().index(attributeName)]
        except IndexError:
            sys.stderr.write("Error: can't find attribute name: %s\n" % attributeName)
            return None

    def isValidAttributeDescription(self,a):
        if (isinstance(a,list) or isinstance(a,tuple)) and (len(a) > 1):
            if isinstance(a[1],str):
                if a[1].upper() in ('NUMERIC','STRING'):
                    return True
            elif isinstance(a[1],list) or isinstance(a[1],tuple):
                return True
        return False

    def completeAttributes(self,attributes):
        newAttributes = attributes[:]
        for i,a in enumerate(newAttributes):
            if not self.isValidAttributeDescription(a):
                columnData = [r for r in self.getColumn(i) if not r == None]
                dataType = self.determineDataType(columnData)
                if dataType == 'class':
                    nominalValues = set(columnData)
                    newAttributes[i] = (str(a),nominalValues)
                else:
                    newAttributes[i] = (str(a),dataType)
        return newAttributes

    def getIndexOfAttribute(self,attributeName):
        try:
            return self.getAttributeNames().index(attributeName)
        except ValueError:
            return None

    def getMatchingInstances(self,attributeName,test):
        k = self.getIndexOfAttribute(attributeName)
        return [i for i in self.instances if test(i[k])]

    def determineDataType(self,dataColumn):
        for i in dataColumn:
            dataType = type(utilities.interpretField(i))
            if dataType == str:
                return 'class'
            elif dataType in (int,float):
                return 'numeric'
        print 'Warning, did not find any values in data column'
        return None

