#!/usr/bin/env python

import sys,getopt

def breakText(text,width,rejoin=False):
    result = []
    current = []
    for w in text.split(' '):
        if len(' '.join(current))+len(w)+1 > width:
            result.append(' '.join(current))
            current = []
        current.append(w)
    if current is not []:
        result.append(' '.join(current))
    if rejoin:
        return '\n'.join(result)
    else:
        return result

class Option:
    """
    The Option class represents a command line option, to be used as part of CommandLineHandler.
    """
    def __init__(self,shortOption=None,longOption=None,hasArgument=False,description='',defaultArg='',cb=None):
        self.shortOption = shortOption
        self.longOption = longOption
        self.hasArgument = hasArgument
        self.description = description
        self.defaultArg = defaultArg
        self.argument = None
        self.cb = cb

    def describe(self,cStart,cEnd):
        blankLinePrefix='\t'+' '*(cStart-1)
        shortopt = '-'+self.shortOption if self.shortOption else ''
        longopt = '--'+self.longOption if self.longOption else ''
        if self.hasArgument:
            longopt += '=ARG'
        result = '\t'
        if shortopt is not '':
            result += shortopt
        else:
            result += '  '
        result += ', ' if shortopt and longopt else '  '
        if longopt is not '':
            result += longopt
        if len(result) > cStart-1:
            result += '\n'+blankLinePrefix
        else:
            result += ' '*(cStart-len(result))
        defaultText = '(default: {0})'.format(self.defaultArg) if self.defaultArg is not '' else ''
        descriptionLines = breakText(self.description+' '+defaultText,cEnd-cStart)
        result += descriptionLines[0]
        if len(descriptionLines) > 0:
            for l in descriptionLines[1:]:
                result += '\n'+blankLinePrefix+l
        return result

    def getShortOpt(self):
        if self.shortOption:
            return self.shortOption+(':' if self.hasArgument else '')
        else:
            return None

    def getLongOpt(self):
        if self.longOption:
            return self.longOption+('=' if self.hasArgument else '')
        else:
            return None

class CommandLineHandler(object):
    """
    The CommandLineHandler class is built on the getopt module and facilitates 
    the use of command line options. Using the addOption you can add options, optionally
    specifying default values and callback handlers.
    """
    def __init__(self,columnStart=20,columnEnd=70,addHelpOption=True):
        self.options = []
        self.optionDict = {}
        self.optValues = {}
        self.columnStart = columnStart
        self.columnEnd = columnEnd
        self.argDescription = ['']
        self.commandName = 'COMMAND'
        self.args = {}
        self.argNames = []
        self.description = ''
        if addHelpOption:
            self.addOption('h','help',False,'Show help',cb=self.showHelp)

    def setDescription(self,d):
        self.description = d

    def setArgNames(self,l):
        self.argNames = l

    def addOption(self,shortOption=None,longOption=None,hasArgument=False,description='',defaultArg='',cb=None):
        o = Option(shortOption,longOption,hasArgument,description,defaultArg,cb)
        self.options.append(o)
        if o.shortOption:
            self.optionDict[o.shortOption] = o
        if o.longOption:
            self.optionDict[o.longOption] = o

    def getOption(self,name):
        return self.optionDict[name]

    def getOptionDefault(self,name):
        try:
            return self.optionDict[name].defaultArg
        except KeyError:
            return None
        
    def getOptionValue(self,name,orDefault=False):
        """Return the value of option as a string (or '' if the option has no value). If (and only if) 
        the option was not specified at the command line, this function returns None.
        """
        try:
            return self.optValues[self.optionDict[name]]
        except KeyError:
            return None

    def parseCommandline(self,cmdl):
        self.commandName = cmdl[0]
        shortopts = ''.join(self.getShortOpts())
        longopts = self.getLongOpts()
        try:
            opts,args = getopt.getopt(cmdl[1:],shortopts,longopts)

            for o,v in opts:
                option = self.optionDict[o.lstrip('-')]
                self.optValues[option] = v
                if option.cb:
                    option.cb(v)
                    
            if len(args) < len(self.argNames):
                print(args,self.argNames)
                sys.stdout.write('Error: not enough arguments given.\n\n')
                self.showHelp()
                sys.exit()
            elif len(args) > len(self.argNames):
                print(args,self.argNames)
                sys.stdout.write('Error: more arguments given than required.\n')
                self.showHelp()
                sys.exit()
                #N = len(self.argNames)
                #self.args = dict(zip(self.argNames[:N-2],args[:N-2]))
                #self.args[self.argNames[N-1]] = args[N-1:]
            else:
                self.args = dict(zip(self.argNames,args))


               
        except getopt.GetoptError, opt:
            sys.stderr.write('Error: {0}\n'.format(str(opt)))
            sys.stderr.write(self.usage())
            sys.exit()

    def getArgs(self,name=None):
        if name:
            return self.args[name]
        else:
            return self.args

    def usage(self):
        hasOptions = len(self.options) > 0
        usageLine = 'Usage: {0}'.format(self.commandName)
        if hasOptions:
            usageLine += ' [OPTION]...'
        if self.argDescription:
            usageLine += ' {0}'.format(' '.join(self.argNames))

        if self.description:
            usage = [breakText('Description: '+self.description,self.columnEnd,rejoin=True),'',usageLine]
        else:
            usage = [usageLine]


        if hasOptions:
            usage += ['Options:']
            for o in self.options:
                usage.append(o.describe(self.columnStart,self.columnEnd)+'\n')
        return '\n'.join(usage)+'\n'
 
    def getShortOpts(self):
        return [o.getShortOpt() for o in self.options if o.shortOption]
    def getLongOpts(self):
        return [o.getLongOpt() for o in self.options if o.longOption]
    
    def showHelp(self,*args):
        sys.stdout.write(self.usage())
        # if user needs help, don't execute any further:
        sys.exit()

if __name__ == '__main__':
    cmdh = CommandLineHandler()
    cmdh.addOption('v','verbose',hasArgument=False,description="""Verbose""")
    cmdh.addOption('h','help',False,'Show help',cb=cmdh.showHelp)
    cmdh.addOption('o','output-file',True,'Save output to file ARG','/tmp/out.txt')
    cmdh.addOption('t','test-option',False,'Try out')
    cmdh.parseCommandline(sys.argv)

    print(cmdh.getOptionValue('output-file'),cmdh.getOptionValue('output-file')==None)
    #print(cmdh.getOptionValue('t')==None)
    #print(cmdh.getArgs())

