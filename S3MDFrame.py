'''
Created on Jun 25, 2010


@author: MarkDBenedict
'''
import matplotlib
matplotlib.use('WXAgg')
#import matplotlib.pyplot as plt

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

import wx
import time
import os
import math
import subprocess
import scipy.misc.pilutil as pilutil
import Image #PIL image library
from numpy import *


class PlotPanel (wx.Panel):
    """The PlotPanel has a Figure and a Canvas. OnSize events simply set a 
flag, and the actual resizing of the figure is triggered by an Idle event."""
    def __init__( self, parent, color=None, dpi=None, **kwargs ):
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from matplotlib.figure import Figure
        self.parent=parent
        wx.Panel.__init__( self, parent, **kwargs )

        # initialize matplotlib stuf
        
        self.figure = Figure( None, dpi )
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.SetColor( color )

        self._SetSize()
        self.x=[]
        self.y=[]
        #self.draw()

        self._resizeflag = False

        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

    def SetColor( self, rgbtuple=None ):
        """Set figure and canvas colours to be the same."""
        if rgbtuple is None:
            rgbtuple = wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ).Get()
        clr = [c/255. for c in rgbtuple]
        self.figure.set_facecolor( clr )
        self.figure.set_edgecolor( clr )
        self.canvas.SetBackgroundColour( wx.Colour( *rgbtuple ) )

    def _onSize( self, event ):
        self._resizeflag = True
        #print 'told to resize'
        self._SetSize()

    def _onIdle( self, evt ):
        if self._resizeflag:
            self._resizeflag = False
            #print 'told to resize'
            self._SetSize()

    def _SetSize( self ):
        pixels = self.GetClientSize() #tuple( self.parent.GetClientSize() )
        self.SetSize( pixels )
        self.canvas.SetSize( pixels )
        self.figure.set_size_inches( float( pixels[0] )/self.figure.get_dpi(),
                                     float( pixels[1] )/self.figure.get_dpi() )

    def draw(self):
       
        if self.x != None and self.y !=None:
            self.figure.subplots_adjust(left=0.125)
            self.axes.plot(self.x[0][1:],self.y[0][1:], 'rv',linestyle='--',linewidth=4.0)
            self.axes.plot(self.x[1][1:],self.y[1][1:], 'bo',linestyle=':',linewidth=2.0)
            self.axes.plot(self.x[2][1:],self.y[2][1:], 'g.',linestyle='-',linewidth=1.0)
            self.axes.legend(('Base','ANN','ANN-GPU'),loc='lower right',fancybox=True)
            self.axes.set_xlabel('Timestep')
            self.axes.set_ylabel=('Pressure')
            self.axes.set_xlim((0,3000))
            self.axes.set_ylim((0,-2.0))
            #ticks=[0.98*self.y[2].max(),self.y[2].max(),1.02*self.y[2].max()]
            #self.axes.set_ylim((ticks[0],ticks[2]))
            #strLabels=['%.3f'%(0.98*ticks[0]),
            #           '%.3f'%ticks[1],
            #           '%.3f'%(1.02*ticks[2])]
            #self.axes.set_yticklabels(strLabels)
            #self.axes.set_yticks(ticks)
            self.canvas.draw()
            
class BarPanel (wx.Panel):
    """The BarPanel has a Figure and a Canvas. OnSize events simply set a flag,
    and the actual resizing of the figure is triggered by an Idle event."""
    def __init__( self, parent, color=None, dpi=None, **kwargs ):
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from matplotlib.figure import Figure
        self.parent=parent
        wx.Panel.__init__( self, parent, **kwargs )

        # initialize matplotlib stuf
        
        self.figure = Figure( None, dpi )
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.SetColor( color )

        self._SetSize()
        self.speedup1=[1.,1.,1.,1.,1.,1.,1.] #used for rolling average
        self.speedup2=[1.,1.,1.,1.,1.,1.,1.]
        self._resizeflag = False

        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

    def SetColor( self, rgbtuple=None ):
        """Set figure and canvas colours to be the same."""
        if rgbtuple is None:
            rgbtuple = wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ).Get()
        clr = [c/255. for c in rgbtuple]
        self.figure.set_facecolor( clr )
        self.figure.set_edgecolor( clr )
        self.canvas.SetBackgroundColour( wx.Colour( *rgbtuple ) )

    def _onSize( self, event ):
        self._resizeflag = True
        #print 'told to resize'
        self._SetSize()

    def _onIdle( self, evt ):
        if self._resizeflag:
            self._resizeflag = False
            #print 'told to resize'
            self._SetSize()

    def _SetSize( self ):
        pixels = self.GetClientSize() #tuple( self.parent.GetClientSize() )
        self.SetSize( pixels )
        self.canvas.SetSize( pixels )
        self.figure.set_size_inches( float( pixels[0] )/self.figure.get_dpi(),
                                     float( pixels[1] )/self.figure.get_dpi() )
    def setData(self,inValList):
        tempVals=inValList
        if tempVals[0]==0:
            tempVals[0]=1
        
        val1=tempVals[1]/float(tempVals[0])
        val2=tempVals[2]/float(tempVals[0])
        del self.speedup1[0]
        self.speedup1.append(val1)   
        del self.speedup2[0]
        self.speedup2.append(val2)
    
    def draw(self):
        self.axes.clear()
        self.figure.subplots_adjust(top=0.83)
        self.axes.bar([1,2,3],[1.0,mean(self.speedup1),mean(self.speedup2)],color=['r','b','g'])
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_xticklabels([])
        self.axes.set_yticklabels([])
        self.axes.set_title('Speedup Relative to Base')
        self.axes.text(1.3,0.5,'1x')
        speedStr1 = '%.1fx'%mean(self.speedup1)
        speedStr2 = '%.1fx'%mean(self.speedup2)
        self.axes.text(2.15,mean(self.speedup1)/2.0,speedStr1)
        self.axes.text(3.15,mean(self.speedup2)/2.0,speedStr2)
        self.canvas.draw()
   
from threading import Thread
class SimulationWorker(Thread):
    """Worker Thread Class."""
    def __init__(self, notify_window,processName):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.processName = processName
       
    def run(self):
        """Run Worker Thread."""
        inputName='%s'%self.processName
        fileName='testData%s.txt'%self.processName
        exeName='./%s'%self.processName
        print exeName
        fileptr=open(fileName,'w')
        
        self.p=subprocess.Popen([exeName,inputName],executable=exeName,stdout=fileptr)

    def abort(self):
        """abort worker thread."""
        # Method for use by main thread to signal an abort
        if hasattr(self,'p'):
            self.p.kill()

class S3MDFrame(wx.Frame):
    def __init__(self, parent, title):
        self.counter=0
        wx.Frame.__init__(self, parent, title=title, size=(450, 700))
        self.control = wx.TextCtrl(self, style = wx.TE_MULTILINE,size=(450,65))
        
        self.simulationWorkers=None
        self.checkUpdates=False
        self.lastUpdateTime=time.time()
         # Set up event handler for any worker thread results
        self.Temperature=0
        self.Density=0
        
        ###loading an image
        fileList = os.listdir(os.getcwd())
        imageList=[]
        typeList=['.bmp','.png']
        for name in fileList:
            if name[-4:] in typeList:
                imageList.append(name)
        ###
        
        
        ###make the option bar across top
        self.CreateStatusBar()
        ###       
        
        self.Bind(wx.EVT_IDLE, self.onIdle)
        
        ###make the file and help menus with different options        
        filemenu = wx.Menu()
        helpmenu = wx.Menu()
        
        menuOpen = filemenu.Append(wx.ID_OPEN, "Open", "Open an existing program.")
        self.Bind(wx.EVT_MENU, self.OnOpen, menuOpen)

        menuSave = filemenu.Append(wx.ID_SAVE, "Save", "Save current program.")
        self.Bind(wx.EVT_MENU, self.OnSave, menuSave)
        
        menuClose = filemenu.Append(wx.ID_CLOSE, "Close", "Close current window.")
        self.Bind(wx.EVT_MENU, self.OnClose, menuClose)

        helpmenu.Append(wx.ID_HELP_SEARCH, "Program Help")
                        
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "File")
        
        self.SetMenuBar(menuBar)
        ###
        
        
        ###put in panels
        #panel = wx.Panel(self, -1)# size = (450,450), style = wx.SUNKEN_BORDER)
        self.windTunnelPanel = PlotPanel(self)
        self.barPanel = BarPanel(self)
        self.controlPanel = wx.Panel(self, -1, style = wx.SUNKEN_BORDER)    
        
            
        panelSizer = wx.BoxSizer(wx.VERTICAL)
        panelSizer.Add(self.windTunnelPanel, 5, wx.EXPAND)
        
        pieSizer = wx.BoxSizer(wx.HORIZONTAL)
        pieSizer.Add(self.control, 1, wx.EXPAND)
        pieSizer.Add(self.barPanel, 1, wx.EXPAND)
        
        panelSizer.Add(pieSizer, 3, wx.EXPAND)
        panelSizer.Add(self.controlPanel, 1, wx.EXPAND)
            
        self.SetAutoLayout(True)
        self.SetSizer(panelSizer)
        self.Layout()
                       
        #self.windTunnelPanel = PlotPanel(panel)
        ###
        
        
        ###put in buttons    
        self.playbutton = wx.Button(self.controlPanel, label = "Run")
        self.Bind(wx.EVT_BUTTON, self.OnClickPlay, self.playbutton)
        
        self.idlebutton = wx.Button(self.controlPanel, label = "Idle")
        self.Bind(wx.EVT_BUTTON, self.OnClickIdle, self.idlebutton)
        
        self.pausebutton = wx.Button(self.controlPanel, label = "Stop")
        self.Bind(wx.EVT_BUTTON, self.OnClickStop, self.pausebutton)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer.Add(self.playbutton, 1, wx.ALL, 15)
        hsizer.Add(self.idlebutton, 1, wx.ALL, 15)
        hsizer.Add(self.pausebutton, 1, wx.ALL, 15)
        ###
        
        
        
        ###put in sliders
        #self.velocityslider = wx.Slider(self.controlPanel, minValue = 0, maxValue = 100, 
        #                                style = wx.SL_AUTOTICKS | wx.SL_LABELS)        
        #velocityLabel = wx.StaticText(self.controlPanel, -1, "Density")
        #self.Bind(wx.EVT_SLIDER, self.OnSlideVelocity, self.velocityslider)        
        #
        #self.tempslider = wx.Slider(self.controlPanel, minValue = 0, maxValue = 2000, 
        #                            style = wx.SL_AUTOTICKS | wx.SL_LABELS)
        #tempLabel = wx.StaticText(self.controlPanel, -1, "Temperature (K)")
        #self.Bind(wx.EVT_SLIDER, self.OnSlideTemp, self.tempslider)
        #
        #
        #tempVSizer = wx.BoxSizer(wx.VERTICAL)
        #tempVSizer2 = wx.BoxSizer(wx.VERTICAL)
        #tempHSizer = wx.BoxSizer(wx.HORIZONTAL)
        #
        #tempVSizer.Add(velocityLabel, 1, wx.ALL, 5)
        #tempVSizer.Add(self.velocityslider, 3, wx.ALL, 5)
        
        #tempVSizer2.Add(tempLabel, 1, wx.ALL, 5)
        #tempVSizer2.Add(self.tempslider, 3, wx.ALL, 5)
        
        #tempHSizer.Add(tempVSizer, 1, wx.EXPAND)
        #tempHSizer.Add(tempVSizer2, 1, wx.EXPAND)
        ###
        
        ###putting the sizers in place
        #vsizer.Add(tempHSizer, 3, wx.EXPAND)
        vsizer.Add(hsizer, 1, wx.EXPAND)
        self.controlPanel.SetSizer(vsizer)
        self.SetAutoLayout(True)
        vsizer.Fit(self.controlPanel)
        ###
        
        self.controlPanel.SetBackgroundColour("GRAY")
        self.control.SetBackgroundColour("GRAY")
        self.Show(True)
    
    def readRDF(self,fileName):
        rdfFile=open(fileName,'r')
        rdfData=rdfFile.readlines()
        rdfFile.close()
        rdfData.reverse()
        
        data=[] #list of rdfs
        currRdf=[] #rdf at a single point in time
        
        while len(rdfData)>0:
            item = rdfData.pop()
            if 'rdf' in item:
                data.append(array(currRdf))
                currRdf=[]
            else:
                line=item.split()
                currRdf.append((float(line[0]),float(line[1])))
        return data
    
    def onIdle(self,event):
        if self.checkUpdates==True:
            if time.time()-self.lastUpdateTime > 0.5 : 
                self.counter+=1
                in1=open('testDataorigMD.txt','r')
                data1=in1.readlines()
                in1.close()
                if len(data1)>0:
                    A=array( [ [float(line.split()[0]),float(line.split()[3])+float(line.split()[5])] for line in data1])
                else:
                    A=zeros((1,2))
                in2=open('testDataANNMD.txt','r')
                data2=in2.readlines()
                in2.close()
                if len(data2)>0:
                    B=array( [ [float(line.split()[0]),float(line.split()[3])+float(line.split()[5])] for line in data2])
                else:
                    B=zeros((1,2))
                in3=open('testDataGPUANNMD.txt','r')
                data3=in3.readlines()
                in3.close()
                if len(data3)>0:
                    C=array( [ [float(line.split()[0]),float(line.split()[3])+float(line.split()[5])] for line in data3])
                else:
                    C=zeros((1,2))
                self.windTunnelPanel.x=[A[:,0],B[:,0],C[:,0]]
                self.windTunnelPanel.y=[A[:,1],B[:,1],C[:,1]]
                self.barPanel.setData([A[-1,0],B[-1,0],C[-1,0]])
                print 'Idling %d'%self.counter
                self.windTunnelPanel.draw()
                self.barPanel.draw()
                self.lastUpdateTime=time.time()
                wx.Yield()
            

            
    ###defining events on the menu              
    def OnOpen(self, e):       
        dlg = wx.FileDialog(self, message = "Choose a file", defaultDir = os.getcwd(), 
                            defaultFile = "", style = wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPath() 
            fp = open(paths, "r")
            
            theString = fp.read()
            theString = theString.split("\n")
            data = []
            for line in theString[:-1]:
                thePoint = line.split()
                data.append((int(thePoint[0]), int(thePoint[1])))
            self.windTunnelPanel.points = data
            self.windTunnelPanel.Refresh(False)
        dlg.Destroy()                
        self.control.AppendText("Opened \n")


    def OnSave(self, e):       
        dlg = wx.FileDialog(self, message = "Save file as...", defaultDir = os.getcwd(), 
                            defaultFile = "", style = wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            fp = file(path, "w")
            data = self.windTunnelPanel.points
            for point in data:
                theString = str(point[0])+" "+str(point[1])
                fp.write(theString + "\n")           
            fp.close()
        dlg.Destroy()
        self.control.AppendText("Saved \n")
 
     
    def OnClose(self, e):
        self.OnClickPause(e)
        self.Close(True)
     
    ###defining events on the control panel
    def EvtComboBox(self, e):
        self.control.AppendText("EventComboBox Chosen: %s\n" %e.GetString())
        chosen = e.GetString()
        print chosen
        theImage=wx.Image(chosen)
        if self.windTunnelPanel.bitmap!=None:
            self.windTunnelPanel.bitmap.Destroy()
        self.windTunnelPanel.bitmap=wx.BitmapFromImage(theImage)
       
        self.windTunnelPanel.reInitBuffer=True
        self.Refresh()
    
        
    def OnClickPlay(self, e):
        self.control.AppendText("Clicked on Run button\n")
        self.controlPanel.SetBackgroundColour("YELLOW")
        self.checkUpdates=True
        #run simulation in seperate thread
        if self.simulationWorkers!=None:
            del self.simulationWorkers
            
        self.simulationWorkers = [
            SimulationWorker(self,'origMD'),
            SimulationWorker(self,'ANNMD'),
            SimulationWorker(self,'GPUANNMD')
        ]
        self.simulationWorkers[0].start()
        self.simulationWorkers[1].start()
        self.simulationWorkers[2].start()
            
    def OnClickStop(self, e):
        self.control.AppendText("Clicked on Stop button\n")
        self.controlPanel.SetBackgroundColour("GREEN")
        self.checkUpdates=False
        if self.simulationWorkers!=None:
            for worker in self.simulationWorkers:
                worker.abort()
                
    def OnClickIdle(self, e):
        self.control.AppendText("Clicked on Idle button\n")
        self.controlPanel.SetBackgroundColour("GRAY")
        self.checkUpdates=False
        if self.simulationWorkers!=None:
            for worker in self.simulationWorkers:
                worker.abort()
                
    def OnSlideVelocity(self, e):
        self.control.AppendText("Velocity Slider was moved %d\n" %e.GetInt())
        self.Velocity = e.GetInt()
    
        
    def OnSlideTemp(self, e):
        self.control.AppendText("Temperature Slider was moved %d\n" %e.GetInt())
        self.Temperature = e.GetInt()
    

        



   
app = wx.App(False)
frame = S3MDFrame(None, "Tech Demo")
app.MainLoop()