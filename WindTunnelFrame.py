'''
Created on Jun 25, 2010


@author: sarahjackson
'''

import wx
import os
import math

import scipy.misc.pilutil as pilutil
import Image #PIL image library

class BezierPanel(wx.Panel):
    
    def __init__(self, parent, inBitmap = None):
        wx.Panel.__init__(self, parent, size = parent.GetClientSize(), 
                          style=wx.NO_FULL_REPAINT_ON_RESIZE)
        self.bitmap = inBitmap
        self.reInitBuffer = True
        self.initBuffer()
        self.initDrawing()
        self.bindEvents()
    
    def setBitmap(self, inBitmap):
        self.bitmap = inBitmap
        dc = wx.BufferedDC(None, self.drawBuffer)
        if self.bitmap != None: #use bitmap as background of panel
            dc.DrawBitmap(self.bitmap, 0, 0)
            self.drawPoints(dc,self.points)
        self.Refresh(True)    


    def initDrawing(self):
        self.points = [(75, 225), (175, 185), (275, 185), (375, 215), 
                       (375, 235), (275, 265), (175, 265)]

        self.PointSize=4
        dc = wx.BufferedDC(wx.ClientDC(self), self.drawBuffer)
        dc.BeginDrawing()
        self.drawPoints(dc, self.points)
        dc.EndDrawing()
        self.currentPoint=None
    
    def initBuffer(self):
        self.reInitBuffer=False
        self.SetBackgroundColour("WHITE")
        parent = self.GetParent()
        psize = parent.GetClientSize()
        self.SetSize(psize)
        
#        print 'init memory buffer to ', psize.width, ",", psize.height
        self.drawBuffer = wx.EmptyBitmap(psize.width, psize.height)
        dc=wx.BufferedDC(None, self.drawBuffer)
        
        if self.bitmap != None: #use bitmap as background of panel
            dc.DrawBitmap(self.bitmap, 0, 0)
            self.drawPoints(dc,self.points)
        else:  #use background color to clear panel
            dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
            dc.Clear()
        return
    
    def bindEvents(self): 
        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.Bind(wx.EVT_MOTION, self.onMotion)
        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_IDLE, self.onIdle)
        return

    def drawPoints(self, dc, points,drawControlPoints=True):
        pen = wx.Pen(wx.NamedColour("PURPLE"), 1, wx.SOLID)
        brush = wx.Brush(wx.NamedColour("PURPLE"))
        dc.SetPen(pen)
        dc.SetBrush(brush)
        
        if drawControlPoints:
            for point in points:
                dc.DrawCircle(point[0], point[1], self.PointSize)
            
        pen2 = wx.Pen(wx.NamedColour("BLACK"), 2, wx.SOLID)
        dc.SetPen(pen2)
        brush2 = wx.Brush(wx.NamedColour("BLACK"))
        brush2.SetStyle(wx.SOLID)        
        dc.SetBrush(brush2) 
        
        tempList=self.points[:]
        tempList.append(self.points[0])
        dc.DrawSpline(tempList)
#        dc.FloodFillPoint(self.points[0], "PURPLE", wx.FLOOD_BORDER)

    def GetWing(self):
        dc=wx.BufferedPaintDC(self, self.drawBuffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        self.drawPoints(dc,self.points,False)
        theImage = self.drawBuffer.ConvertToImage()
        pil = Image.new('RGB', (theImage.GetWidth(), theImage.GetHeight()))
        pil.fromstring(theImage.GetData())
        theImageArray = pilutil.fromimage(pil,flatten=True)
        return theImageArray
        
        
#Event Handlers
    #identify selected point
    def onLeftDown(self, event):
        pos = event.GetPositionTuple()
        for i, point in enumerate(self.points):
            r = math.sqrt((pos[0]-point[0])**2+(pos[1]-point[1])**2)
            if r < self.PointSize:  #cursor selected this point
                self.currentPoint = i
#                print i
        self.CaptureMouse()
        return
    
    
    def onLeftUp(self, event):
        self.currentPoint = None
        if self.HasCapture():
            self.ReleaseMouse()
            dc = wx.BufferedDC(wx.ClientDC(self), self.drawBuffer)
            if self.bitmap != None: #use bitmap as background of panel
                dc.DrawBitmap(self.bitmap, 0, 0)
            else:  #use background color to clear panel
                dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
                dc.Clear()
            self.drawPoints(dc, self.points)
            dc.EndDrawing()
        return
    
    #move the pos tuple of current point
    def onMotion(self, event):
        if event.Dragging() and event.LeftIsDown() and self.currentPoint != None:   
            dc = wx.BufferedDC(wx.ClientDC(self), self.drawBuffer)
            pos = event.GetPositionTuple()
            oldPos = self.points[self.currentPoint] 
            pen = wx.Pen(wx.NamedColour("White"), 1, wx.SOLID)
            brush = wx.Brush(wx.NamedColour("White"))
            dc.SetPen(pen)
            dc.SetBrush(brush)
            self.points[self.currentPoint] = pos
            
            if self.bitmap != None: #use bitmap as background of panel
                dc.DrawBitmap(self.bitmap, 0, 0)
                self.drawPoints(dc, self.points)
            else:  #use background color to clear panel
                dc.DrawCircle(oldPos[0], oldPos[1], self.PointSize)#clear old point
                self.drawPoints(dc, [pos])
        return

    def onPaint(self,event):
        dc=wx.BufferedPaintDC(self, self.drawBuffer)
        
    def onIdle(self,event):
        if self.reInitBuffer: #panel was resized
            self.initBuffer()
            self.reInitBuffer = False
            self.Refresh(False)
            
    def onExit(self,event):
        return

    def onSize(self,event):
        print 'told to resize'
        self.reInitBuffer = True
        return



import StableFluidsCython

class WindTunnelFrame(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(450, 700))
        self.control = wx.TextCtrl(self, style = wx.TE_MULTILINE,size=(450,65))
        self.control.SetBackgroundColour(wx.GREEN)
        
        self.Temperature=0
        self.Velocity=0
        
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
        panel = wx.Panel(self, -1, size = (450,450), style = wx.SUNKEN_BORDER)   
        self.controlPanel = wx.Panel(self, -1, style = wx.SUNKEN_BORDER)    
        self.controlPanel.SetBackgroundColour("YELLOW")
            
        panelSizer = wx.BoxSizer(wx.VERTICAL)
        panelSizer.Add(panel)#, 1, wx.EXPAND)
        panelSizer.Add(self.control)#, 1, wx.EXPAND)
        panelSizer.Add(self.controlPanel, 1, wx.EXPAND)
            
        self.SetAutoLayout(True)
        self.SetSizer(panelSizer)
        self.Layout()
                       
        self.windTunnelPanel = BezierPanel(panel)
        ###
        
        
        ###put in buttons    
        self.playbutton = wx.Button(self.controlPanel, label = "Play")
        self.Bind(wx.EVT_BUTTON, self.OnClickPlay, self.playbutton)
        
        self.pausebutton = wx.Button(self.controlPanel, label = "Pause")
        self.Bind(wx.EVT_BUTTON, self.OnClickPause, self.pausebutton)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer.Add(self.playbutton, 1, wx.ALL, 5)
        hsizer.Add(self.pausebutton, 1, wx.ALL, 5)
        ###
        
        
        
        ###put in sliders
        self.velocityslider = wx.Slider(self.controlPanel, minValue = 0, maxValue = 200, 
                                        style = wx.SL_AUTOTICKS | wx.SL_LABELS)        
        velocityLabel = wx.StaticText(self.controlPanel, -1, "Wind Velocity")
        self.Bind(wx.EVT_SLIDER, self.OnSlideVelocity, self.velocityslider)        
        
        self.tempslider = wx.Slider(self.controlPanel, minValue = -25, maxValue = 25, 
                                    style = wx.SL_AUTOTICKS | wx.SL_LABELS)
        tempLabel = wx.StaticText(self.controlPanel, -1, "Temperature")
        self.Bind(wx.EVT_SLIDER, self.OnSlideTemp, self.tempslider)
        
        
        tempVSizer = wx.BoxSizer(wx.VERTICAL)
        tempVSizer2 = wx.BoxSizer(wx.VERTICAL)
        tempHSizer = wx.BoxSizer(wx.HORIZONTAL)
        
        tempVSizer.Add(velocityLabel, 1, wx.ALL, 5)
        tempVSizer.Add(self.velocityslider, 3, wx.ALL, 5)
        
        tempVSizer2.Add(tempLabel, 1, wx.ALL, 5)
        tempVSizer2.Add(self.tempslider, 3, wx.ALL, 5)
        
        tempHSizer.Add(tempVSizer, 1, wx.EXPAND)
        tempHSizer.Add(tempVSizer2, 1, wx.EXPAND)
        ###
        
        
        ###put in the combobox
        self.combobox = wx.ComboBox(self.controlPanel, size=(125, -1), 
                                    choices = imageList, style = wx.CB_DROPDOWN)
        self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox, self.combobox)
        hsizer.Add(self.combobox, 1, wx.ALL,5)
        ###
        
        
        ###putting the sizers in place
        vsizer.Add(tempHSizer, 3, wx.EXPAND)
        vsizer.Add(hsizer, 1, wx.EXPAND)
        self.controlPanel.SetSizer(vsizer)
        self.SetAutoLayout(True)
        vsizer.Fit(self.controlPanel)
        ###
        
             
        self.Show(True)
    
    
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
#            test = ["cat", "dog", "bird", "pig", "cow"]
#            for animal in test:
#                fp.write(animal + "\n")
            fp.close()
        dlg.Destroy()
        self.control.AppendText("Saved \n")
 
     
    def OnClose(self, e):
        self.Close(True)
    ###
    
     
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
        self.control.AppendText("Clicked on Play button\n")
        theWing=self.windTunnelPanel.GetWing()
        result = StableFluidsCython.mainloop(theWing,self.Velocity,self.Temperature)        
        #redraw windTunnelPanel with results
        theImage=wx.Image(result)
        if self.windTunnelPanel.bitmap!=None:
            self.windTunnelPanel.bitmap.Destroy()
        self.windTunnelPanel.bitmap=wx.BitmapFromImage(theImage)
       
        self.windTunnelPanel.reInitBuffer=True
        self.Refresh()
            
    def OnClickPause(self, e):
        self.control.AppendText("Clicked on Pause button\n")
        self.controlPanel.SetBackgroundColour("GRAY")     
    
        
    def OnSlideVelocity(self, e):
        
        self.control.AppendText("Velocity Slider was moved %d\n" %e.GetInt())
        self.Velocity = e.GetInt()
    
        
    def OnSlideTemp(self, e):
        self.control.AppendText("Temperature Slider was moved %d\n" %e.GetInt())
        self.Temperature = e.GetInt()
    ###

   
app = wx.App(False)
frame = WindTunnelFrame(None, "Wind Tunnel Frame")
app.MainLoop()