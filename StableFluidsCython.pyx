#!/usr/bin/env python
import scipy.misc.pilutil as pilutil
import os
import matplotlib.pyplot
#from Cython import *
import numpy as np
cimport numpy as np
cimport cython



cdef extern from "solver.h":
    void dens_step ( int N,float * x, float * x0, float * u, float * v, float* b,float diff, float dt ,float vel)
    void vel_step ( int N,float * u, float * v, float * u0, float * v0, float* b,float visc, float dt ,float vel)
    void runstep( int N,float * u, float * v, float * u0, float * v0, float* diff,float* diff_prev,float* b, float visc, float dt ,float diff,float vel)
    void testCUDA()
    #int testOpenCL(int probSize)

DTYPE = np.float32
DTYPE2 = np.int

ctypedef np.float32_t DTYPE_t
ctypedef np.int_t DTYPE2_t
class Engine():
    def __init__(self):
        
        #simulation environment parameters
        self.tempFileLocation = os.getcwd()+'/temp/'
        self.gCurrIter=0
        self.TotalIterNeeded=0
        #control variables sent to C code
        self.vel=0.0
        self.visc =0.0
        self.diff=0.0
        self.dt=0.0
    
    def testGPU(self,N):
        testCUDA()
        return
        
    def initSim(self, wingImage,velocity,temperature):
        print 'setting up calc'
        self.N=wingImage.shape[0]
        self.NShape = wingImage.shape
        self.NShape = (self.NShape[0]+2,self.NShape[1]+2)
        self.TotalSize=(self.N+2)*(self.N+2)
        
        print 'TotalSize=',self.TotalSize
        #allocate global storage
        self.dens_prev=np.zeros(self.TotalSize,dtype=DTYPE)
        self.dens=np.zeros(self.TotalSize,dtype=DTYPE)
        self.u_prev=np.zeros(self.TotalSize,dtype=DTYPE)
        self.v_prev=np.zeros(self.TotalSize,dtype=DTYPE)
        self.u=np.zeros(self.TotalSize,dtype=DTYPE)
        self.v=np.zeros(self.TotalSize,dtype=DTYPE)
        self.b=np.ones(self.TotalSize,dtype=DTYPE)
        self.wing=np.zeros(self.NShape,dtype=DTYPE)
        self.wing[1:-1,1:-1]=wingImage[:]
        self.wing.shape=self.TotalSize
        self.b=self.wing.copy()
                
        densInit=np.zeros(self.NShape,dtype=DTYPE)
        densInit[:,5:50]=0.1
     
        densInit.shape=self.TotalSize
        self.dens=densInit[:]
        
        self.distance = 7.0 #this is an estimate.
        self.dt= 0.05
        self.visc=0.000025
        self.diff=0.0001
        self.gCurrIter=0
        self.vel=0.0
        self.vel = velocity/200. / 4.0 
        self.visc = (25.-temperature)/50.*self.visc
        self.duration = self.distance/self.vel
        self.dt = 0.25 * np.exp(-3*self.vel)
        self.TotalIterNeeded = int(self.duration/self.dt)
        
        return self.TotalIterNeeded
        
    
    #@cython.boundscheck(False)
    def doIterations(self,numIter=1,outputFreq=5):
        
        #setup local buffers
        cdef np.ndarray[DTYPE_t,ndim=1] _u_prev = self.u_prev
        cdef np.ndarray[DTYPE_t,ndim=1] _v_prev= self.v_prev
        cdef np.ndarray[DTYPE_t,ndim=1] _u=self.u 
        cdef np.ndarray[DTYPE_t,ndim=1] _v=self.v 
        cdef np.ndarray[DTYPE_t,ndim=1] _b=self.b
        cdef np.ndarray[DTYPE_t,ndim=1] _dens_prev=self.dens_prev
        cdef np.ndarray[DTYPE_t,ndim=1] _dens=self.dens
        
        cdef int currIter = int(self.gCurrIter)
        
        i=0
        while (i <= numIter):
            self.testGPU(self.N)
            i=i+1
            print 'current iteration=',currIter
            runstep(self.N, <DTYPE_t*> _u.data, <DTYPE_t*> _v.data,<DTYPE_t*> _u_prev.data,<DTYPE_t*> _v_prev.data,
                    <DTYPE_t*> _dens.data,<DTYPE_t*> _dens_prev.data,<DTYPE_t*> _b.data, self.visc, self.dt,self.diff,self.vel)
            currIter=currIter+1
            if i%outputFreq==0:
                print 'current iteration=',currIter
                #draw output
                tempDens=self.dens.copy()
                tempDens.shape=self.NShape
                matplotlib.pyplot.imsave(self.tempFileLocation+'DensityImage'+str(currIter)+'.png',tempDens)
                
                tempDens=np.sqrt(self.u*self.u+self.v*self.v)
                tempDens.shape=self.NShape
                matplotlib.pyplot.imsave(self.tempFileLocation+'VelocityImage'+str(currIter)+'.png',tempDens)
        
        tempDens=self.dens.copy()
        tempDens.shape=self.NShape
        matplotlib.pyplot.imsave(self.tempFileLocation+'FinalImage'+'.png',tempDens)
        self.gCurrIter=currIter
        return self.tempFileLocation+'FinalImage'+'.png'