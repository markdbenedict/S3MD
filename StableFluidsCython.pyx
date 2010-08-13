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
    int testOpenCL(int probSize)

DTYPE = np.float32
DTYPE2 = np.int

ctypedef np.float32_t DTYPE_t
ctypedef np.int_t DTYPE2_t

def testGPU(N):
    testOpenCL(N)

#@cython.boundscheck(False)
def mainloop(wingImage,velocity,temperature):
    
    cdef int N=wingImage.shape[0]
    print N
    cdef theSize=(N+2)*(N+2)
    print theSize
    theShape = wingImage.shape
    theShape = (theShape[0]+2,theShape[1]+2)
    
    wing=np.zeros(theShape,dtype=DTYPE)
    wing[1:-1,1:-1]=wingImage[:]
    wing.shape=theSize
    print 'starting calc'
    densInit=np.zeros(theShape,dtype=DTYPE)
    densInit[:,5:50]=0.1
    
    cdef np.ndarray[DTYPE_t,ndim=1] dens_prev=np.zeros(theSize,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] dens=np.zeros(theSize,dtype=DTYPE)
 
    densInit.shape=theSize
    dens=densInit[:]

    uInit=np.zeros(theShape,dtype=DTYPE)
    uInit[:,5:10]=0.1
    uInit.shape=theSize
    
    #np.asarray(0.03*(np.random.random(theSize)-0.5)+0.02,dtype=np.float32)
    cdef np.ndarray[DTYPE_t,ndim=1] u_prev=np.zeros(theSize,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] v_prev=np.zeros(theSize,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] u=np.zeros(theSize,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] v=np.zeros(theSize,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] b=np.ones(theSize,dtype=DTYPE)
    
    b=wing.copy()
    u=uInit[:]
    distance = 7.0 #this is an estimate.
    cdef float dt= 0.05
    cdef float visc=0.000025
    cdef float diff=0.0001
    cdef int i=0
    velocity = velocity/200. / 4.0 
    visc = (25.-temperature)/50.*visc
    duration = distance/velocity
    dt = 0.25 * np.exp(-3*velocity)
    numIter = duration/dt
    
    print 'numIterTotal = ',numIter
    print 'time =', duration
    print 'dt=',dt
    print 'visc=',visc
    print 'velocity=',velocity
    tempFileLocation = os.getcwd()+'/temp/'
    while (i < numIter):
        i+=1
        runstep(N, <DTYPE_t*> u.data, <DTYPE_t*> v.data,<DTYPE_t*> u_prev.data,<DTYPE_t*> v_prev.data,
                <DTYPE_t*> dens.data,<DTYPE_t*> dens_prev.data,<DTYPE_t*> b.data, visc, dt,diff,velocity)
        if i%3==0:
            #draw output
            print 'i=',i
            tempDens=dens.copy()
            tempDens.shape=theShape
            matplotlib.pyplot.imsave(tempFileLocation+'DensityImage'+str(i)+'.png',tempDens)
            tempDens=np.sqrt(u*u+v*v)
            tempDens.shape=theShape
            matplotlib.pyplot.imsave(tempFileLocation+'VelocityImage'+str(i)+'.png',tempDens)
    
    tempDens=dens.copy()
    tempDens.shape=theShape
    matplotlib.pyplot.imsave(tempFileLocation+'FinalImage'+'.png',tempDens)
    return tempFileLocation+'FinalImage'+'.png'