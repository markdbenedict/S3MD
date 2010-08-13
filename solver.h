/*
 *  solver.h
 *  
 *
 *  Created by Mark D. Benedict on 8/9/10.
 *  Copyright 2010 University of Cambridge. All rights reserved.
 *
 */

void dens_step( int N,float * x, float * x0, float * u, float * v,float* b, float diff, float dt ,float vel);

void vel_step( int N,float * u, float * v, float * u0, float * v0,float* b, float visc, float dt ,float vel);

void runstep( int N,float * u, float * v, float * u0, float * v0, float* dens,float* dens_prev,float* b, float visc, float dt ,float diff,float vel);

int testOpenCL(int probSize);
