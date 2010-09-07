#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>

typedef float real;

#define CHAR_MINUS  '-'
#define CHAR_ZERO   '0'

typedef struct {
  real u[9];
} RMat;

#define MAT(a, n, i, j)  (a)[(i) + n * (j)]

#define AllocMem(a, n, t)  a = (t *) malloc ((n) * sizeof (t))

#define AllocMem2(a, n1, n2, t)                             \
   AllocMem (a, n1, t *);                                   \
   AllocMem (a[0], (n1) * (n2), t);                         \
   for (k = 1; k < n1; k ++) a[k] = a[k - 1] + n2;

#define MAX_MPEX_ORD  2
#define I(i, j)  ((i) * ((i) + 1) / 2 + (j))
#define c(i, j)  c[I(i, j)]
#define s(i, j)  s[I(i, j)]

typedef struct {
  real c[I(MAX_MPEX_ORD, MAX_MPEX_ORD) + 1], s[I(MAX_MPEX_ORD, MAX_MPEX_ORD) + 1];
} MpTerms;
typedef struct {
  MpTerms le, me;
  int occ;
} MpCell;

#include "in_vdefs.h"
#include "in_namelist.h"
#include "in_proto.h"

#define DO_MOL  for (n = 0; n < nMol; n ++)
#define DO_CELL(j, m)  for (j = cellList[m]; j >= 0; j = cellList[j])

#define VWrap(v, t)                                         \
   if (v.t >= 0.5 * (*inRegion).t)      v.t -= (*inRegion).t;         \
   else if (v.t < -0.5 * (*inRegion).t) v.t += (*inRegion).t

#define VShift(v, t)                                        \
   if (v.t >= 0.5 * (*inRegion).t)      shift.t -= (*inRegion).t;     \
   else if (v.t < -0.5 * (*inRegion).t) shift.t += (*inRegion).t

#define VShiftWrap(v, t)                                    \
   if (v.t >= 0.5 * (*inRegion).t) {                             \
     shift.t -= (*inRegion).t;                                   \
     v.t -= inRegion.t;                                       \
   } else if (v.t < -0.5 * (*inRegion.t)) {                      \
     shift.t += (*inRegion.t);                                   \
     v.t += (*inRegion).t;                                       \
   }

#define VCellWrap(t)                                        \
   if (m2v.t >= cells.t) {                                  \
     m2v.t = 0;                                             \
     shift.t = inRegion.t;                                    \
   } else if (m2v.t < 0) {                                  \
     m2v.t = cells.t - 1;                                   \
     shift.t = - inRegion.t;                                  \
   }

#define VWrapAll(v)                                         \
   {VWrap (v, x);                                           \
   VWrap (v, y);                                            \
   VWrap (v, z);}
#define VShiftAll(v)                                        \
   {VShift (v, x);                                          \
   VShift (v, y);                                           \
   VShift (v, z);}
#define VCellWrapAll()                                      \
   {VCellWrap (x);                                          \
   VCellWrap (y);                                           \
   VCellWrap (z);}

#define OFFSET_VALS                                           \
   { {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, {-1,1,0},            \
     {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}, {-1,1,1}, {-1,0,1},  \
     {-1,-1,1}, {0,-1,1}, {1,-1,1}                            \
   }

 
// Includes
#include <stdio.h>
#include <cutil_inline.h>

// Variables
typedef struct {
  VecR r, rv, ra, ra1, ra2, ro, rvo;
} Mol;



Mol     * d_mol;
VecR    *d_region;
int     *d_nebrTabPtr, *d_nebrTab;
real    *d_atomPotential,*d_g;
int     *d_indexSum;
real    *d_accum;
real    *h_atomPotential;
int     h_nMol;
VecI    h_UCell;
bool noprompt = false;

// Functions
void Cleanup(void);

void AllocGPUMemory(int nebrTabMax, int nebrTabLen,int nMol,VecR region,VecI inUnitCell)
{
    h_nMol = nMol;
    h_UCell=inUnitCell;
    cudaDeviceProp prop;
    int Dev;
    cudaGetDevice(&Dev);
    cudaGetDeviceProperties(&prop,Dev);
    
    printf("ID of current CUDA Device = %d\n",Dev);
    printf("the name of device is %s\n",prop.name);
    printf("compute capability %d.%d\n",prop.major,prop.minor);
    printf("inside .cu nMol=%d\n\n",nMol);
    printf("h_nMol=%d\n\n",h_nMol);
    
    h_atomPotential=(real*)malloc(sizeof(real)*h_nMol);
    cutilSafeCall(cudaMalloc((void**)&d_nebrTabPtr, (h_nMol+1)*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&d_nebrTab, nebrTabMax*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&d_mol, h_nMol*sizeof(Mol)));
    cutilSafeCall(cudaMalloc((void**)&d_g, 15*h_nMol*sizeof(real)));
    cutilSafeCall(cudaMalloc((void**)&d_atomPotential, h_nMol*sizeof(real)));
    cutilSafeCall(cudaMalloc((void**)&d_indexSum,sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&d_region,sizeof(VecR)));
    cutilSafeCall(cudaMemcpy(d_region, &region, sizeof(VecR), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMalloc((void**)&d_accum, h_nMol*sizeof(real)));    
}

void UpdateGPUNeighbors(int* nebrTabPtr,int* nebrTab,int nebrTabMax)
{
    cudaMemcpy(d_nebrTabPtr, nebrTabPtr, (h_nMol+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nebrTab, nebrTab, nebrTabMax*sizeof(int), cudaMemcpyHostToDevice);
}

__global__ void ZeroAccumulators(Mol* inMol,real* inAtomPotential,int inNumMol,real* accum,int* inCount)
{
  int n=blockIdx.x;//8*gridDim.x *gridDim.x * blockIdx.x + 8*gridDim.y*blockIdx.y + threadIdx.x;
  if(n < inNumMol)
  {
    atomicAdd(inCount,n);
    inMol[n].ra.x=0.;
    inMol[n].ra.y=0.;
    inMol[n].ra.z=0.;
    inAtomPotential[n]=0.;
    accum[n]=0.;
    
  }
}

__global__ void ComputeForcesGPU(Mol *inMol,real* inAtomPotential,int* inNebrTabPtr,int*inNebrTab,VecR* inRegion,real inRCut,int inNMol,int* inCounter,real* accum)
{
    VecR dr, dr12, dr13, w2, w3;
    real aCon = 7.0496, bCon = 0.60222, cr, er, fcVal,
       gCon = 1.2, lCon = 21., p12, p13, ri, ri3, rm,
       rm12, rm13, rm23,rr, rr12, rr13, rrCut;
    int j2, j3, m2, m3;
    int CURR=blockIdx.x;
    real fc=0,theta=0;
    real eta[5] = {0.01,0.1,0.5,1.0,10.0};
    real Rs[6] = {1.0,2.0,3.0,4.0,5.0,6.0};
    real g[15] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int zeta[3] = {1,2,3};
    int lambda[2] = {-1,1};
    if(CURR<inNMol)
    {
        atomicAdd(inCounter,CURR);
        accum[CURR]=inRCut;
        rrCut = Sqr (inRCut) - 0.001;
        for (m2 = inNebrTabPtr[CURR]; m2 < inNebrTabPtr[CURR + 1]; m2 ++)
        {
            j2 = inNebrTab[m2];
            if (CURR<j2)
            {
                //atomicMax(inCounter,j2);   
                VSub (dr, inMol[CURR].r, inMol[j2].r);
                VWrapAll (dr);
                rr = VLenSq (dr);
                if (rr < rrCut) {
                     //calculate if i,j conribution to G1(i)and add it to sum for i
                     rm = sqrt (rr);
                     er = exp (1. / (rm - inRCut));
                     ri = 1. / rm;
                     ri3 = Cube (ri);
                     fcVal = aCon * (4. * bCon * Sqr (ri3) +
                                         (bCon * ri3 * ri - 1.) * ri / Sqr (rm - inRCut)) * er;
                     VVSAdd (inMol[CURR].ra, fcVal, dr);
                     VVSAdd (inMol[j2].ra, - fcVal, dr);
                     //uSum += aCon * (bCon * ri3 * ri - 1.) * er;
                     inAtomPotential[CURR]+=aCon * (bCon * ri3 * ri - 1.) * er;
                     fc=1.0+0.5*cos(3.1419*rm/inRCut);
                     g[0]+=exp(-eta[0]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                     g[1]+=exp(-eta[0]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                     
                     g[2]+=exp(-eta[1]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                     g[3]+=exp(-eta[4]*(rm-Rs[1])*(rm-Rs[1]))*fc;
                     g[4]+=exp(-eta[1]*(rm-Rs[2])*(rm-Rs[2]))*fc;
                     g[5]+=exp(-eta[1]*(rm-Rs[4])*(rm-Rs[4]))*fc;
                     g[6]+=exp(-eta[1]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                     
                     g[7]+=exp(-eta[2]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                     g[8]+=exp(-eta[2]*(rm-Rs[3])*(rm-Rs[3]))*fc;
                     
                }
            }
        }
	
        //3 body terms
        /*
        for (m2 = inNebrTabPtr[CURR]; m2 < inNebrTabPtr[CURR + 1] - 1; m2 ++)
        {
           j2 = inNebrTab[m2];
           VSub (dr12, inMol[CURR].r, inMol[j2].r);
           VWrapAll (dr12);
           rr12 = VLenSq (dr12);
           if (rr12 < rrCut)
           {
                rm12 = sqrt (rr12);
                VScale (dr12, 1. / rm12);
                for (m3 = m2 + 1; m3 < inNebrTabPtr[CURR + 1]; m3 ++)
                {
                     j3 = inNebrTab[m3];
                     VSub (dr13, inMol[CURR].r, inMol[j3].r);
                     VWrapAll (dr13);
                     rr13 = VLenSq (dr13);
                     if (rr13 < rrCut)
                     {
                          rm13 = sqrt (rr13);
                          VScale (dr13, 1. / rm13);
                          cr = VDot (dr12, dr13);
                          er = lCon * (cr + 1./3.) * exp (gCon / (rm12 - inRCut) + gCon /
                                                                  (rm13 - inRCut));
                          p12 = gCon * (cr + 1./3.) / Sqr (rm12 - inRCut);
                          p13 = gCon * (cr + 1./3.) / Sqr (rm13 - inRCut);
                          VSSAdd (w2, p12 + 2. * cr / rm12, dr12, - 2. / rm12, dr13);
                          VSSAdd (w3, p13 + 2. * cr / rm13, dr13, - 2. / rm13, dr12);
                          VScale (w2, - er);
                          VScale (w3, - er);
                          VVSub (inMol[CURR].ra, w2);
                          VVSub (inMol[CURR].ra, w3);
                          VVAdd (inMol[j2].ra, w2);
                          VVAdd (inMol[j3].ra, w3);
                          //uSum += (cr + 1./3.) * er;
                          inAtomPotential[CURR]+=(cr + 1./3.) * er;
                          theta=acos(cr/rm13/rm12);
                          VecR dr23;
                          VSub (dr23, inMol[j2].r, inMol[j3].r); 
                          rm23= sqrt(VLenSq (dr23));
                          fc=(1.0+0.5*cos(3.1419*rm12/inRCut))*(1.0+0.5*cos(3.1419*rm13/inRCut))*(1.0+0.5*cos(3.1419*rm23/inRCut));
                          g[9]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[0]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          g[10]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[1]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          g[11]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          g[12]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[3]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          g[13]+=pow((real)2,1-zeta[1])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[1]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          g[14]+=pow((real)2,1-zeta[2])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[2]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          
                     }
                }
           }
        }*/
        
    }
    
}
 
/*
__device__ void zeroArray(double* theArray)
{
  

}*/




// Host code
void doForceIterartion(double inRCut,Mol* inMol,double &outUSum)
{
    //real g1Sum=0;
    //int n;
    int counter=0;
    real localUSum=outUSum;
    dim3 theSize(8000,1,1);
    //printf("h_nMol=%d\n",h_nMol);
    /*g1Sum=0;
    for(n=0;n<h_nMol;n++)
    {
        g1Sum+=mol[n].ra.x+mol[n].ra.y+mol[n].ra.z;
    }*/
    //printf("mol[n].ra sum just before zero=%6.4f\n",g1Sum);
    
    //clear out accumulators
    real sum=0;
    int n;
    real* theAccum=(real*)malloc(h_nMol*sizeof(real));
    cutilSafeCall(cudaMemcpy(d_indexSum,&counter, sizeof(int), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_mol,inMol,h_nMol*sizeof(Mol), cudaMemcpyHostToDevice));
    int blocks=8000;
    int threads=1;
    ZeroAccumulators<<<blocks,threads>>>(d_mol,d_atomPotential,8000,d_accum,d_indexSum);
    //cutilSafeCall(cudaMemcpy(theAccum, d_accum, h_nMol*sizeof(real), cudaMemcpyDeviceToHost));
    //cutilSafeCall(cudaMemcpy(h_atomPotential, d_atomPotential, h_nMol*sizeof(real), cudaMemcpyDeviceToHost));
    //cudaMemcpy(&counter, d_indexSum, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("sum of iterators=%d\n",counter);
    //for(n=0;n<h_nMol;n++) sum+=theAccum[n];
    //printf("Accum after zero=%f with h_nMol=%d\n",sum/h_nMol,h_nMol);
    //localUSum=0;
    //for(n=0;n<h_nMol;n++) localUSum+=h_atomPotential[n];
    //printf("localUSum after zero=%f\n",localUSum/h_nMol);
    //outUSum=localUSum/h_nMol;
    //printf("uSum after zero=%f\n",outUSum);
    //cudaThreadSynchronize();
    cutilSafeCall(cudaMemcpy(d_indexSum,&counter, sizeof(int), cudaMemcpyHostToDevice));
    ComputeForcesGPU<<<blocks,threads>>>(d_mol,d_atomPotential, d_nebrTabPtr,d_nebrTab,d_region,inRCut,h_nMol,d_indexSum,d_accum);
    cutilSafeCall(cudaMemcpy(inMol, d_mol, h_nMol*sizeof(Mol), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(h_atomPotential, d_atomPotential, h_nMol*sizeof(real), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(theAccum, d_accum, h_nMol*sizeof(real), cudaMemcpyDeviceToHost));
    cudaMemcpy(&counter, d_indexSum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("sum of iterations=%d\n",counter);
    for(n=0;n<h_nMol;n++) sum+=theAccum[n];
    printf("Accum after zero=%f with h_nMol=%d\n",sum/h_nMol,h_nMol);
    localUSum=0;
    for(n=0;n<h_nMol;n++) localUSum+=h_atomPotential[n];
    printf("localUSum after zero=%f\n",localUSum/h_nMol);
    outUSum=localUSum/h_nMol;
    printf("uSum after zero=%f\n",outUSum);
  
    free(theAccum);

}

void Cleanup(void)
{
    // Free device memory
    //if (d_A)
    //    cudaFree(d_A);
        
    //cudaThreadExit();
    /*
    if (!noprompt) {
        printf("\nPress ENTER to exit...\n");
        fflush( stdout);
        fflush( stderr);
        getchar();
    }*/

    //exit(0);
}



