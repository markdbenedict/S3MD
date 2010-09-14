
/* [[pr_12_1 - silicon, RDF]] */

/*********************************************************************

  (C) 2004  D. C. Rapaport

  This software is copyright material accompanying the book
  "The Art of Molecular Dynamics Simulation", 2nd edition,
  by D. C. Rapaport, published by Cambridge University Press (2004).

**********************************************************************/


#include "in_mddefs.h"

typedef struct {
  VecR r, rv, ra, ra1, ra2, ro, rvo;
} Mol;

Mol *mol;
VecR region, vSum;
VecI initUcell;
real deltaT, density, rCut, temperature, timeNow, uSum, velMag, vvSum;
Prop kinEnergy, totEnergy;
int moreCycles, nMol, stepAvg, stepCount, stepEquil, stepLimit;
VecI cells;
int *cellList;
real dispHi, rNebrShell;
int *nebrTab, nebrNow, nebrTabFac, nebrTabLen, nebrTabMax;
int *nebrTabPtr;
int stepAdjustTemp;
real *histRdf, rangeRdf,*atomPotential,*g1,*g2,*g3,*g4,*g5,*g6;
real *g7,*g8,*g9,*g10;
real *g25,*g26,*g27,*g28,*g29,*g30;
real *rCutTemp;
int countRdf, limitRdf, sizeHistRdf, stepRdf;


FILE *rdfFile,*trainingData;

NameList nameList[] = {
  NameR (deltaT),
  NameR (density),
  NameI (initUcell),
  NameI (limitRdf),
  NameI (nebrTabFac),
  NameR (rangeRdf),
  NameR (rCut),
  NameR (rNebrShell),
  NameI (sizeHistRdf),
  NameI (stepAdjustTemp),
  NameI (stepAvg),
  NameI (stepEquil),
  NameI (stepLimit),
  NameI (stepRdf),
  NameR (temperature),
};

void SetupTraining();
void ComputeTraining();
void WriteOutTrainingData();
void doForceIterartion(double inRCut,Mol* inMol,double& outUSum);
void AllocGPUMemory(int nebrTabMax, int nebrTabLen,int inNMol,double rx,double ry, double rz, int ua, int ub, int uc);
void UpdateGPUNeighbors(int* nebrTabPtr,int* nebrTab,int nebrTabMax);

int main (int argc, char **argv)
{
  
  GetNameList (argc, argv);
  char fileStr[80];
  strcat(fileStr,argv[0]);
  strcat(fileStr,".rdf");
  
  //PrintNameList (stdout);
  rdfFile=fopen(fileStr,"w");
  
  SetParams ();
  SetupJob ();
  SetupTraining();
  //printf("region just before Alloc is %f,%f,%f\n",region.x,region.y,region.z);
  AllocGPUMemory(nebrTabMax, nebrTabLen, nMol,region.x,region.y,region.z,initUcell.x,initUcell.x,initUcell.x);  
  
  //printf("after init x=%f,y=%f,z=%f\n",mol[100].r.x,mol[100].r.y,mol[100].r.z);
  moreCycles = 1;
  while (moreCycles) {
    SingleStep ();
    if (stepCount >= stepLimit) moreCycles = 0;
  }
}


void SingleStep ()
{
  //printf("region in .C is %f,%f,%f\n",region.x,region.y,region.z);
  ++ stepCount;
  timeNow = stepCount * deltaT;
  if (nebrNow) {
    nebrNow = 0;
    dispHi = 0.;
    BuildNebrList ();
  }
  PredictorStep ();
  //ComputeTraining();
  doForceIterartion(rCut,mol,uSum);
  //ComputeForces();
  //printf("atomPotential in .C after zero=%f\n",uSum);
  
  ApplyThermostat ();
  CorrectorStep ();
  ApplyBoundaryCond ();
  //printf("step %d uSum in .C=%f\n\n",stepCount,uSum);
  EvalProps ();
  if (stepCount % stepAdjustTemp == 0) AdjustTemp ();
  AccumProps (1);
  if (stepCount % stepAvg == 0) {
    AccumProps (2);
    PrintSummary (stdout);
    AccumProps (0);
  }
  //if (stepCount >= stepEquil && (stepCount - stepEquil) %
   //  stepRdf == 0) WriteOutTrainingData();//ComputeTraining();//EvalRdf ();
}

void SetupJob ()
{
  AllocArrays ();
  stepCount = 0;
  InitCoords ();
  InitVels ();
  InitAccels ();
  AccumProps (0);
  nebrNow = 1;
  countRdf = 0;
}

void SetParams ()
{
  VSCopy (region, 1. / pow (density / 8., 1./3.), initUcell);
     //printf("region initialized to %f,%f,%f\n",region.x,region.y,region.z);
      //printf("initUcell is %d,%d,%d\n",initUcell.x,initUcell.y,initUcell.z);
  nMol = 8 * VProd (initUcell);
  velMag = sqrt (NDIM * (1. - 1. / nMol) * temperature);
  VSCopy (cells, 1. / (rCut + rNebrShell), region);
  nebrTabMax = nebrTabFac * nMol;
}

void SetupTraining()
{
	//trainingData=fopen("/Users/MarkB/Desktop/trainingData.txt","w");
    //printf("nMol=%d  size of arrays=%d",nMol,sizeof(double)*nMol);
	atomPotential =(real*)malloc(sizeof(real)*nMol);
	AllocMem (g1, nMol, real);
	AllocMem (g2, nMol, real);
	AllocMem (g3, nMol, real);
	AllocMem (g4, nMol, real);
	AllocMem (g5, nMol, real);
	AllocMem (g6, nMol, real);
	AllocMem (g7, nMol, real);
	AllocMem (g8, nMol, real);
	AllocMem (g9, nMol, real);
	AllocMem (g10, nMol, real);
	AllocMem (g25, nMol, real);
	AllocMem (g26, nMol, real);
	AllocMem (g27, nMol, real);
	AllocMem (g28, nMol, real);
	AllocMem (g29, nMol, real);
	AllocMem (g30, nMol, real);
	
}
void AllocArrays ()
{
  AllocMem (mol, nMol, Mol);
  AllocMem (cellList, VProd (cells) + nMol, int);
  AllocMem (nebrTab, nebrTabMax, int);
  AllocMem (nebrTabPtr, nMol + 1, int);
  AllocMem (histRdf, sizeHistRdf, real);
}



#undef OFFSET_VALS

#define OFFSET_VALS                                       \
   { {-1,-1,-1}, {0,-1,-1}, {1,-1,-1},                    \
     {-1,0,-1}, {0,0,-1}, {1,0,-1}, {-1,1,-1}, {0,1,-1},  \
     {1,1,-1}, {-1,-1,0}, {0,-1,0}, {1,-1,0}, {-1,0,0},   \
     {0,0,0}, {1,0,0}, {-1,1,0}, {0,1,0}, {1,1,0},        \
     {-1,-1,1}, {0,-1,1}, {1,-1,1}, {-1,0,1},  {0,0,1},   \
     {1,0,1}, {-1,1,1}, {0,1,1}, {1,1,1}                  \
   }

void BuildNebrList ()
{
  VecR dr, invWid, rs, shift;
  VecI cc, m1v, m2v, vOff[] = OFFSET_VALS;
  real rrNebr;
  int c, j1, j2, m2, n, offset;

  rrNebr = Sqr (rCut + rNebrShell);
  VDiv (invWid, cells, region);
  for (n = nMol; n < nMol + VProd (cells); n ++) //set default of -1 for all cells
     cellList[n] = -1;
  DO_MOL {
    VSAdd (rs, mol[n].r, 0.5, region); // add 1/2 the region value to each atom position 
    VMul (cc, rs, invWid); //multiply the position by the inverse width
    c = VLinear (cc, cells) + nMol; //c is which cell its in?
    cellList[n] = cellList[c];
    cellList[c] = n;
  }
  nebrTabLen = 0;
  for (j1 = 0; j1 < nMol; j1 ++) {
    VSAdd (rs, mol[j1].r, 0.5, region);
    VMul (m1v, rs, invWid);
    nebrTabPtr[j1] = nebrTabLen;
    for (offset = 0; offset < 27; offset ++) {
      VAdd (m2v, m1v, vOff[offset]);
      VZero (shift);
      VCellWrapAll ();
      m2 = VLinear (m2v, cells) + nMol;
      DO_CELL (j2, m2) {
        if (j2 != j1) {
          VSub (dr, mol[j1].r, mol[j2].r);
          VVSub (dr, shift);
          if (VLenSq (dr) < rrNebr) {
            if (nebrTabLen >= nebrTabMax) ErrExit (ERR_TOO_MANY_NEBRS);
            nebrTab[nebrTabLen] = j2;
            ++ nebrTabLen;
          }
        }
      }
    }
  }
  nebrTabPtr[nMol] = nebrTabLen;
  UpdateGPUNeighbors(nebrTabPtr,nebrTab, nebrTabMax);
}

void ComputeForces ()
{
  VecR dr, dr12, dr13, w2, w3;
  real aCon = 7.0496, bCon = 0.60222, cr, er, fcVal,
     gCon = 1.2, lCon = 21., p12, p13, ri, ri3, rm,
     rm12, rm13, rr, rr12, rr13, rrCut;
  int j1, j2, j3, m2, m3, n;
  rrCut = Sqr (rCut) - 0.001;
  DO_MOL VZero (mol[n].ra);
  uSum = 0.;
  for (j1 = 0; j1 < nMol; j1 ++) { //2body terms
    for (m2 = nebrTabPtr[j1]; m2 < nebrTabPtr[j1 + 1]; m2 ++) {
      j2 = nebrTab[m2];
      if (j1 < j2) {  //will have to take this out with ANN, Newtons 3rd law wont help, need to consider all atoms pairs for each atom to get correct G1(i) and G2(i)
        VSub (dr, mol[j1].r, mol[j2].r);
        VWrapAll (dr);
        rr = VLenSq (dr);
        if (rr < rrCut) {
		  //calculate if i,j conribution to G1(i)and add it to sum for i
          rm = sqrt (rr);
          er = exp (1. / (rm - rCut));
          ri = 1. / rm;
          ri3 = Cube (ri);
          fcVal = aCon * (4. * bCon * Sqr (ri3) +
             (bCon * ri3 * ri - 1.) * ri / Sqr (rm - rCut)) * er;
          VVSAdd (mol[j1].ra, fcVal, dr);
          VVSAdd (mol[j2].ra, - fcVal, dr);
          uSum += aCon * (bCon * ri3 * ri - 1.) * er;
        } 
      }
	}
  }
  for (j1 = 0; j1 < nMol; j1 ++) { //3 body terms
    for (m2 = nebrTabPtr[j1]; m2 < nebrTabPtr[j1 + 1] - 1; m2 ++) {
      j2 = nebrTab[m2];
      VSub (dr12, mol[j1].r, mol[j2].r);
      VWrapAll (dr12);
      rr12 = VLenSq (dr12);
      if (rr12 < rrCut) {
        rm12 = sqrt (rr12);
        VScale (dr12, 1. / rm12);
        for (m3 = m2 + 1; m3 < nebrTabPtr[j1 + 1]; m3 ++) {
          j3 = nebrTab[m3];
          VSub (dr13, mol[j1].r, mol[j3].r);
          VWrapAll (dr13);
          rr13 = VLenSq (dr13);
          if (rr13 < rrCut) {
            rm13 = sqrt (rr13);
            VScale (dr13, 1. / rm13);
            cr = VDot (dr12, dr13);
            er = lCon * (cr + 1./3.) * exp (gCon / (rm12 - rCut) + gCon /
               (rm13 - rCut));
            p12 = gCon * (cr + 1./3.) / Sqr (rm12 - rCut);
            p13 = gCon * (cr + 1./3.) / Sqr (rm13 - rCut);
            VSSAdd (w2, p12 + 2. * cr / rm12, dr12, - 2. / rm12, dr13);
            VSSAdd (w3, p13 + 2. * cr / rm13, dr13, - 2. / rm13, dr12);
            VScale (w2, - er);
            VScale (w3, - er);
            VVSub (mol[j1].ra, w2);
            VVSub (mol[j1].ra, w3);
            VVAdd (mol[j2].ra, w2);
            VVAdd (mol[j3].ra, w3);
            uSum += (cr + 1./3.) * er;
          }
        }
      }
    }
  }
}

void ComputeTraining()
{
	VecR dr, dr12, dr13, w2, w3;
	real fc=0;
	static real eta[5] = {0.01,0.1,0.5,1.0,10.0};
	static real Rs[6] = {1.0,2.0,3.0,4.0,5.0,6.0};
	static int zeta[3] = {1,2,3};
	static int lambda[2] = {-1,1};

	real aCon = 7.0496, bCon = 0.60222, cr, er, fcVal,theta;
	real gCon = 1.2, lCon = 21., p12, p13, ri, ri3, rm, rm12, rm13,rm23, rr, rr12, rr13, rrCut;
	int j1, j2, j3, m2, m3, n;

	rrCut = Sqr (rCut) - 0.001;
     
     //float accum=0,accum2=0;
     //DO_MOL
     //{
      //accum+=mol[n].ra.x+mol[n].ra.y+mol[n].ra.z;
      //accum2+=g1[n];//+g2[n]+g3[n]+g4[n];
    // }
     //printf("Inside Training\n");
     //printf("mol.ra accum=%f\n",accum);
     //printf("g1 accum=%f\n",accum2);
	DO_MOL VZero (mol[n].ra);
	DO_MOL
	{
		atomPotential[n]=0;
		g1[n]=0;
		g2[n]=0;
		g3[n]=0;
		g4[n]=0;
		g5[n]=0;
		g6[n]=0;
		
		g7[n]=0;
		g8[n]=0;
		g9[n]=0;
		g10[n]=0;

		g25[n]=0;
		g26[n]=0;
		g27[n]=0;
		g28[n]=0;
		g29[n]=0;
		g30[n]=0;
	}
	uSum = 0.;
	for (j1 = 0; j1 < nMol; j1 ++) { //2body terms
		for (m2 = nebrTabPtr[j1]; m2 < nebrTabPtr[j1 + 1]; m2 ++) {
			j2 = nebrTab[m2];
               if(j1<j2)
               {
                  VSub (dr, mol[j1].r, mol[j2].r);
                  VWrapAll (dr);
                  rr = VLenSq (dr);
                  if (rr < rrCut)
                  {
                       //calculate if i,j conribution to G1(i)and add it to sum for i
                       rm = sqrt (rr);
                       er = exp (1. / (rm - rCut));
                       ri = 1. / rm;
                       ri3 = Cube (ri);
                       fcVal = aCon * (4. * bCon * Sqr (ri3) +
                                           (bCon * ri3 * ri - 1.) * ri / Sqr (rm - rCut)) * er;
                       VVSAdd (mol[j1].ra, fcVal, dr);
                       VVSAdd (mol[j2].ra, - fcVal, dr);
                       uSum += aCon * (bCon * ri3 * ri - 1.) * er;
                       atomPotential[j1]+=aCon * (bCon * ri3 * ri - 1.) * er;
                       fc=1.0+0.5*cos(3.1419*rm/rCut);
                       g1[j1]+=exp(-eta[0]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g2[j1]+=exp(-eta[0]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                       
                       g3[j1]+=exp(-eta[1]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g4[j1]+=exp(-eta[4]*(rm-Rs[1])*(rm-Rs[1]))*fc;
                       g5[j1]+=exp(-eta[1]*(rm-Rs[2])*(rm-Rs[2]))*fc;
                       g6[j1]+=exp(-eta[1]*(rm-Rs[4])*(rm-Rs[4]))*fc;
                       g7[j1]+=exp(-eta[4]*(rm-Rs[3])*(rm-Rs[3]))*fc;
                       g8[j1]+=exp(-eta[1]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                       
                       g9[j1]+=exp(-eta[2]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g10[j1]+=exp(-eta[2]*(rm-Rs[3])*(rm-Rs[3]))*fc;
                       
                       g1[j1]+=exp(-eta[0]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g2[j1]+=exp(-eta[0]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                       
                       g3[j1]+=exp(-eta[1]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g4[j1]+=exp(-eta[4]*(rm-Rs[1])*(rm-Rs[1]))*fc;
                       g5[j1]+=exp(-eta[1]*(rm-Rs[2])*(rm-Rs[2]))*fc;
                       g6[j1]+=exp(-eta[1]*(rm-Rs[4])*(rm-Rs[4]))*fc;
                       g7[j1]+=exp(-eta[4]*(rm-Rs[3])*(rm-Rs[3]))*fc;
                       g8[j1]+=exp(-eta[1]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                       
                       g9[j1]+=exp(-eta[2]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g10[j1]+=exp(-eta[2]*(rm-Rs[3])*(rm-Rs[3]))*fc;
                       
                       g1[j1]+=exp(-eta[0]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g2[j1]+=exp(-eta[0]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                       
                       g3[j1]+=exp(-eta[1]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g4[j1]+=exp(-eta[4]*(rm-Rs[1])*(rm-Rs[1]))*fc;
                       g5[j1]+=exp(-eta[1]*(rm-Rs[2])*(rm-Rs[2]))*fc;
                       g6[j1]+=exp(-eta[1]*(rm-Rs[4])*(rm-Rs[4]))*fc;
                       g7[j1]+=exp(-eta[4]*(rm-Rs[3])*(rm-Rs[3]))*fc;
                       g8[j1]+=exp(-eta[1]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                       
                       g9[j1]+=exp(-eta[2]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                       g10[j1]+=exp(-eta[2]*(rm-Rs[3])*(rm-Rs[3]))*fc;
                       
                  }
                }
		} 
	}
	for (j1 = 0; j1 < nMol; j1 ++) { //3 body terms
		for (m2 = nebrTabPtr[j1]; m2 < nebrTabPtr[j1 + 1] - 1; m2 ++) {
			j2 = nebrTab[m2];
			VSub (dr12, mol[j1].r, mol[j2].r);
			VWrapAll (dr12);
			rr12 = VLenSq (dr12);
			if (rr12 < rrCut) {
				rm12 = sqrt (rr12);
				VScale (dr12, 1. / rm12);
				for (m3 = m2 + 1; m3 < nebrTabPtr[j1 + 1]; m3 ++) {
					j3 = nebrTab[m3];
					VSub (dr13, mol[j1].r, mol[j3].r);
					VWrapAll (dr13);
					rr13 = VLenSq (dr13);
					if (rr13 < rrCut) {
						rm13 = sqrt (rr13);
						VScale (dr13, 1. / rm13);
						cr = VDot (dr12, dr13);
						er = lCon * (cr + 1./3.) * exp (gCon / (rm12 - rCut) + gCon /
														(rm13 - rCut));
						p12 = gCon * (cr + 1./3.) / Sqr (rm12 - rCut);
						p13 = gCon * (cr + 1./3.) / Sqr (rm13 - rCut);
						VSSAdd (w2, p12 + 2. * cr / rm12, dr12, - 2. / rm12, dr13);
						VSSAdd (w3, p13 + 2. * cr / rm13, dr13, - 2. / rm13, dr12);
						VScale (w2, - er);
						VScale (w3, - er);
						VVSub (mol[j1].ra, w2);
						VVSub (mol[j1].ra, w3);
						VVAdd (mol[j2].ra, w2);
						VVAdd (mol[j3].ra, w3);
						uSum += (cr + 1./3.) * er;
						atomPotential[j1]+=(cr + 1./3.) * er;
						theta=acos(cr/rm13/rm12);
						VecR dr23;
						VSub (dr23, mol[j2].r, mol[j3].r); 
						rm23= sqrt(VLenSq (dr23));
					
                                                fc=(1.0+0.5*cos(3.1419*rm12/rCut))*(1.0+0.5*cos(3.1419*rm13/rCut))*(1.0+0.5*cos(3.1419*rm23/rCut));
						g25[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[0]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g26[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[1]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g27[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g28[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[3]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g29[j1]+=pow((real)2,1-zeta[1])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[1]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g30[j1]+=pow((real)2,1-zeta[2])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[2]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                                                
                                                g25[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[0]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g26[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[1]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g27[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g28[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[3]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g29[j1]+=pow((real)2,1-zeta[1])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[1]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g30[j1]+=pow((real)2,1-zeta[2])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[2]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                                                
                                                g25[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[0]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g26[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[1]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g27[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g28[j1]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[3]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g29[j1]+=pow((real)2,1-zeta[1])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[1]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						g30[j1]+=pow((real)2,1-zeta[2])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[2]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
						
					}
				}
			}
		}
	}
	//uSum /=2.0;
}

#define PCR4(r, ro, v, a, a1, a2, t)                        \
   r.t = ro.t + deltaT * v.t +                              \
   wr * (cr[0] * a.t + cr[1] * a1.t + cr[2] * a2.t)
#define PCV4(r, ro, v, a, a1, a2, t)                        \
   v.t = (r.t - ro.t) / deltaT +                            \
   wv * (cv[0] * a.t + cv[1] * a1.t + cv[2] * a2.t)

#define PR(t)                                               \
   PCR4 (mol[n].r, mol[n].r, mol[n].rv,                     \
   mol[n].ra, mol[n].ra1, mol[n].ra2, t)
#define PRV(t)                                              \
   PCV4 (mol[n].r, mol[n].ro, mol[n].rv,                    \
   mol[n].ra, mol[n].ra1, mol[n].ra2, t)
#define CR(t)                                               \
   PCR4 (mol[n].r, mol[n].ro, mol[n].rvo,                   \
   mol[n].ra, mol[n].ra1, mol[n].ra2, t)
#define CRV(t)                                              \
   PCV4 (mol[n].r, mol[n].ro, mol[n].rv,                    \
   mol[n].ra, mol[n].ra1, mol[n].ra2, t)

void PredictorStep ()
{

  real cr[] = {19.,-10.,3.}, cv[] = {27.,-22.,7.}, div = 24., wr, wv;
  int n;

  wr = Sqr (deltaT) / div;
  wv = deltaT / div;
  DO_MOL {
    mol[n].ro = mol[n].r;
    mol[n].rvo = mol[n].rv;
    PR (x);
    PRV (x);
    PR (y);
    PRV (y);
    PR (z);
    PRV (z);
    mol[n].ra2 = mol[n].ra1;
    mol[n].ra1 = mol[n].ra;
  }
}




void CorrectorStep ()
{
  real cr[] = {3.,10.,-1.}, cv[] = {7.,6.,-1.}, div = 24., wr, wv;
  int n;

  wr = Sqr (deltaT) / div;
  wv = deltaT / div;
  DO_MOL {
    CR (x);
    CRV (x);
    CR (y);
    CRV (y);
    CR (z);
    CRV (z);
  }
}


void ApplyBoundaryCond ()
{
  int n;

  DO_MOL VWrapAll (mol[n].r);
}


void AdjustTemp ()
{
  real vFac;
  int n;

  vvSum = 0.;
  DO_MOL vvSum += VLenSq (mol[n].rv);
  vFac = velMag / sqrt (vvSum / nMol);
  DO_MOL VScale (mol[n].rv, vFac);
}


void ApplyThermostat ()
{
  real s1, s2, vFac;
  int n;

  s1 = s2 = 0.;
  DO_MOL {
    s1 += VDot (mol[n].rv, mol[n].ra);
    s2 += VLenSq (mol[n].rv);
  }
  vFac = - s1 / s2;
  DO_MOL VVSAdd (mol[n].ra, vFac, mol[n].rv);
}


void EvalRdf ()
{
  VecR dr;
  real deltaR, normFac, rr;
  int j1, j2, n;

  if (countRdf == 0) {
    for (n = 0; n < sizeHistRdf; n ++) histRdf[n] = 0.;
  }
  deltaR = rangeRdf / sizeHistRdf;
  for (j1 = 0; j1 < nMol - 1; j1 ++) {
    for (j2 = j1 + 1; j2 < nMol; j2 ++) {
      VSub (dr, mol[j1].r, mol[j2].r);
      VWrapAll (dr);
      rr = VLenSq (dr);
      if (rr < Sqr (rangeRdf)) {
        n = sqrt (rr) / deltaR;
        ++ histRdf[n];
      }
    }
  }
  ++ countRdf;
  if (countRdf == limitRdf) {
    normFac = VProd (region) / (2. * M_PI * Cube (deltaR) *
       Sqr (nMol) * countRdf);
    for (n = 0; n < sizeHistRdf; n ++)
       histRdf[n] *= normFac / Sqr (n - 0.5);
    PrintRdf (rdfFile);
    countRdf = 0;
  }
}

void PrintRdf (FILE *fp)
{
  real rb;
  int n;

  fprintf (fp, "rdf\n");
  for (n = 0; n < sizeHistRdf; n ++) {
    rb = (n + 0.5) * rangeRdf / sizeHistRdf;
    fprintf (fp, "%8.4f %8.4f\n", rb, histRdf[n]);
  }
  fflush (fp);
}


void InitCoords ()
{
  VecR c, gap;
  real subShift;
  int j, m, n, nx, ny, nz;

  VDiv (gap, region, initUcell);
  n = 0;
  for (nz = 0; nz < initUcell.z; nz ++) {
    for (ny = 0; ny < initUcell.y; ny ++) {
      for (nx = 0; nx < initUcell.x; nx ++) {
        VSet (c, nx + 0.125, ny + 0.125, nz + 0.125);
        VMul (c, c, gap);
        VVSAdd (c, -0.5, region);
        for (m = 0; m < 2; m ++) {
          subShift = (m == 1) ? 0.25 : 0.;
          for (j = 0; j < 4; j ++) {
            VSAdd (mol[n].r, c, subShift, gap);
            if (j != 3) {
              if (j != 0) mol[n].r.x += 0.5 * gap.x;
              if (j != 1) mol[n].r.y += 0.5 * gap.y;
              if (j != 2) mol[n].r.z += 0.5 * gap.z;
            }
            ++ n;
          }
        }
      }
    }
  }
}


void InitVels ()
{
  int n;

  VZero (vSum);
  DO_MOL {
    VRand (&mol[n].rv);
    VScale (mol[n].rv, velMag);
    VVAdd (vSum, mol[n].rv);
  }
  DO_MOL VVSAdd (mol[n].rv, - 1. / nMol, vSum);
}


void InitAccels ()
{
  int n;

  DO_MOL {
    VZero (mol[n].ra);
    VZero (mol[n].ra1);
    VZero (mol[n].ra2);
  }
}


void EvalProps ()
{
  real vv, vvMax;
  int n;

  VZero (vSum);
  vvSum = 0.;
  vvMax = 0.;
  DO_MOL {
    VVAdd (vSum, mol[n].rv);
    vv = VLenSq (mol[n].rv);
    vvSum += vv;
    vvMax = Max (vvMax, vv);
  }
  dispHi += sqrt (vvMax) * deltaT;
  if (dispHi > 0.5 * rNebrShell) nebrNow = 1;
  kinEnergy.val = 0.5 * vvSum / nMol;
  totEnergy.val = kinEnergy.val + uSum / nMol;
}


void AccumProps (int icode)
{
	if (icode == 0) {
		PropZero (totEnergy);
		PropZero (kinEnergy);
	} else if (icode == 1) {
		PropAccum (totEnergy);
		PropAccum (kinEnergy);
	} else if (icode == 2) {
		PropAvg (totEnergy, stepAvg);
		PropAvg (kinEnergy, stepAvg);
	}
}


void PrintSummary (FILE *fp)
{
	fprintf (fp,
			 "%5d %8.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n",
			 stepCount, timeNow, VCSum (vSum) / nMol, PropEst (totEnergy),
			 PropEst (kinEnergy));
	fflush (fp);
}

void WriteOutTrainingData()
{
	int n;
	
    for (n = 0; n < nMol; n ++) {
		fprintf (trainingData, "%8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n", 
				 g1[n], g2[n],g3[n],g4[n],g5[n],g6[n], 
				 g7[n], g8[n],g9[n],g10[n],/*g11[n],g12[n], 
				 g13[n], g14[n],g15[n],g16[n],g17[n],g18[n], 
				 g19[n], g20[n],g21[n],g22[n],g23[n],g24[n], */
				 g25[n], g26[n],g27[n],g28[n],g29[n],g30[n],/* 
				 g31[n], g32[n],g33[n],g34[n],g35[n],g36[n],
				 g37[n], g38[n],g39[n],g40[n],g41[n],g42[n], 
				 g43[n], g44[n],g45[n],g46[n],g47[n],g48[n],*/
				 
				 atomPotential[n]);
	}
	fflush (trainingData);
}

#include "in_rand.c"
#include "in_errexit.c"
#include "in_namelist.c"



 
// Includes
#include <cutil_inline.h>

Mol     * d_mol;
VecR    *d_region;
int     *d_nebrTabPtr, *d_nebrTab;
real    *d_atomPotential,*d_g;
int     *d_indexSum;
real    *d_accum;
real    *h_atomPotential;
int     h_nMol;
VecI    h_UCell;
VecR    h_Region;
bool noprompt = false;

// Functions
void Cleanup(void);

void AllocGPUMemory(int nebrTabMax, int nebrTabLen,int inNMol,double rx,double ry, double rz, int ua, int ub, int uc)
{
    h_nMol = inNMol;
    h_UCell.x=ua;
    h_UCell.y=ub;
    h_UCell.z=uc;
    h_Region.x=rx;
    h_Region.y=rx;
    h_Region.z=rx;
    
    cudaDeviceProp prop;
    int Dev;
    cudaGetDevice(&Dev);
    cudaGetDeviceProperties(&prop,Dev);
    
    printf("ID of current CUDA Device = %d\n",Dev);
    printf("the name of device is %s\n",prop.name);
    printf("compute capability %d.%d\n",prop.major,prop.minor);
    printf("inside .cu nMol=%d\n\n",inNMol);
    printf("h_nMol=%d\n\n",h_nMol);
    
    printf("region is %f,%f,%f\n",rx,ry,rz);
    printf("h_Region is %f,%f,%f\n",h_Region.x,h_Region.y,h_Region.z);
    printf("unitCell is %d,%d,%d\n",h_UCell.x,h_UCell.y,h_UCell.z);
    
    h_atomPotential=(real*)malloc(sizeof(real)*h_nMol);
    cudaMalloc((void**)&d_nebrTabPtr, (h_nMol+1)*sizeof(int));
    cudaMalloc((void**)&d_nebrTab, nebrTabMax*sizeof(int));
    cudaMalloc((void**)&d_mol, h_nMol*sizeof(Mol));
    cudaMalloc((void**)&d_g, 15*h_nMol*sizeof(real));
    cudaMalloc((void**)&d_atomPotential, h_nMol*sizeof(real));
    cudaMalloc((void**)&d_indexSum,sizeof(int));
    cudaMalloc((void**)&d_region,sizeof(VecR));
    cudaMemcpy(d_region, &h_Region, sizeof(VecR), cudaMemcpyHostToDevice);
    VecR testRegion;
    cudaMemcpy(&testRegion, d_region, sizeof(VecR), cudaMemcpyDeviceToHost);
    //printf("testRegion is %f,%f,%f\n",testRegion.x,testRegion.y,testRegion.z);
    cudaMalloc((void**)&d_accum, h_nMol*sizeof(real));    
}

void UpdateGPUNeighbors(int* nebrTabPtr,int* nebrTab,int nebrTabMax)
{
    cudaMemcpy(d_nebrTabPtr, nebrTabPtr, (h_nMol+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nebrTab, nebrTab, nebrTabMax*sizeof(int), cudaMemcpyHostToDevice);
}

__global__ void ZeroAccumulators(Mol* inMol,real* inAtomPotential,int inNumMol)//real* accum,int* inCount)
{
  int n=blockIdx.x;//8*gridDim.x *gridDim.x * blockIdx.x + 8*gridDim.y*blockIdx.y + threadIdx.x;
  if(n < inNumMol)
  {
    //atomicAdd(inCount,n);
    inMol[n].ra.x=0.;
    inMol[n].ra.y=0.;
    inMol[n].ra.z=0.;
    inAtomPotential[n]=0.;
    //accum[n]=0.;
    
  }
}

__global__ void ComputeForcesGPU(Mol *inMol,real* inAtomPotential,int* inNebrTabPtr,int*inNebrTab,VecR* inRegion,real inRCut,int inNMol) //int* inCounter,real* accum)
{
    VecR dr, dr12, dr13, w2, w3;
    real aCon = 7.0496, bCon = 0.60222, cr, er, fcVal,
       gCon = 1.2, lCon = 21., p12, p13, ri, ri3, rm,
       rm12, rm13, rm23,rr, rr12, rr13, rrCut;
    int j2, j3, m2, m3;
    //int CURR=blockDim.x*blockDim.y*blockIdx.x+blockIdx.y;
    int CURR=blockIdx.x;
    real fc=0,theta=0;
    real eta[5] = {0.01,0.1,0.5,1.0,10.0};
    real Rs[6] = {1.0,2.0,3.0,4.0,5.0,6.0};
    real g[15] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int zeta[3] = {1,2,3};
    int lambda[2] = {-1,1};
    int nCount=0;
    if(CURR<inNMol)
    {
        //atomicMax(inCounter,CURR);
        //accum[CURR]=0;
        rrCut = inRCut*inRCut - 0.001;
        for (m2 = inNebrTabPtr[CURR]; m2 < inNebrTabPtr[CURR + 1]; m2 ++)
        {
            nCount+=1;
            j2 = inNebrTab[m2];
            
            if (CURR<j2)
            {
                dr.x= inMol[CURR].r.x - inMol[j2].r.x;
                dr.y= inMol[CURR].r.y - inMol[j2].r.y;
                dr.z= inMol[CURR].r.z - inMol[j2].r.z;
                
                if(dr.x >= 0.5 *25.490652)dr.x -=25.490652;         
                if(dr.x< -0.5 * 25.490652)dr.x += 25.490652;
                if(dr.y >= 0.5 *25.490652)dr.y -=25.490652;         
                if(dr.y< -0.5 * 25.490652)dr.y += 25.490652;
                if(dr.z >= 0.5 *25.490652)dr.z -=25.490652;         
                if(dr.z< -0.5 * 25.490652)dr.z += 25.490652;
                
                rr = VLenSq(dr);
                //if(rr<0.1)accum[CURR]=999999;
                if (rr < rrCut && rr>.64)
                {
                     //calculate if i,j conribution to G1(i)and add it to sum for i
                     rm = sqrt (rr);
                     er = exp (1. / (rm - inRCut));
                     ri = 1. / rm;
                     ri3 =  ri*ri*ri;
                     
                     fcVal = aCon * (4. * bCon * Sqr (ri3) +
                                         (bCon * ri3 * ri - 1.) * ri / Sqr (rm - inRCut)) * er;
                     VVSAdd (inMol[CURR].ra, fcVal, dr);
                     VVSAdd (inMol[j2].ra, - fcVal, dr);
                     //uSum += aCon * (bCon * ri3 * ri - 1.) * er;
                     inAtomPotential[CURR]+= aCon * (bCon * ri3 * ri - 1.) *er;
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
                     
                     g[0]+=exp(-eta[0]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                     g[1]+=exp(-eta[0]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                     
                     g[2]+=exp(-eta[1]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                     g[3]+=exp(-eta[4]*(rm-Rs[1])*(rm-Rs[1]))*fc;
                     g[4]+=exp(-eta[1]*(rm-Rs[2])*(rm-Rs[2]))*fc;
                     g[5]+=exp(-eta[1]*(rm-Rs[4])*(rm-Rs[4]))*fc;
                     g[6]+=exp(-eta[1]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                     
                     g[7]+=exp(-eta[2]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                     g[8]+=exp(-eta[2]*(rm-Rs[3])*(rm-Rs[3]))*fc;
                     
                     g[0]+=exp(-eta[0]*(rm-Rs[0])*(rm-Rs[0]))*fc;
                     g[1]+=exp(-eta[0]*(rm-Rs[5])*(rm-Rs[5]))*fc;
                     
                     
                }
            }
        }
        
        //3 body terms
        
        for (m2 = inNebrTabPtr[CURR]; m2 < inNebrTabPtr[CURR + 1] - 1; m2 ++)
        {
           j2 = inNebrTab[m2];
           VSub (dr12, inMol[CURR].r, inMol[j2].r);
           if(dr12.x >= 0.5 *25.490652)dr12.x -=25.490652;         
            if(dr12.x< -0.5 * 25.490652)dr12.x += 25.490652;
            if(dr12.y >= 0.5 *25.490652)dr12.y -=25.490652;         
            if(dr12.y< -0.5 * 25.490652)dr12.y += 25.490652;
            if(dr12.z >= 0.5 *25.490652)dr12.z -=25.490652;         
            if(dr12.z< -0.5 * 25.490652)dr12.z += 25.490652;
           rr12 = VLenSq (dr12);
           if (rr12 < rrCut)
           {
                rm12 = sqrt (rr12);
                VScale (dr12, 1. / rm12);
                for (m3 = m2 + 1; m3 < inNebrTabPtr[CURR + 1]; m3 ++)
                {
                    j3 = inNebrTab[m3];
                    VSub (dr13, inMol[CURR].r, inMol[j3].r);
                    if(dr13.x >= 0.5 *25.490652)dr13.x -=25.490652;         
                    if(dr13.x< -0.5 * 25.490652)dr13.x += 25.490652;
                    if(dr13.y >= 0.5 *25.490652)dr13.y -=25.490652;         
                    if(dr13.y< -0.5 * 25.490652)dr13.y += 25.490652;
                    if(dr13.z >= 0.5 *25.490652)dr13.z -=25.490652;         
                    if(dr13.z< -0.5 * 25.490652)dr13.z += 25.490652;

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
                          
                          g[9]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[0]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          g[10]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[1]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          g[11]+=pow((real)2,1-zeta[0])*pow((real)(1.0+lambda[0]*cos(theta)),zeta[0]) * exp(-eta[2]*(rm12*rm12+rm13*rm13+rm23*rm23)) * fc;
                          
                     }
                }
           }
        }
        
    }
    
}
 
               




// Host code
void doForceIterartion(double inRCut,Mol* inMol,double &outUSum)
{
 
    int counter=0;
    int n;
    real localUSum=outUSum;
    dim3 theSize(250,1,1);
    //printf("x=%f,y=%f,z=%f\n",inMol[100].r.x,inMol[100].r.y,inMol[100].r.z);
    double sum=0;
    int j=0;
    double minR=999,r,maxR=0;
    double avgR=0;
    VecR dr;
    /*for(n=0;n<h_nMol;n++){
        for(j=n+1;j<h_nMol;j++){
            VSub (dr, inMol[n].r, inMol[j].r);
            if(dr.x >= 0.5 *25.490652)dr.x -=25.490652;         
            if(dr.x< -0.5 * 25.490652)dr.x += 25.490652;
            if(dr.y >= 0.5 *25.490652)dr.y -=25.490652;         
            if(dr.y< -0.5 * 25.490652)dr.y += 25.490652;
            if(dr.z >= 0.5 *25.490652)dr.z -=25.490652;         
            if(dr.z< -0.5 * 25.490652)dr.z += 25.490652;
            r = sqrt(VLenSq (dr));
            sum+=r;
            if(r<minR)minR=r;
            if(r>maxR)maxR=r;
        }
    }   
    printf("average r=%f\n",sum/(h_nMol*(h_nMol-1)/2));
    printf("min r=%f\n",minR);
    printf("max r=%f\n",maxR);
    */
    //clear out accumulators
    
    
    //real* theAccum=(real*)malloc(h_nMol*sizeof(real));
    //cudaMemcpy(d_indexSum,&counter, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol,inMol,h_nMol*sizeof(Mol), cudaMemcpyHostToDevice);
    int blocks=8000;
    int threads=1;
    ZeroAccumulators<<<blocks,threads>>>(d_mol,d_atomPotential,8000);//d_accum,d_indexSum);
    //cudaMemcpy(theAccum, d_accum, h_nMol*sizeof(real), cudaMemcpyDeviceToHost);
    sum=0;
    //for(n=0;n<h_nMol;n++) sum+=theAccum[n];
    //printf("Accum after zero=%f \n",sum/h_nMol);
    
    //cudaMemcpy(h_atomPotential, d_atomPotential, h_nMol*sizeof(real), cudaMemcpyDeviceToHost);
    localUSum=0;
    //for(n=0;n<h_nMol;n++) localUSum+=h_atomPotential[n];
    //printf("localUSum after zero=%f\n",localUSum/h_nMol);
    
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
    cudaThreadSynchronize();    
    threads=1;
    counter=0;
    //cudaMemcpy(d_indexSum,&counter, sizeof(int), cudaMemcpyHostToDevice);
    ComputeForcesGPU<<<blocks,threads>>>(d_mol,d_atomPotential, d_nebrTabPtr,d_nebrTab,d_region,inRCut,h_nMol);//d_indexSum,d_accum);

    cudaMemcpy(inMol, d_mol, h_nMol*sizeof(Mol), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_atomPotential, d_atomPotential, h_nMol*sizeof(real), cudaMemcpyDeviceToHost);
    //cudaMemcpy(theAccum, d_accum, h_nMol*sizeof(real), cudaMemcpyDeviceToHost));
    //cudaMemcpy(&counter, d_indexSum, sizeof(int), cudaMemcpyDeviceToHost);
    
    sum=0;
    j=0;
    minR=999,r,maxR=0;
    avgR=0;
    /*for(n=0;n<h_nMol;n++){
        for(j=n+1;j<h_nMol;j++){
            VSub (dr, inMol[n].r, inMol[j].r);
            if(dr.x >= 0.5 *25.490652)dr.x -=25.490652;         
            if(dr.x< -0.5 * 25.490652)dr.x += 25.490652;
            if(dr.y >= 0.5 *25.490652)dr.y -=25.490652;         
            if(dr.y< -0.5 * 25.490652)dr.y += 25.490652;
            if(dr.z >= 0.5 *25.490652)dr.z -=25.490652;         
            if(dr.z< -0.5 * 25.490652)dr.z += 25.490652;
            r = sqrt(VLenSq (dr));
            sum+=r;
            if(r<minR)minR=r;
            if(r>maxR)maxR=r;
        }
    }   
    printf("after forces average r=%f\n",sum/(h_nMol*(h_nMol-1)/2));
    printf("after forces min r=%f\n",minR);
    printf("after forces max r=%f\n",maxR);
    */
    
    //printf("sum of iterations=%d\n",counter);
    sum=0;
    //for(n=0;n<h_nMol;n++) sum+=theAccum[n];
    //printf("Accum after forces=%f\n",sum/999999);
    localUSum=0;
    for(n=0;n<h_nMol;n++) localUSum+=h_atomPotential[n];
    printf("localUSum after forces=%f\n",localUSum/h_nMol);
    outUSum=localUSum;///h_nMol;
    //printf("uSum after zero=%f\n",outUSum);
  
    //free(theAccum);

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



