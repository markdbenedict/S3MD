
/* [[pr_andiffus - diffusion analysis]] */


/*********************************************************************

  (C) 2004  D. C. Rapaport

  This software is copyright material accompanying the book
  "The Art of Molecular Dynamics Simulation", 2nd edition,
  by D. C. Rapaport, published by Cambridge University Press (2004).

**********************************************************************/


#include "in_mddefs.h"

#define BUFF_LEN  1024
#define NDIFF  2

char *txtDiff = "diffusion";

int main (int argc, char **argv)
{
  Prop **diff;
  real *tVal;
  int j, k, n, nData, nDataSkip, nv, nValDiffuse;
  char buff[BUFF_LEN], *bp;
  FILE *fp;

  if ((fp = fopen (argv[1], "r")) == 0) {
    printf ("no file\n");
    exit (0);
  }
  while (1) {
    bp = fgets (buff, BUFF_LEN, fp);
    if (*bp == CHAR_MINUS) break;
    NameVal (nValDiffuse);
  }
  AllocMem2 (diff, NDIFF, nValDiffuse, Prop);
  for (j = 0; j < NDIFF; j ++) {
    for (n = 0; n < nValDiffuse; n ++) PropZero (diff[j][n]);
  }
  AllocMem (tVal, nValDiffuse, real);
  nDataSkip = 3;
  nv = 0;
  nData = 0;
  while (1) {
    if (! (bp = fgets (buff, BUFF_LEN, fp))) break;
    if (! strncmp (bp, txtDiff, strlen (txtDiff))) {
      if (++ nData > nDataSkip) {
        ++ nv;
        for (n = 0; n < nValDiffuse; n ++) {
          bp = fgets (buff, BUFF_LEN, fp);
          tVal[n] = strtod (bp, &bp);
          for (j = 0; j < NDIFF; j ++) {
            diff[j][n].val = strtod (bp, &bp);
            PropAccum (diff[j][n]);
          }
        }
      }
    }
  }
  fclose (fp);
  printf ("%d\n", nv);
  for (n = 0; n < nValDiffuse; n ++) {
    printf ("%9.4f", tVal[n]);
    for (j = 0; j < NDIFF; j ++) {
      PropAvg (diff[j][n], nv);
      printf (" %9.4f %9.4f", PropEst (diff[j][n]));
    }
    printf ("\n");
  }
}

