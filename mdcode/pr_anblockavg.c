
/* [[pr_anblockavg - block-averaged variance]] */


/*********************************************************************

  (C) 2004  D. C. Rapaport

  This software is copyright material accompanying the book
  "The Art of Molecular Dynamics Simulation", 2nd edition,
  by D. C. Rapaport, published by Cambridge University Press (2004).

**********************************************************************/


#include "in_mddefs.h"

#define BUFF_LEN  1024

int main ()
{
  Prop w;
  real wVal[20000];
  int bSize, n, nv, nvh;
  char *bp, buff[BUFF_LEN];

  nv = 0.;
  while (1) {
    if (! (bp = fgets (buff, BUFF_LEN, stdin))) break;
    wVal[nv] = strtod (bp, &bp);
    ++ nv;
  }
  printf ("%d\n", nv);
  nvh = nv;
  bSize = 1;
  while (nvh > 1) {
    PropZero (w);
    for (n = 0; n < nvh; n ++) {
      w.val = wVal[n];
      PropAccum (w);
    }
    PropAvg (w, nvh);
    if (bSize == 1) printf ("av stdev %.4f %.4f\n", PropEst (w));
    printf ("%6d %6d %.4f\n", nvh, bSize, w.sum2 / sqrt (nvh - 1.));
    for (n = 0; n < nvh / 2; n ++)
       wVal[n] = 0.5 * (wVal[2 * n] + wVal[2 * n + 1]);
    nvh /= 2;
    bSize *= 2;
  }
}

