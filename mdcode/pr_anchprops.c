
/* [[pr_anchprops - polymer analysis]] */


/*********************************************************************

  (C) 2004  D. C. Rapaport

  This software is copyright material accompanying the book
  "The Art of Molecular Dynamics Simulation", 2nd edition,
  by D. C. Rapaport, published by Cambridge University Press (2004).

**********************************************************************/


#include "in_mddefs.h"

#define BUFF_LEN  1024

char *txtChn = "chain props:";

int main (int argc, char **argv)
{
  Prop vProp[5];
  real bSum[5];
  int chainLen, j, nData, nDataGroup, nDataSkip, nv;
  char *bp, buff[BUFF_LEN];
  FILE *fp;

  if ((fp = fopen (argv[1], "r")) == 0) {
    printf ("no file\n");
    exit (0);
  }
  while (1) {
    bp = fgets (buff, BUFF_LEN, fp);
    if (*bp == CHAR_MINUS) break;
    NameVal (chainLen);
  }
  nDataGroup = 30;
  nDataSkip = 10;
  nData = 0;
  for (j = 0; j < 5; j ++) {
    PropZero (vProp[j]);
    bSum[j] = 0.;
  }
  nv = 0.;
  while (1) {
    if (! (bp = fgets (buff, BUFF_LEN, fp))) break;
    if (! strncmp (bp, txtChn, strlen (txtChn))) {
      if (++ nData > nDataSkip) {
        bp += strlen (txtChn);
        for (j = 0; j < 5; j ++) bSum[j] += strtod (bp, &bp);
        if ((nData - nDataSkip) % nDataGroup == 0) {
          for (j = 0; j < 5; j ++) {
            vProp[j].val = bSum[j] / nDataGroup;
            PropAccum (vProp[j]);
            bSum[j] = 0.;
          }
          ++ nv;
        }
      }
    }
  }
  printf ("%d %d %s\n", chainLen, nData - nDataSkip, txtChn);
  for (j = 0; j < 5; j ++) {
    PropAvg (vProp[j], nv);
    printf (" %9.4f %9.4f\n", PropEst (vProp[j]));
  }
}

