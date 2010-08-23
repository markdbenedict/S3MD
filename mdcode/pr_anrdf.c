
/* [[pr_anrdf - RDF analysis]] */


/*********************************************************************

  (C) 2004  D. C. Rapaport

  This software is copyright material accompanying the book
  "The Art of Molecular Dynamics Simulation", 2nd edition,
  by D. C. Rapaport, published by Cambridge University Press (2004).

**********************************************************************/


#include "in_mddefs.h"

#define BUFF_LEN  1024
#define NRDF  3

char *txtRdf = "rdf";

int main (int argc, char **argv)
{
  Prop **rdf;
  real *rVal;
  int j, k, n, nData, nDataSkip, nv, sizeHistRdf;
  char buff[BUFF_LEN], *bp;
  FILE *fp;

  if ((fp = fopen (argv[1], "r")) == 0) {
    printf ("no file\n");
    exit (0);
  }
  while (1) {
    bp = fgets (buff, BUFF_LEN, fp);
    if (*bp == CHAR_MINUS) break;
    NameVal (sizeHistRdf);
  }
  AllocMem2 (rdf, NRDF, sizeHistRdf, Prop);
  for (j = 0; j < NRDF; j ++) {
    for (n = 0; n < sizeHistRdf; n ++) PropZero (rdf[j][n]);
  }
  AllocMem (rVal, sizeHistRdf, real);
  nDataSkip = 0;
  nv = 0;
  nData = 0;
  while (1) {
    if (! (bp = fgets (buff, BUFF_LEN, fp))) break;
    if (! strncmp (bp, txtRdf, strlen (txtRdf))) {
      if (++ nData > nDataSkip) {
        ++ nv;
        for (n = 0; n < sizeHistRdf; n ++) {
          bp = fgets (buff, BUFF_LEN, fp);
          rVal[n] = strtod (bp, &bp);
          for (j = 0; j < NRDF; j ++) {
            rdf[j][n].val = strtod (bp, &bp);
            PropAccum (rdf[j][n]);
          }
        }
      }
    }
  }
  fclose (fp);
  printf ("%d\n", nv);
  for (j = 0; j < NRDF; j ++) {
    for (n = 0; n < sizeHistRdf; n ++) PropAvg (rdf[j][n], nv);
  }
  for (n = 0; n < sizeHistRdf; n ++) {
    printf ("%9.4f", rVal[n]);
    for (j = 0; j < NRDF; j ++) printf (" %9.4f %9.4f", PropEst (rdf[j][n]));
    printf ("\n");
  }
}

