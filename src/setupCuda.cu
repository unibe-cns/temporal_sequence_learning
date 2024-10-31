
#include <stdio.h>
#include <stdlib.h>
#include "setupCuda.h"
#include "mainCuda.h"
#include <gsl/gsl_randist.h>

void initNetwork(Neuron *N)
{
	gsl_rng *randV;
	time_t t;

	randV = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(randV, SEED_MAKE+123);

	for(int i = 0; i < Vis+Hid; i++)
	{
		N[i].U = EL;
		N[i].Um = EL;
		N[i].rU = PHI_EL;
		N[i].V = EL;
		N[i].Vstar = EL;
		N[i].rV = PHI_EL;
		N[i].Vnew = 0;
		N[i].gEwrite = 0;
		N[i].gEread = 1;
		N[i].gIwrite = 0;
		N[i].gIread = 1;
		N[i].rUwrite = 0;
		N[i].rUread = 1;
		
		N[i].dendDly = gsl_rng_uniform_int(randV,DLY_MAX-49)+50;
		printf("%d, ", N[i].dendDly);
		N[i].gEDly = gsl_rng_uniform_int(randV,GE_DLY-49)+50;
		N[i].gIDly = N[i].gEDly + 250;

		N[i].dVdw = 0;
		
		N[i].gEout = PHI_EL*GE0;
		N[i].gIout = PHI_EL*GI0;
		N[i].riseFlag = 0;
		N[i].gETimer = 0;
		N[i].gITimer = 0; 
		N[i].nGE = 0;
		N[i].nGI = 0;
		N[i].tag = 0;

		for(int t = 0; t < DLY_MAX; t++)
		{
			N[i].rUBuffer[t] = PHI_EL;
		}

		for(int t = 0; t < GE_DLY; t++)
		{
			N[i].gEBuffer[t] = GE0 * PHI_EL;
		}

		for(int t = 0; t < GI_DLY; t++)
		{
			N[i].gIBuffer[t] = GI0 * PHI_EL;
		}
	}
}


void nudgeNeuron(int idT, int idL, Neuron *N)
{
	int nE;
	int nI;
	nE = N[idL].nGE;
	N[idL].nGE = nE+1;
	N[idL].gE[nE] = idT;

	nI = N[idL].nGI;
	N[idL].nGI = nI+1;
	N[idL].gI[nI] = idT;
}

