
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "simCuda.h"
#include "mainCuda.h"


/************************************************************************************
  Somatic Inputs Only
  updating Membrane Potential with Hidden Neurons only receiving somatic inputs
  Hidden Neurons are now point neurons, only governed by somatic inputs
  ***********************************************************************************/
__global__ void updateMemPotSoma(Neuron *N, double *d_teacherGE, double *d_teacherGI, int TB, int lambda, int t, int c)
{

	int i = threadIdx.x;
	double gEtot = 0;
	double gItot = 0;

	/************** set gE(t) and gI(t) *****************/
	if(i < Vis)
	{
		gEtot = d_teacherGE[i*TB + t];
		gItot = d_teacherGI[i*TB + t];
	}
	else
	{
		for(int j = 0; j < N[i].nGE; j++)
		{
			gEtot += N[N[i].gE[j]].gEout;
		}
		
		for(int j = 0; j < N[i].nGI; j++)
		{
			gItot += N[N[i].gI[j]].gIout;
		}
	}
	
			
	N[i].Um = gItot*EI/(gItot+gEtot);


	/***************** calculate U(t+1) for Vis and Hid neurons *******************/
	/******* Visible Neurons, receiving dendritic and somatic inputs *******/
	if(i < Vis)
	{
		N[i].U = (1-DT*(GD+GL+lambda*(gEtot+gItot)))*N[i].U
			+ GL*DT*EL
			+ EI*DT*lambda*gItot
			+ GD*DT*N[i].V;
	}
	/******* Hidden Neurons, somatic inputs only *******/
	else
	{
		/**************** somatic inputs only *******************/
		N[i].U = (1-DT*(GL+(gEtot+gItot)))*N[i].U
			+ GL_DT_EL
			+ EI_DT*gItot;
	}

	
	/***************** calculate Vstar(t+1) *************************/
	N[i].Vstar = 1/(GD+GL) * (GL*EL + GD*N[i].V);
	
	
	/***************** calculate V(t+1) *************************/
	N[i].V = (1-GL*DT)*N[i].V + GL*DT*(EL + N[i].Vnew);


	/***************** calculate dVstardw(t+1) *************************/
	N[i].dVdw = (1-GL_DT)*N[i].dVdw + BETA*N[i].rUBuffer[N[i].rUread];
	N[i].Vnew = 0;
}

__global__ void updateMemPotDendrite(Neuron *N, double *d_teacherGE, double *d_teacherGI, int TB, int lambda, int t, int c)
{

	int i = threadIdx.x;
	double gEtot = 0;
	double gItot = 0;

	/************** set gE(t) and gI(t) *****************/
	if(i < Vis)
	{
		gEtot = d_teacherGE[i*TB + t];
		gItot = d_teacherGI[i*TB + t];
	}
	else
	{
		for(int j = 0; j < N[i].nGE; j++)
		{
			gEtot += N[N[i].gE[j]].gEout;
		}
		
		for(int j = 0; j < N[i].nGI; j++)
		{
			gItot += N[N[i].gI[j]].gIout;
		}
	}
	
			
	N[i].Um = gItot*EI/(gItot+gEtot);


	/***************** calculate U(t+1) for Vis and Hid neurons *******************/
	/******* Visible Neurons, receiving dendritic and somatic inputs *******/
	if(i < Vis)
	{
		N[i].U = (1-DT*(GD+GL+lambda*(gEtot+gItot)))*N[i].U
			+ GL*DT*EL
			+ EI*DT*lambda*gItot
			+ GD*DT*N[i].V;
	}
	/******* Hidden Neurons, somatic and dendritic inputs *******/
	else
	{
		/**************** with dendritic inputs *******************/
		N[i].U = (1-DT*(GD+GL+(gEtot+gItot)))*N[i].U
			+ GL*DT*EL
			+ EI*DT*gItot
			+ GD*DT*N[i].V;
	}

	
	/***************** calculate Vstar(t+1) *************************/
	N[i].Vstar = 1/(GD+GL) * (GL*EL + GD*N[i].V);
	
	
	/***************** calculate V(t+1) *************************/
	N[i].V = (1-GL*DT)*N[i].V + GL*DT*(EL + N[i].Vnew);


	/***************** calculate dVstardw(t+1) *************************/
	N[i].dVdw = (1-GL_DT)*N[i].dVdw + BETA*N[i].rUBuffer[N[i].rUread];

	N[i].Vnew = 0;
}

void updateMemPotWrapper(Neuron *N, double *d_teacherGE, double *d_teacherGI, int TB, int lambda, int t, int c)
{
	if(strcmp(MODE, "soma only") == 0)
	{
		updateMemPotSoma<<< 1,(Vis+Hid) >>>(N, d_teacherGE, d_teacherGI, TB, lambda, t, c);
	}
	else if(strcmp(MODE, "with dendrite") == 0)
	{
		updateMemPotDendrite<<< 1,(Vis+Hid) >>>(N, d_teacherGE, d_teacherGI, TB, lambda, t, c);
	}
	else if(strcmp(MODE, "reservoir") == 0)
	{
		updateMemPotDendrite<<< 1,(Vis+Hid) >>>(N, d_teacherGE, d_teacherGI, TB, lambda, t, c);
	}
}

__global__ void updateMemPotReplaySoma(Neuron *N, 
		double *d_teacherGE, double *d_teacherGI, int TB, int lambda, int t, int c)
{

	int i = threadIdx.x;
	double gEtot = 0;
	double gItot = 0;

	/************** set gE(t) and gI(t) *****************/
	if(i < Vis)
	{
		gEtot = d_teacherGE[i*TB + t];
		gItot = d_teacherGI[i*TB + t];
	}
	else
	{
		for(int j = 0; j < N[i].nGE; j++)
		{
			gEtot += N[N[i].gE[j]].gEout;
		}
		
		for(int j = 0; j < N[i].nGI; j++)
		{
			gItot += N[N[i].gI[j]].gIout;
		}
	}
	
			
	N[i].Um = gItot*EI/(gItot+gEtot);


	/***************** calculate U(t+1) for Vis and Hid neurons *******************/
	/******* Visible Neurons, receiving dendritic and somatic inputs *******/
	if(i < Vis)
	{
        N[i].U = (1-DT*(GD+GL+lambda*(gEtot+gItot)))*N[i].U
            + GL*DT*EL
            + EI*DT*lambda*gItot
            + GD*DT*N[i].V;
	}
	/******* Hidden Neurons, somatic inputs only *******/
	else
	{
		N[i].U = (1-DT*(GL+(gEtot+gItot)))*N[i].U
			+ GL_DT_EL
			+ EI_DT*gItot;
	}

	
	/***************** calculate Vstar(t+1) *************************/
	N[i].Vstar = 1/(GD+GL) * (GL*EL + GD*N[i].V);
	
	
	/***************** calculate V(t+1) *************************/
	N[i].V = (1-GL*DT)*N[i].V + GL*DT*(EL + N[i].Vnew);


	/***************** calculate dVstardw(t+1) *************************/
	N[i].dVdw = (1-GL_DT)*N[i].dVdw + BETA*N[i].rUBuffer[N[i].rUread];

	N[i].Vnew = 0;
}

__global__ void updateMemPotReplayDendrite(Neuron *N, 
		double *d_teacherGE, double *d_teacherGI, int TB, int lambda, int t, int c, int *visDisrupted, float disrupt_offset)
{
	int i = threadIdx.x;
	double gEtot = 0;
	double gItot = 0;

	/************** set gE(t) and gI(t) *****************/
	if(i < Vis)
	{
		gEtot = d_teacherGE[i*TB + t];
		gItot = d_teacherGI[i*TB + t];
	}
	else
	{
		for(int j = 0; j < N[i].nGE; j++)
		{
			gEtot += N[N[i].gE[j]].gEout;
		}
		
		for(int j = 0; j < N[i].nGI; j++)
		{
			gItot += N[N[i].gI[j]].gIout;
		}
	}
	
			
	N[i].Um = gItot*EI/(gItot+gEtot);


	/***************** calculate U(t+1) for Vis and Hid neurons *******************/
	/******* Visible Neurons, receiving dendritic and somatic inputs *******/
	if(i < Vis)
	{
		/************** configurable disruption read from file *****************/
		if(DISRUPTION == 1)
		{
			if(c == 4 && visDisrupted[i] == 1)
			{
				N[i].U = EL + disrupt_offset;
			}
			else
			{
				N[i].U = (1-DT*(GD+GL+lambda*(gEtot+gItot)))*N[i].U
					+ GL*DT*EL
					+ EI*DT*lambda*gItot
					+ GD*DT*N[i].V;
			}
		}
		else
		{
			N[i].U = (1-DT*(GD+GL+lambda*(gEtot+gItot)))*N[i].U
				+ GL*DT*EL
				+ EI*DT*lambda*gItot
				+ GD*DT*N[i].V;
		}
	}
	/******* Hidden Neurons, somatic and dendritic inputs *******/
	else
	{
        /**************** with dendritic inputs *******************/
        N[i].U = (1-DT*(GD+GL+(gEtot+gItot)))*N[i].U
            + GL*DT*EL
            + EI*DT*gItot
            + GD*DT*N[i].V;
	}

	
	/***************** calculate Vstar(t+1) *************************/
	N[i].Vstar = 1/(GD+GL) * (GL*EL + GD*N[i].V);
	
	
	/***************** calculate V(t+1) *************************/
	N[i].V = (1-GL*DT)*N[i].V + GL*DT*(EL + N[i].Vnew);


	/***************** calculate dVstardw(t+1) *************************/
	N[i].dVdw = (1-GL_DT)*N[i].dVdw + BETA*N[i].rUBuffer[N[i].rUread];

	N[i].Vnew = 0;
}

void updateMemPotReplayWrapper(Neuron *N, double *d_teacherGE, double *d_teacherGI, int TB, int lambda, int t, int c, int *visDisrupted, float disrupt_offset)
{
	if(strcmp(MODE, "soma only") == 0)
	{
		updateMemPotReplaySoma<<< 1,(Vis+Hid) >>>(N, d_teacherGE, d_teacherGI, TB, lambda, t, c);
	}
	else if(strcmp(MODE, "with dendrite") == 0 or strcmp(MODE, "reservoir") == 0)
	{
		updateMemPotReplayDendrite<<< 1,(Vis+Hid) >>>(N, d_teacherGE, d_teacherGI, TB, lambda, t, c, visDisrupted, disrupt_offset);
	}
}


