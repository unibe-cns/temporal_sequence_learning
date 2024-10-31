
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <error.h>
#include "mainCuda.h"
#include "setupCuda.h"
#include "simCuda.h"
#include <gsl/gsl_randist.h>

int TIMEBINS;
double **teacher;

FILE *ratesUF;
FILE *UmF;
FILE *weightsF;
FILE *weightsF_pre;
FILE *nudgeF;
FILE *nOutF;
FILE *nInF;

FILE *NetStructureF;


gsl_rng *randVar;

void initFiles()
{
	ratesUF = fopen(FILENAME_RATEU, "a+");
    if (ratesUF == NULL)
    {
      printf("Filename: %s \n", FILENAME_RATEU);
      printf("Error with file creation ratesUF. Value of errno: %d\n", errno);
    }

	UmF = fopen(FILENAME_UM, "a+");
    if (UmF == NULL)
    {
      printf("Filename: %s \n", FILENAME_UM);
      printf("Error with file creation UmF. Value of errno: %d\n", errno);
    }

	weightsF = fopen(FILENAME_W, "a+");
    if (weightsF == NULL)
    {
      printf("Filename: %s \n", FILENAME_W);
      printf("Error with file creation weightsF. Value of errno: %d\n", errno);
    }

	weightsF_pre = fopen(FILENAME_W_pre, "a+");
    if (weightsF_pre == NULL)
    {
      printf("Filename: %s \n", FILENAME_W_pre);
      printf("Error with file creation weightsF_pre. Value of errno: %d\n", errno);
    }

	nudgeF = fopen(FILENAME_N, "a+");
    if (nudgeF == NULL)
    {
      printf("Filename: %s \n", FILENAME_N);
      printf("Error with file creation nudgeF. Value of errno: %d\n", errno);
    }

	nOutF = fopen(FILENAME_NOUT, "a+");
    if (nOutF == NULL)
    {
      printf("Filename: %s \n", FILENAME_NOUT);
      printf("Error with file creation nOutF. Value of errno: %d\n", errno);
    }

	nInF = fopen(FILENAME_NIN, "a+");
    if (nInF == NULL)
    {
      printf("Filename: %s \n", FILENAME_NIN);
      printf("Error with file creation nInF. Value of errno: %d\n", errno);
    }

	
	randVar = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(randVar, SEED_MAKE+42);     // not use same rng as as in network setup but still deterministic one

}


void confNetworkRandCmplt(Neuron *N, double *d_teacherGE, double *d_teacherGI)
{
	double pOut = P1;
	double pIn = Q1;
	double p0 = P0;
	time_t t;

	gsl_rng_env_setup();
	randVar = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(randVar, SEED_MAKE);

	int nudgeN = 0;
	int lenTeachers = Vis;
	int teach;
	int id;
	int connect = 0;
	int student = 0;
	int *teachers;

	// add all visible neurons to teachers
	teachers = (int*) malloc(sizeof(int)*Vis);

	printf("p: %f, q: %f, p0: %f\n", pOut, pIn, p0);

	for(int i = 0; i < Vis; i++)
	{
		teachers[i] = i;
		N[i].tag = 1;
	}

    printf("Set tags for visible neurons\n");

	while(lenTeachers > 0)
	{
		// get next teacher
		id = (int)round(gsl_rng_uniform(randVar)*(lenTeachers-1));
		teach = teachers[id];
		//N[teach].tag = 1;

		// delete id element from teachers
		for(int i = id; i < lenTeachers-1; i++)
		{
			teachers[i] = teachers[i+1];
		}
		lenTeachers--;
		teachers = (int*)realloc(teachers,sizeof(int)*lenTeachers);

		// get number of outgoing connections nudgeN
		nudgeN = 0;
		if(gsl_rng_uniform(randVar) >= p0)
		{
			nudgeN = 1;
			while(gsl_rng_uniform(randVar) < pow(pOut,nudgeN))
			{
				nudgeN++;
			}
		}
		fprintf(nOutF, "%d, ", nudgeN);         // WARNING: this is just for statistics, not all nrns in that file and also nrn sequence not kept!!

		// make nudgeN outgoing connections
		for(int j = 0; j < nudgeN; j++)
		{
			connect = 0;

			while(connect == 0)
			{
				student = (int)round(gsl_rng_uniform(randVar)*(Hid-1));
				student += Vis;

				if(student != teach)
				{
					// printf("pIn: %f, nGE: %d, p^n: %f\n",pIn, N[student].nGE, pow(pIn,N[student].nGE));
					if(gsl_rng_uniform(randVar) < pow(pIn, N[student].nGE))
					{
						nudgeNeuron(teach,student,N);
						fprintf(nudgeF,"%d,%d;", teach, student);
						//printf("%d,%d\n", teach, student);
						connect = 1;

						// add student to teachers, if it did not make connections already
						if(N[student].tag == 0)
						{
							lenTeachers++;
							teachers = (int*)realloc(teachers,sizeof(int)*lenTeachers);
							teachers[lenTeachers-1] = student;
							N[student].tag = 1;
						}
					}
				}
			}
		}
	}

    printf("done setting nudging\n");
	for(int i = Vis; i < Vis+Hid; i++)
	{
		fprintf(nInF, "%d, ", N[i].nGE);         // WARNING: this is just for statistics, not all nrns in that file and also nrn sequence not kept!!
	}	
    setupDendriticConnections(N);
    printf("done setting dendritic weights\n");
}


__global__ void resetBuffer(Neuron *N)
{
	int idx = blockIdx.x;
	int i = threadIdx.x;

	N[idx].gIBuffer[i] = 3.5;
}

__global__ void resetGE(Neuron *N)
{
	int idx = threadIdx.x;

	N[idx].gEout = 0.35;

}

__global__ void calcV(Neuron *N)
{
	int idx = blockIdx.x;
	int i = threadIdx.x;

	if(N[idx].dend[i] == 1)
	{
		atomicAdd(&N[idx].Vnew, N[i].rUBuffer[N[i].rUread] * N[idx].w[i]);
	}

}

__global__ void updateWeights(Neuron *N)
{
	int idx = blockIdx.x;
	int i = threadIdx.x;

    // using dVdw from presynaptic neuron, careful, this only works if timeconstants of
    // pre and post are the same. otherwise, lowpass over r_pre (dVdw) must be
    // calculated with tau of post-nrn not with tau of pre-nrn
    // using tau-pre allows for more efficient calc, because can be variable of
    // pre-neuron, otherwise it would need to be synapse specific
    if(N[idx].dend[i] == 1 && idx < Vis && i < Vis)
    {
        atomicAdd(&N[idx].w[i], ETAV * (N[idx].rU - N[idx].rV) * N[i].dVdw);

    }
    else if(N[idx].dend[i] == 1)
    {
        atomicAdd(&N[idx].w[i], ETAH * (N[idx].rU - N[idx].rV) * N[i].dVdw);
    }

}

__global__ void updateWeightsReservoir(Neuron *N)
{
	int idx = blockIdx.x;
	int i = threadIdx.x;

    // Weight updates for Reservoir comparison (only update Hid -> Vis and Vis -> Vis)
    if(N[idx].dend[i] == 1 && idx < Vis && i < Vis)
    {
        atomicAdd(&N[idx].w[i], ETAV * (N[idx].rU - N[idx].rV) * N[i].dVdw);

    }
    else if(N[idx].dend[i] == 1 && idx < Vis && i >= Vis)
    {
        atomicAdd(&N[idx].w[i], ETAH * (N[idx].rU - N[idx].rV) * N[i].dVdw);
    }

}

__global__ void updateRates(Neuron *N)
{
	int i = threadIdx.x;


	N[i].rU = 1.0 / (1.0 + exp(0.3 * (-58 - N[i].U)));
	N[i].rV = 1.0 / (1.0 + exp(0.3 * (-58 - N[i].Vstar)));

	N[i].rUBuffer[N[i].rUwrite] = N[i].rU;
	N[i].rUwrite = (N[i].rUwrite + 1) % N[i].dendDly;
	N[i].rUread = (N[i].rUwrite + 1) % N[i].dendDly;
}
	

__global__ void calcGEOut(Neuron *N)
{
	int i = threadIdx.x;

	double sollGE;
	
	if(N[i].rU < PHI_EL)
	{
		sollGE = GE0 * PHI_EL;
	}
	else
	{
		sollGE = GE0 * N[i].rU;
	}
	
	N[i].gEBuffer[N[i].gEwrite] = sollGE;
	N[i].gEout = N[i].gEBuffer[N[i].gEread];
	N[i].gEwrite = (N[i].gEwrite + 1) % N[i].gEDly;
	N[i].gEread = (N[i].gEwrite + 1) % N[i].gEDly;
}

	
__global__ void calcGIOut(Neuron *N)
{
	int i = threadIdx.x;

	double sollGI;
	//int idxOld;
	
	if(N[i].rU < PHI_EL)
	{
		sollGI = GI0 * PHI_EL;
	}
	else
	{
		sollGI = GI0 * N[i].rU;
	}
	N[i].gIBuffer[N[i].gIwrite] = sollGI;
	N[i].gIout = N[i].gIBuffer[N[i].gIread];
	N[i].gIwrite = (N[i].gIwrite + 1) % N[i].gIDly;
	N[i].gIread = (N[i].gIwrite + 1) % N[i].gIDly;
}


void writeData(Neuron *N)
{
    // buffer length: num_neurons * 13 because: %f has 6 digits after . +
    // dot itself + 3 digits before . + potentially a minus + space + comma
	int len = 13*(Vis+Hid);
	char bufRU[len];
	char bufUM[len];
	int posRU = 0;
	int posUM = 0;


	int num_neurons = 0;
	
	if(PLOT_CMPLT_NET) {
		num_neurons = Vis+Hid;
	}
	else {
		num_neurons = Vis;
	}

	for(int i = 0; i < num_neurons; i++)	
	{
		if(PlotRU == 1)
		{
			posRU += snprintf(bufRU+posRU, len-posRU, "%f, ", N[i].rU); 
		}

        // only record matching potential for vis neurons because it is only needed
        // to calculate target signal during eval, Um of hiddens not used
		if(PlotUM == 1 and i < Vis)
		{
			//fprintf(UmF,"%f, ", N[i].Um);
			posUM += snprintf(bufUM+posUM, len-posUM, "%f, ", N[i].Um); 
		}
		/*if(N[i].nGE > 0)
		{
			fprintf(GeF,"%f, ", N[N[i].gE[0]].gEout);
		}
		if(N[i].nGI > 0)
		{
			fprintf(GiF,"%f, ", N[N[i].gI[0]].gIout);
		}*/

	}

    // printing with %s will only print the part of the buffer that is actually a string
    // so it does not matter if buffer was fully filled up or not
	fprintf(ratesUF, "%s", bufRU);
	fprintf(UmF, "%s", bufUM);

	//return error;
}



void writeWeights(Neuron *N, bool pre)
{
    if (pre)
    {
        printf("write pre-trainings som-dend weights to file\n");
    }
    else
    {
        printf("write post-trainings som-dend weights to file\n");
    }
    fflush(stdout);
	for(int k = 0; k < Vis+Hid; k++)
	{
		
		for(int i = 0; i < Vis+Hid; i++)
		{
            if (pre)
            {
                fprintf(weightsF_pre,"%f, ", N[k].w[i]);
            }
            else
            {
                fprintf(weightsF,"%f, ", N[k].w[i]);
            }
		}
	}
}


void calcVMaster(Neuron *Net)
{
	calcV<<< (Vis+Hid), (Vis+Hid) >>>(Net);
}


double phi(double u) 
{
	double r;

	r = 1.0 / (1.0 + exp(0.3 * (-58 - u)));
	return r;	
}


void setupDendriticConnections(Neuron *N)
{
	gsl_rng_env_setup();
	randVar = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(randVar, SEED_MAKE+17);     // not use same rng as as in scaffold setup but still deterministic one

	//all to all somato-dendritic connectivity
	if(strcmp(CONNECT, "all-to-all") == 0 || strcmp(CONNECT, "innervation ratio") == 0 || strcmp(CONNECT, "sparse") == 0)
	{
        printf("Network setup, all-to-all, innervation ratio or sparse mode\n");
		for(int i = 0; i < Vis+Hid; i++)
		{
			for(int j = 0; j < Vis+Hid; j++)
			{
				if(strcmp(MODE, "soma only") != 0)      // if NOT soma only
				{
					if(i != j)
					{
                        // check if connection is made in case of limited innveration of hidden
                        bool connected = true;
                        if(strcmp(CONNECT, "innervation ratio") == 0 && i >= Vis && j < Vis)
                        {
                            if (gsl_rng_uniform(randVar) <= InnervRatio)
                            {
                                N[i].dend[j] = 1;
                            }
                            else
                            {
                                N[i].dend[j] = 0;
                                connected = false;
                            }
                        }
                        else if(strcmp(CONNECT, "sparse") == 0)
                        {
                            if (gsl_rng_uniform(randVar) <= ConnDensity)
                            {
                                N[i].dend[j] = 1;
                            }
                            else
                            {
                                // in case of sparse connections PROTECT_SCAFFOLD prevents
                                // som-dend connections that match scaffold from being deleted
                                if (PROTECT_SCAFFOLD)
                                {
                                    // i -> post, j --> pre
                                    int n_nudge_partners = N[i].nGE;
                                    bool protect = false;
                                    for(int pre_idx = 0; pre_idx < n_nudge_partners; pre_idx++)
                                    {
                                        if (N[i].gE[pre_idx] == j)
                                        {
                                            protect = true;
                                        }
                                    }
                                    if (protect)
                                    {
                                        N[i].dend[j] = 1;
                                    }
                                    else
                                    {
                                        N[i].dend[j] = 0;
                                        connected = false;
                                    }
                                }
                                else
                                {
                                    N[i].dend[j] = 0;
                                    connected = false;
                                }
                            }
                        }
                        else
                        {
                            N[i].dend[j] = 1;
                        }

                        if (i < Vis && j < Vis)
                        {
                            if (connected)
                            {
                                N[i].w[j] = gsl_ran_gaussian(randVar, W_INIT_VAR_V_TO_V) + W_INIT_MEAN_V_TO_V;
                            }
                            else
                            {
                                N[i].w[j] = 0.0;
                            }
                        }
                        else if (i < Vis && j >= Vis)
                        {
                            if (connected)
                            {
                                N[i].w[j] = gsl_ran_gaussian(randVar, W_INIT_VAR_H_TO_V) + W_INIT_MEAN_H_TO_V;
                            }
                            else
                            {
                                N[i].w[j] = 0.0;
                            }
                        }
                        else if (i >= Vis && j >= Vis)
                        {
                            if (connected)
                            {
                                N[i].w[j] = gsl_ran_gaussian(randVar, W_INIT_VAR_H_TO_H) + W_INIT_MEAN_H_TO_H;
                            }
                            else
                            {
                                N[i].w[j] = 0.0;
                            }
                        }
                        else if (i >= Vis && j < Vis)
                        {
                            if (connected)
                            {
                                N[i].w[j] = gsl_ran_gaussian(randVar, W_INIT_VAR_V_TO_H) + W_INIT_MEAN_V_TO_H;
                            }
                            else
                            {
                                N[i].w[j] = 0.0;
                            }
                        }
                        else
                        {
                            printf("ERROR: Should not end up here during network setup, something went wrong! Check!\n");
                        }
					//N[i].w[j] = 0;
					}
					else
					{
						N[i].dend[j] = 0;
						N[i].w[j] = 0.0;
					}
				}
				else
				{
                    // potentially deprecated!
					if(i < Vis && i != j)
					{
						N[i].dend[j] = 1;
						N[i].w[j] = gsl_ran_gaussian(randVar, 0.5);
					}
					else
					{
						N[i].dend[j] = 0;
						N[i].w[j] = 0.0;
					}
				}
			}

		}
	}
}

/*****************************************************************************************************

  running the simulations in main.c
  main iteration over training cycles and timebin

*****************************************************************************************************/

int main()
{
	/*************************************************************
	  variable init
	*************************************************************/

	NetStructureF = fopen(FILE_NETSTRUCTURE, "a+");
    if (NetStructureF == NULL)
    {
      printf("Filename: %s \n", FILE_NETSTRUCTURE);
      printf("Error with file creation NetStructureF. Value of errno: %d\n", errno);
    }


	double *d_teacherGI;
	double *d_teacherGE;
	double *Um;

	int lambda = 1;
	int ratio;

	Neuron *Net;	

	/*************************************************************
	  variable allocation and initialization procedures
	*************************************************************/
	initFiles();
	printf("CUDA error: %s\n",  cudaGetErrorString(cudaMallocManaged(&Net, (Vis+Hid)*sizeof(Neuron))));

    TIMEBINS = initTeacherCounting(&d_teacherGE, &d_teacherGI);
    
    Um = (double *) malloc(Vis * TIMEBINS * sizeof(double));
    initNetwork(Net);
    confNetworkRandCmplt(Net, d_teacherGE, d_teacherGI);
    writeNetStruct(Net);
	
    // init teaching matching potential to reasonable values (not random malloc)
	for(int i = 0; i < Vis * TIMEBINS; i++)
	{
		Um[i] = phi(d_teacherGI[i]*EI/(d_teacherGI[i] + d_teacherGE[i]));
	}

    // write also pre-training som-dend weights to illustrate training effect
	writeWeights(Net, true);

	/*************************************************************
	  starting training
	*************************************************************/
	printf("Starting training\n");
    // do not print at every training cycle (due to outfile printing on cluster)
    int print_freq = 100;
    int print_counter = 1;
	for(int c = -1; c < TRAININGCYCLES; c++)
	{

		/*************************************************************
		  progress bar generation
		*************************************************************/
		ratio = ((double)c+1)/TRAININGCYCLES*100;
        if (print_counter >= print_freq)
        {
            if(ratio < 100)
            {
                printf("\rTraining: [ %d%% finished", (int)ratio);
                for(int p = 0; p < 20; p++)
                {
                    if(((double)c+1)/TRAININGCYCLES*100 > p*(100/20))
                        printf("*");
                    else
                        printf(" ");
                }
                printf("]\n");
                fflush(stdout);
            }
            else
            {
                printf("\rTraining: [%d%% finished", (int)ratio);
                for(int p = 0; p < 20; p++)
                {
                        printf("*");
                }
                printf("]\n");
            }
            print_counter = 1;
        }
        else
        {
            print_counter++;
        }

		/*************************************************************/

		/**** lambda set to 0 when training patterns are interrupted by free runs ****/	
		if(c%PlotFreq == 0)
		{
			lambda = 0;             // flag meaning free run (lambda=0) switches nudging of visible neurons on or off
		}
	

		/*************************************************************
		  iterating through all the timebins
		*************************************************************/
		for(int t = 0; t < TIMEBINS; t++)
		{
			calcVMaster(Net);       // dendritic potential
			cudaDeviceSynchronize();
		
			updateMemPotWrapper(Net, d_teacherGE, d_teacherGI, TIMEBINS, lambda, t, c);             // somatic potential
			cudaDeviceSynchronize();

			updateRates<<< 1,(Vis+Hid) >>>(Net);
			cudaDeviceSynchronize();
				
			/************** updating weights all-to-all ***************/	
	        if(strcmp(MODE, "reservoir") == 0)
            {
                updateWeightsReservoir<<< (Vis+Hid),(Vis+Hid) >>>(Net);
            }
            else
            {
                updateWeights<<< (Vis+Hid),(Vis+Hid) >>>(Net);
            }

			cudaDeviceSynchronize();
			
			calcGEOut<<< 1,(Vis+Hid) >>>(Net);          // calc new nudge conds for next time steps
			calcGIOut<<< 1,(Vis+Hid) >>>(Net);
			cudaDeviceSynchronize();

            // if this is a free run, then record rates
			if(c%PlotFreq == 0 && t%2 == 0)
			{
				writeData(Net);
			}
            // if the next cycle will be a free run, then record rates to have the nudged reference
			else if((c+1)%PlotFreq == 0 && t%2 == 0 && WriteNudge == 1)
			{
				writeData(Net);
			}

			if((c == TRAININGCYCLES-1 && t == TIMEBINS-1))
			{
				writeWeights(Net, false);
			}

		}
		lambda =1;
	}
	
	
	/*************************************************************
	  starting Replay
	*************************************************************/
	for(int c = 0; c < REPLAYCYCLES; c++)
	{

		/*************************************************************
		  progress bar generation
		*************************************************************/
		ratio = ((double)c+1)/REPLAYCYCLES*100;
		if(ratio < 100)
		{
			printf("\rReplay: [ %d%% finished", (int)ratio);
			for(int p = 0; p < 20; p++)
			{
				if(((double)c+1)/REPLAYCYCLES*100 > p*(100/20))
					printf("*");
				else
					printf(" ");
			}
			printf("]\n");
			fflush(stdout);
		}
		else
		{
			printf("\rReplay: [%d%% finished", (int)ratio);
			for(int p = 0; p < 20; p++)
			{
					printf("*");
			}
			printf("]\n");
		}

		/*************************************************************/

		/**** lambda is zero during replay phase apart from first 3 phases ****/	
		if(c < 3)
		{
			lambda = 1;
		}

        // only relevant if replay disruption is activated
        int max_disrupt_lines = 5;
        float disrupt_offset_el = 0;
        int t_starts[max_disrupt_lines] = {-1, -1, -1, -1, -1};
        int t_ends[max_disrupt_lines] = {-1, -1, -1, -1, -1};
        int nrn_starts[max_disrupt_lines] = {-1, -1, -1, -1, -1};
        int nrn_ends[max_disrupt_lines] = {-1, -1, -1, -1, -1};

        if (c == 4 && DISRUPTION == 1)
        {
            initDisruption(t_starts, t_ends, nrn_starts, nrn_ends, max_disrupt_lines, &disrupt_offset_el);
        }
	
		/*************************************************************
		  iterating through all the timebins
		*************************************************************/
        // array of flags that indicate wether vis nrn is disrupted for certain timestep
        int *visDisrupted = NULL;
		for(int t = 0; t < TIMEBINS; t++)
		{
            if (c == 4 && DISRUPTION == 1)
            {
                cudaMallocManaged(&visDisrupted, Vis*sizeof(int));
                for (int i = 0; i < Vis; i++)
                {
                    visDisrupted[i] = 0;
                }

                for (int i = 0; i < max_disrupt_lines; i++)
                {
                    if (t_starts[i] != -1 && t >= t_starts[i] && t < t_ends[i])
                    {
                        for (int idx = nrn_starts[i]; idx < nrn_ends[i]; idx++)
                        {
                            visDisrupted[idx] = 1;
                        }
                    }
                }
            }
			calcVMaster(Net);
			cudaDeviceSynchronize();

		
			updateMemPotReplayWrapper(Net, d_teacherGE, d_teacherGI, TIMEBINS, lambda, t, c, visDisrupted, disrupt_offset_el);
			cudaDeviceSynchronize();

			updateRates<<< 1,(Vis+Hid) >>>(Net);
			cudaDeviceSynchronize();
				
			/************** updating weights all-to-all ***************/	
	        if(strcmp(MODE, "reservoir") == 0)
            {
                updateWeightsReservoir<<< (Vis+Hid),(Vis+Hid) >>>(Net);
            }
            else
            {
                updateWeights<<< (Vis+Hid),(Vis+Hid) >>>(Net);
            }

			cudaDeviceSynchronize();
			
			calcGEOut<<< 1,(Vis+Hid) >>>(Net);
			calcGIOut<<< 1,(Vis+Hid) >>>(Net);
			cudaDeviceSynchronize();

			if(t%2 == 0)
			{
				writeData(Net);
			}
		}

		lambda = 0;
        if (c == 4 && DISRUPTION == 1)
        {
            cudaFree(visDisrupted);
        }
	}

	cudaFree(Net);
	cudaFree(d_teacherGE);
	cudaFree(d_teacherGI);
	free(teacher);
	
	fclose(NetStructureF);
}


void initDisruption(int *t_starts, int *t_ends, int *nrn_starts, int *nrn_ends, int max_disrupt_lines, float *disrupt_offset_el)
{
    FILE *disruptF;
    printf("Load pattern disruption from from file %s \n", FILENAME_DISRUPTION);
    disruptF = fopen(FILENAME_DISRUPTION, "r");
    if (disruptF == NULL)
    {
      printf("Filename: %s \n", FILENAME_DISRUPTION);
      printf("Error with loading file of pattern disruption. Value of errno: %d\n", errno);
    }

    // read the max_disrupt_lines lines from disruption file and enter data into arrays
    char *line = NULL;
    size_t len = 0;
    int line_idx = 0;
    while(getline(&line, &len, disruptF) != -1) {
        if (line_idx >= max_disrupt_lines)
        {
            printf("Error: trying to read too many lines from disruption file (reading line number: %d max_allowed: %d)\n", line_idx + 1, max_disrupt_lines);
        }
        else
        {
            char label_tstart[10], label_tend[10], label_nrnstart[15], label_nrnend[15], label_offset[15];
            int tstart, tend, nrnstart, nrnend;
            float offset;
            sscanf(line, "%s %d %s %d %s %d %s %d %s %f", label_tstart, &tstart, label_tend, &tend, label_nrnstart, &nrnstart, label_nrnend, &nrnend, label_offset, &offset);
            printf("Reading lines in disruption file (line %d):\n", line_idx);
            printf("%s %d %s %d %s %d %s %d %s %f\n", label_tstart, tstart, label_tend, tend, label_nrnstart, nrnstart, label_nrnend, nrnend, label_offset, offset);
            *disrupt_offset_el = offset;
            if (tstart < 0 || tstart > TIMEBINS || tstart > tend)
            {
                printf("Error: t_start must be >= 0, <= pattern length %d, <= t_end %d, but is %d \n", TIMEBINS, tend, tstart);
            }
            else if (tend < 0 || tend > TIMEBINS)
            {
                printf("Error: t_end must be >= 0 and <= pattern length %d, but is %d \n", TIMEBINS, tend);
            }
            else if (nrnstart < 0 || nrnstart >= Vis || nrnstart > nrnend)
            {
                printf("Error: nrn_start must be >= 0, < number of visible neurons %d and <= nrn_end %d, but is %d \n", Vis, nrnend, nrnstart);
            }
            else if (nrnend < 0 || nrnend > Vis)
            {
                printf("Error: nrn_end must be >= 0 and < number of visible neurons %d , but is %d \n", Vis, nrnend);
            }
            else
            {
                t_starts[line_idx] = tstart;
                t_ends[line_idx] = tend;
                nrn_starts[line_idx] = nrnstart;
                nrn_ends[line_idx] = nrnend;
            }
        }
        line_idx++;
    }
}


int initTeacherCounting(double** teacherGE, double** teacherGI)
{
    FILE *patternF;
    printf("Load target pattern from file %s \n", FILENAME_PATTERN);
    patternF = fopen(FILENAME_PATTERN, "r");
    if (patternF == NULL)
    {
      printf("Filename: %s \n", FILENAME_PATTERN);
      printf("Error with loading file of target pattern. Value of errno: %d\n", errno);
    }

    char hashtag[2], name_label[10], name[100], dim1label[10], dim2label[10];
    int dim1, dim2;
    fscanf(patternF, "%s %s %s %s %d %s %d", hashtag, name_label, name, dim1label, &dim1, dim2label, &dim2);
    printf("Read fields of header as |%s| |%s| |%s| |%s| |%d| |%s| |%d|\n", hashtag, name_label, name, dim1label, dim1, dim2label, dim2);

    if (dim1 != Vis){
        printf("Error: Loaded pattern is for %d visible neurons, but makefile configured %d!\n", dim1, Vis);
    }

    double l = 0.6;
    int stateN = dim2;
    int stateL = 10/DT;
    int patternLength = dim2 * stateL;

    cudaMallocManaged(teacherGI, Vis * patternLength * sizeof(double));
    cudaMallocManaged(teacherGE, Vis * patternLength * sizeof(double));

    int **pattern;
    pattern = (int**) malloc(int(dim1) * sizeof(int *));
    for(int i = 0; i < Vis; i++)
    {
        pattern[i] = (int*) malloc(dim2 * sizeof(int));
    }

    // read remainder of first line
    char *line = NULL;
    size_t len = 0;
    getline(&line, &len, patternF);

    // read the following lines and enter data into pattern
    int nrn_idx = 0;
    while(getline(&line, &len, patternF) != -1) {
        int read_index = 0;
        int step_idx = 0;
        int max = strlen(line);
        while (read_index < max){
            int translated = line[read_index] - '0';
            pattern[nrn_idx][step_idx] = translated;
            read_index = read_index + 2;
            step_idx++;
        }
        nrn_idx++;
    }

    printf("Printing read target pattern: %s \n", name);
    for (int i = 0; i < Vis; i++){
        for (int j = 0; j < dim2; j++){
            printf("%d ", pattern[i][j]);
        }
        printf("\n");
    }
	
	teacher = (double**) malloc(int(Vis) * sizeof(double *));
	for(int i = 0; i < Vis; i++)
	{
		teacher[i] = (double*) malloc(patternLength * sizeof(double));
	}

	int current;
	int prev;
	int next;
	
	for(int id = 0; id < int(Vis); id++)
	{	
		for(int s = 0; s < stateN; s++)
		{
			current = pattern[id][s];

			if(s > 0)
			{
				prev = pattern[id][s-1];
			}
			else
			{
				prev = pattern[id][stateN-1];
			}

			if(s < stateN-1)
			{
				next = pattern[id][s+1];
			}
			else
			{
				next = pattern[id][0];
			}


			for(int t = 0; t < stateL; t++)
			{
				if ((current == 0) && (next == 1))
				{
					if (t >= 2/3.0*stateL)
					{
						teacher[id][t+s*stateL] = rampUp((t - stateL)*DT);
					}
					else
					{
						teacher[id][t+s*stateL] = EL;
					}
				}
				else if ((current == 0) && (next == 0))
				{
					teacher[id][t+s*stateL]= EL;
				}
				else if ((current == 1) && (prev == 0))
				{
					if ((next == 0) && (t <= 1/2.0*stateL))
					{
						teacher[id][t+s*stateL] = rampUp(t*DT);
					}
					else if ((next == 1) && (t <= 2/3.0*stateL))
					{
						teacher[id][t+s*stateL] = rampUp(t*DT);
					}
					else if ((next == 1) && (t > 2/3.0*stateL))
					{
						teacher[id][t+s*stateL]= rampUp(10);
					}
				}

				if ((current == 1) && (next == 0))
				{
					if ((prev == 0) && (t >= 1/2.0*stateL))
					{
						teacher[id][t+s*stateL] = rampDown((t -stateL)*DT);
					}

					if ((prev == 1) && (t >= 1/3.0*stateL)) 
					{
						teacher[id][t+s*stateL] = rampDown((t -stateL)*DT);
					}

					if ((prev == 1) && (t < 1/3.0*stateL))
					{
						teacher[id][t+s*stateL] = rampUp(10);
					}
				}

				if ((current == 1) && (prev == 1) && (next == 1))
				{
					teacher[id][t+s*stateL] = rampUp(10);
				}

				if ((current == 0) && (prev == 1))
				{
					if (t <= 1/3.0*stateL)
					{
						teacher[id][t+s*stateL] = rampDown(t*DT);
					}
				}
			}
		}
	}

	for(int id = 0; id < int(Vis); id++)
	{
		for(int t = 0; t < patternLength; t++)
		{
			(*teacherGE)[id*patternLength + t] = calcGE(teacher[id][t],l);
			(*teacherGI)[id*patternLength + t] = calcGI(teacher[id][t],l);
		}
	}

	free(pattern);
	return patternLength;
}


double rampUp(double x)
{
	return (20.0/(1.0 + exp(1*(-0.7-x))))+EL;
}

double rampDown(double x)
{
	return (20.0/(1.0+ exp(1*(-0.7+x))))+EL;
}

double calcGE(double Um, double l)
{
	return ((GL+GD)*l*(EI-Um))/((1-l)*EI);
}

double calcGI(double Um, double l)
{
	return l*(double(GD)+double(GL))*Um/((1-l)*int(EI));
}

int writeNetStruct(Neuron *N)
{
	for(int i = 0; i < Vis+Hid; i++)
	{
		fwrite(N[i].dend, sizeof(int)*(Vis+Hid), 1, NetStructureF);	
		fwrite(N[i].w, sizeof(N[i].w), 1, NetStructureF);	
		fwrite(&N[i].nGE, sizeof(int), 1, NetStructureF);
		fwrite(N[i].gE, sizeof(int)*5, 1, NetStructureF);	
		fwrite(&N[i].nGI, sizeof(int), 1, NetStructureF);
		fwrite(N[i].gI, sizeof(int)*5, 1, NetStructureF);	
	}

	return 0;
}


