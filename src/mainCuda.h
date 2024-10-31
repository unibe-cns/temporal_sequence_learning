
#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>
#include <math.h>

#ifndef TRAININGCYCLES
	#define TRAININGCYCLES 100
#endif

#ifndef REPLAYCYCLES
	#define REPLAYCYCLES 10
#endif

#ifndef Vis
	#define Vis 15
#endif

#ifndef Hid
	#define Hid 50
#endif

#ifndef PlotFreq
	#define PlotFreq 50
#endif

#ifndef PlotRU
	#define PlotRU 1
#endif

#ifndef PlotUM
	#define PlotUM 1
#endif

/********************************************************
 * MODE: soma only
 *       with dendrite 
 ********************************************************/
#ifndef MODE
	#define MODE "with dendrite"
#endif

/********************************************************
 * DISRUPTION: 0 - no disruption during replay
 *             1 - disruption loaded from file
 ********************************************************/
#ifndef DISRUPTION
	#define DISRUPTION 0
#endif

#ifndef FILENAME_DISRUPTION
	#define FILENAME_DISRUPTION "None"
#endif

/********************************************************
 * CONNECT: all-to-all  - Standard network
 *          innervation ratio - Only a part of the connections from vis to
 *          sparse - som-dend conns only formed with ConnDensity probalility
 ********************************************************/
#ifndef CONNECT
	#define CONNECT "all-to-all"
#endif

#ifndef InnervRatio
	#define InnervRatio 1.0
#endif

#ifndef ConnDensity
	#define ConnDensity 1.0
#endif

#ifndef PROTECT_SCAFFOLD
	#define PROTECT_SCAFFOLD 0
#endif
/********************************************************
 * TRAIN: auto
 *        static
 ********************************************************/
#ifndef TRAIN
	#define TRAIN "static"
#endif

#ifndef P1
	#define P1 0.2
#endif
#ifndef Q1
	#define Q1 0.05
#endif
#ifndef P0
	#define P0 0.04
#endif

#ifndef Iter
	#define Iter 1
#endif

#ifndef WriteNudge
	#define WriteNudge 1
#endif

#ifndef W_INIT_VAR_V_TO_V
	#define W_INIT_VAR_V_TO_V 0.5
#endif

#ifndef W_INIT_VAR_V_TO_H
	#define W_INIT_VAR_V_TO_H 0.5
#endif

#ifndef W_INIT_VAR_H_TO_V
	#define W_INIT_VAR_H_TO_V 0.5
#endif

#ifndef W_INIT_VAR_H_TO_H
	#define W_INIT_VAR_H_TO_H 0.5
#endif

#ifndef W_INIT_MEAN_V_TO_V
	#define W_INIT_MEAN_V_TO_V 0.0
#endif

#ifndef W_INIT_MEAN_V_TO_H
	#define W_INIT_MEAN_V_TO_H 0.0
#endif

#ifndef W_INIT_MEAN_H_TO_V
	#define W_INIT_MEAN_H_TO_V 0.0
#endif

#ifndef W_INIT_MEAN_H_TO_H
	#define W_INIT_MEAN_H_TO_H 0.0
#endif

#define DT 0.1
#define EL -70
#define EI -75
#define GI0 6
#define GE0 0.3
#define GD 2.0
#define GL .1
#define TauE 15
#define TauI 35
#define BI 350
#define BE 100
#define DLY 100

#define DLY_MAX 150
#define GE_DLY 150
#define GI_DLY 400 

#define GL_DT GL*DT
#define GL_DT_EL GL*DT*EL
#define GD_DT GD*DT
#define EI_DT EI*DT
#define GL_EL GL*EL
#define GD_GL GD+GL
#define BETA DT*GL*(GD/(GD+GL))
#define BETA2 DT*(GD/(GD+GL))
#define DT_TAUE DT/TauE
#define TAUI_DT TauI/DT
#define PHI_EL  (1.0 / (1.0 + exp(0.3 * (-58 - EL))))

typedef struct Neuron
{
	double U;
	double rU;
	double Um;
	double V;
	double Vnew;
	double Vstar;
	double rV;
	double dVdw;

	int dend[Vis+Hid];
	double w[Vis+Hid];
	int gE[5];
	int gI[5];
	double gEout;
	double gIout;
	int riseFlag;
	
	double gIBuffer[GI_DLY];
	double gEBuffer[GE_DLY];
	double rUBuffer[DLY_MAX];

	int gITimer;
	int gETimer;

	int gIDly;
	int gEDly;
	int dendDly;
	int gEwrite;
	int gIwrite;
	int gEread;
	int gIread;
	int rUwrite;
	int rUread;
	int nGE;
	int nGI;
	int tag;

	double rand;

} Neuron;

void initDisruption(int *t_starts, int *t_ends, int *nrn_starts, int *nrn_ends, int max_disrupt_lines, float *disrupt_offset_el);
int initTeacherCounting(double **teacherGE, double **teacherGI);
int initTeacherGrammer(double *teacherGE, double *teacherGI);
int initTeacherFix(double* teacherGE, double* teacherGI);
void confN2(Neuron *N);
void setupDendriticConnections(Neuron *N);

double rampUp(double x);
double rampDown(double x);
double calcGE(double Um, double l);
double calcGI(double Um, double l);

int writeNetStruct(Neuron *N);
#endif


