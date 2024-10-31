
#ifndef SIMCUDA_H
#define SIMCUDA_H

#include "mainCuda.h"

void updateMemPotWrapper(Neuron *N, double *d_teacherGE, double *d_teacherGI, int TB, int lambda, int t, int c);
void updateMemPotReplayWrapper(Neuron *N, double *d_teacherGE, double *d_teacherGI, int TB, int lambda, int t, int c, int *visDisrupted, float disrupt_offset);

#endif
