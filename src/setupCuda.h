
#ifndef SETUP_H
#define SETUP_H

#include "mainCuda.h"

void initNetwork(Neuron *N);
void nudgeNeuron(int idT, int idL, Neuron *N);
double phi(double u);
double calcError(Neuron *N);


#endif
