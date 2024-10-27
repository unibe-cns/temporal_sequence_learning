
**Work in progress, code will be fully uploaded in the coming days**

This code contains the simulation code for the paper *"ELiSe: Efficient Learning of Sequences in Structured Recurrent Networks"* ([arxiv](https://arxiv.org/abs/2402.16763)) by Kristin Völk, Laura Kriener, Ben von Hünerbein, Federico Benitez, Walter Senn and Mihai A. Petrovici.

This code base was originally developed by Kristin Völk and then extended and adapted by the other authors.

### Code base

The code base is split in two parts for simulation and evaluation.

#### Simulation
The simulation code is written in C and CUDA and is located in the `src/` subdirectory.
It requires CUDA, the GNU Scientific Library (GSL) and make. 

#### Evaluation
The evaluation code is written in Python and is located in the `evaluation/` subdirectory.
All necessary packages are detailed in the `requirements.txt`.

### Usage

**Compiling and configuring the experiment**

Each experiment is configured using a makefile which specifies
- experiment parameters (e.g. network size, pattern, disruptions, ...)
- location where simulation data is saved
- what variables are recorded

and compiles a binary of the code with this parametrization.
Additionally, the makefile copies itself, the default parameterization as well as the used target pattern file into the location that the code will save the simulation data.
This is done for reproducibility, such that for every set of produced simulation recordings the corresponding parameterization is stored alongside it.

Example:
```
mkdir bin
make -f makefile_single_run TimeSeries
```

**Running the experiment**

After successful compilation the experiment can then be run with e.g.
```
cd bin
./single_run
```

**Evaluating the experiment**
