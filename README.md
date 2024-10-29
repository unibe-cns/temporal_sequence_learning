
**Work in progress, code will be fully uploaded in the coming days**

This code contains the simulation code for the paper *"ELiSe: Efficient Learning of Sequences in Structured Recurrent Networks"* ([arxiv](https://arxiv.org/abs/2402.16763)) by Kristin Völk, Laura Kriener, Ben von Hünerbein, Federico Benitez, Walter Senn and Mihai A. Petrovici.

This code base was originally developed by Kristin Völk and then extended and adapted by the other authors.

### Code base

The code base is split in two parts for simulation and evaluation.

#### Simulation
The simulation code is written in C and CUDA and is located in the `src/` subdirectory.
It requires CUDA, gcc, the GNU Scientific Library (GSL) and make. 

The code has been tested on Ubuntu 22.04 with CUDA 11.8, gcc 11, GSL 2.7 and GNU make 4.3.
Other version should work as well, note however that CUDA does not support all
versions of gcc. If non-matching versions are used, the nvcc compiler will throw
an error.

#### Evaluation
The evaluation code is written in Python and is located in the `evaluation/` subdirectory.
The code was tested with `Python 3.12.4` but should also work with other
versions.
All necessary packages are detailed in the `requirements.txt` and installed with
```
pip install -r requirements.txt
```

The packages have been pinned to specific versions for stability, but others
should also work.

### Usage

**Compiling and configuring the experiment**

Each experiment is configured using a makefile which specifies
- experiment parameters (e.g. network size, pattern, disruptions, ...)
- location where simulation data is saved
- what variables are recorded

and compiles a binary of the code with this parametrization.
Additionally, the makefile copies itself, the default parameterization as well as the used target pattern file into the location that the code will save the simulation data.
This is done for reproducibility, such that for every set of produced simulation recordings the corresponding parameterization is stored alongside it.
On a standard desktop pc this should complete within a few seconds.

Example:
```
mkdir bin
mkdir experiment_data
make -f makefile_single_run SingleRun
```

**Running the experiment**

After successful compilation the experiment can then be run with e.g.
```
bin/single_run
```

This should run within 10-30min on a desktop pc with a gpu, depending on the
hardware available. (Note that if the full network is recorded, due to increased file-IO this might slow down
significantly).

**Evaluating the experiment**

The experiment results are saved in the (timestamped) directory specified in the
makefile.
To evaluate and visualize the results there are several python scripts in the
`evaluation/` subdirectory.
They are used as follows:

```
cd evaluation
python plotRateVizualizations.py <path_to_simulation_results>
python plotMetrics.py <path_to_simulation_results>
python plotNetwork.py <path_to_simulation_results>
```

The files will produce visualization plots on the performance metrics, the
recorded firing rates as well as the pre- and post-training weights.
They are saved alongside the data.
Additionally, the code saves e.g. the performace data for further analysis.
On a typical desktop pc this should run in less than a minute.

### Model/experiment workflow

### Paper results

Most results in the paper are aggregates of several simulation runs (e.g.
different sizes of the latent population and multiple seeds).
To run these sweeps we loop over multiple calls of the sweeps makefile (which
taks the sweeped parameter and the seed as a commandline argument) and run all
generated binaries afterwards.

The bash scripts and makefiles for the data in the paper can be found in
`sweeps/`and are listed below:

- Fig 2: `makefile_sweepelement_size`, `run_sweep_size.sh`
- Fig 3: `makefile_sweepelement_reservoir`, `run_sweep_reservoir.sh`
- Fig 4: `makefile_sweepelement_pq`, `makefile_sweepelement_sparse`, `run_sweep_pq.sh`, `run_sweep_sparse.sh`
- Fig 5: `makefile_sweepelement_disruption`, `run_sweep_disruption.sh`

To reproduce all data in Fig 4 the sweep needs to be run once for the parameter
`PROTECT` set to 1 and once for 0.

To reproduce all data in Fig 5 (i.e. the different disruptions) the sweep needs
to be repeated for all available disruption files (parameter in the makefile: `DISRUPTION_FILE`).
