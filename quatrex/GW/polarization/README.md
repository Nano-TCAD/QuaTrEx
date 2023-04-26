# Semester Project: Design and Implementation of Scalable Parallel Algorithms for NEGF GW Calculations
## Authors
- Dr. Alexandros Nikolaos Ziogas, Postdoctoral Researcher at ETH Zurich
- Leonard Deutschle, PHD student in Electrical Engineering and Information Technology at ETH Zurich
- Ouyang Runsheng, Master's Student in Quantum Engineering at ETH Zurich
- Alexander Maeder, Master's Student in Electrical Engineering and Information Technology at ETH Zurich

## Description 
The goal of this thesis is to design and implement a scalable code in python for the calculation of the polarization functions from the green's functions. 
The polarization functions is calculated in the following way:

```math
P^{\lessgtr}_{ij}\left(E^{\prime}\right) = -i\frac{dE}{2 \pi} \sum \limits_{E} G^{\lessgtr}_{ij}\left(E\right) G^{\gtrless}_{ji}\left(E-E^{\prime}\right)
```

```math
P^{r}_{ij}\left(E^{\prime}\right) = -i\frac{dE}{2 \pi} \sum \limits_{E} G^{<}_{ij}\left(E\right) G^{a}_{ji}\left(E-E^{\prime}\right)+ G^{r}_{ij}\left(E\right) G^{<}_{ji}\left(E-E^{\prime}\right)
```

with

```math
P^{>}_{ij}\left(E\right) = -P^{<}_{ij}\left(-E\right)^{*}
```

```math
P^{>}_{ij}\left(E\right) - P^{>}_{ij}\left(E\right) = P^{r}_{ij}\left(E\right) - P^{a}_{ij}\left(E\right)
```

```math
P^{a}_{ij}\left(E\right) = P^{r}_{ji}\left(-E\right)^{*}
```

## Packages Required
The following packages are required:
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [cupy](https://cupy.dev/)
- [numba](https://numba.pydata.org/)
- [dace](https://github.com/spcl/dace)
- [rocket-fft](https://github.com/styfenschaer/rocket-fft)
- [mpi4py](https://github.com/mpi4py/mpi4py/)
- [h5py](https://github.com/h5py/h5py/m)
- [matplotlib](https://matplotlib.org/)

Where only rocket-fft is a less standard package. It enables the use of scipy.fft/numpy.fft inside numba though using the PocketFFT implementation. It limits the use usable python version.

## Conda environment
Conda is used for managing the python and package versions and such conda has to be installed first.
[Conda installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) explains the conda installation.
Cuda and mpich have to be added to the path for mpi4py and cupy installation. In addition, dace needs a c-compiler like gcc and cmake. 
They all have version dependencies:
- [cupy cuda version](https://docs.cupy.dev/en/stable/install.html)
- [mpi4py python version](https://mpi4py.readthedocs.io/en/stable/install.html)
- [dace compiler and cmake version](https://spcldace.readthedocs.io/en/latest/setup/installation.html)
Note: dace did not work with clang.

Add the following lines to your .tcshrc file (Adapt to your installations):
```bash
setenv PATH /usr/ela/local/linux-local/cuda-11.8/bin:$PATH 
setenv PATH /usr/ela/local/linux-local/mpich-3.4.2/gcc/bin:$PATH 
setenv PATH /usr/pack/gcc-9.2.0-af/linux-x64/bin/:$PATH
```
(Replace setenv with export if .bashrc is used)


Afterwards, he following command has to be run to install the environment:
```bash
conda env create --name environment_name -f environment.yml
```

The environment file should be updated if the installed packages are changed.
```bash
conda env export > environment.yml
```

## Folder Structure
```bash
├── attelas.run
├── dense
│   ├── CONST.py
│   ├── g2p_dense.py
├── docs
│   ├── bench_cpu_parts.py
│   ├── bench_cpu_strong.py
│   ├── bench_cpu_weak.py
│   ├── bench_gpu.py
│   ├── plot_cpu_parts.py
│   ├── plot_cpu_strong.py
│   ├── plot_cpu_weak.py
│   ├── plot_gpu.py
├── environment.yml
├── initialization
│   ├── gf_init.py
├── mf.run
├── README.md
├── sparse
│   ├── g2p_sparse.py
│   ├── helper.py
├── test_dace.py
├── test_gold.py
├── test_mpi.py
└── tmp.py
```
The following folder:
- `dense/`: The dense implementation
- `sparse/`: The sparse implementation
- `docs/`: Benchmarking, plotting scripts and additional documentation in form of presentations
- `initialization/`: The script to random initialize the Green's Functions
The following scripts:
- `attelas.run`: Mpi rank pinning for the attelas cluster
- `environment.yml`: Conda environment file to create a environment with the right packages
- `mf.run`: Mpi rank pinning for the mont-fort cluster
- `test_dace.py`: Script to test the dace implementation
- `test_gold.py`: Script to test the different single node implementations
- `test_mpi.py`: Script to test the mpi/cuda implementation


## Testing 
The resulting code can be tested against a gold solution coming from a MATLAB implementation. In addition, the dense python implementation of Ouyang Runsheng can be also tested against the gold solution, but the higher RAM capacity and patience is needed.

The gold solution has to be provided by the user in the form of a .mat file. 
The gold solution with the name `data_11_PandG.mat` has to be put into the `gold_solution/` folder.  todo change way to provide.
Then the solution has to be first processed with `gold_solution/changeFormat.m`.
The matlab file adjusts the sparse matrix for every energy point to contain non zero elements at the same places.
This is done through filling missing elements with a close to vanishing number of order $10^{-300}$.
The reason to do this, is for uniformity to ease the parallelization. The overhead from a few more non zero elements is vanishing.
In addition, the format is changed from the number of energy points sparse matrices to saving once the row/column indexes (could/should be changed from coo to csr format) and a large 2D array of size number of non zero elements times the number of energy points.

The test against the gold solution can be done for cpu, gpu and multi gpu code.
```bash
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft_inlined
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_conv
python test_gold.py -t gpu_fft
python test_gold.py -t gpu_conv
NUMBA_NUM_THREADS=1 python test_gold.py -t dense
mpiexec -n 4 -f attelas.run python test_mpi.py
```
where `NUMBA_NUM_THREADS` defines the amount of threads used for cpu kernels. 
If NUMBA_NUM_THREADS variable is not specified, numba will take the max number of threads possible.
The two arguments describe which code should be tested.
With the `-n` flag of mpiexec the number of ranks is specified. In addition, with `-f` flag configuration on how to pin the ranks can be given

The different solutions are the same if the printed differences are of the order 1e-16 until 1e-12 and the assert do not fail.

In addition, the `-f` flag is possible to change the solution path to test against.
```bash
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft -f path-to-solution.mat
```

### DaCe Settings
The compiler to be used with dace can be set [manually](https://spcldace.readthedocs.io/en/latest/setup/config.html). 
It should be noted that in the file .dacecache/dace.conf the used compiler flags can be checked.
```bash
bash
export DACE_compiler_cpu_executable=/usr/pack/gcc-9.2.0-af/linux-x64/bin/gcc
```

## Timing
There are benchmarking files in the `docs/` folder and corresponding plotting scripts. 
The enable to create strong and weak scaling plots for the different implementations. 
Please check the code to see which inputs are possible.
As an example for a strong scaling plot for the fft implementation with artificial data:
```bash
python -O docs/bench_cpu_strong.py fft art
python docs/plot_cpu_strong.py fft
```
In addition, there is the `docs/bench_cpu_parts.py` which benchmarks the different parts of 
the fft implementation. This file can be extended if other parts should be benchmarked.

The -OO flag would remove doc strings and asserts.
Asserts should be removed when random data is used.

The weak scaling benchmarking files are the examples on how to use the random generated green's functions inputs.
These inputs are not fully random, but fullfil the physical identities stated above.
The functions for generating the random inputs are inside `initialization/gf_init.py`.

## Profiling
First the cuda toolkit has to be added to the path. This can be done through adding the following lines to your .tcshrc file:
```
setenv PATH /usr/ela/local/linux-local/cuda-11.8/bin:$PATH
setenv PATH /usr/ela/local/linux-local/cuda-11.8:$PATH
```
The meaning of the above is to add the cuda installation folder to your path. 

Then a nvidia nsight report can be created in the following way:
```bash
NUMBA_NUM_THREADS=1 nsys profile --output=name_report python test_gold.py gpu
NUMBA_NUM_THREADS=1 nsys profile --output=name_report python -OO timing.py gpu
```
This report can be analyzed on your local machine with [NVIDIA nsight systems](https://developer.nvidia.com/nsight-systems/get-started).

## Open Problems
- For input data the following identity does not hold: (Reason: Through noise it is not always zero.)
```math
G^{>}_{ij}\left(E\right) - G^{<}_{ij}\left(E\right) = G^{r}_{ij}\left(E\right) - G^{a}_{ij}\left(E\right)
```
Better to test:
```math
max(G^{>}\left(E\right) - G^{<}\left(E\right) - (G^{r}\left(E\right) - G^{a}\left(E\right)))/max(G^{r}\left(E\right) - G^{a}\left(E\right))
```
```math
max(G^{>}\left(E\right) - G^{<}\left(E\right) - (G^{r}\left(E\right) - G^{a}\left(E\right)))/max(G^{>}\left(E\right) - G^{<}\left(E\right))
```
Both should be close to zero

- GPU conv implementation is dog
- DaCe implementation
- After DaCe implementation, combine `test_dace.py` with `test_gold.py`.
- Connecting the different parts of the solver.
- Make the README.md less garbo.
- Investigate why the cpu fft scaling is suboptimal.
