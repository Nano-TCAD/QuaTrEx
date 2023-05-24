# Green's Function to Polarization
## Description 
Implements the calculation of the polarization function from the green's function. 
The polarization function is calculated in the following way:

```math
P^{\lessgtr}_{ij}\left(E^{\prime}\right) = -i\frac{dE}{2 \pi} \sum \limits_{E} G^{\lessgtr}_{ij}\left(E\right) G^{\gtrless}_{ji}\left(E-E^{\prime}\right)
```

```math
P^{r}_{ij}\left(E^{\prime}\right) = -i\frac{dE}{2 \pi} \sum \limits_{E} G^{<}_{ij}\left(E\right) G^{a}_{ji}\left(E-E^{\prime}\right)+ G^{r}_{ij}\left(E\right)G^{<}_{ji}\left(E-E^{\prime}\right)
```

with the following identities:

```math
P^{>}_{ij}\left(E\right) = -P^{<}_{ij}\left(-E\right)^{*}
```

```math
P^{>}_{ij} \left( E \right) - P^{>}_{ij}\left(E\right) = P^{r}_{ij}\left(E\right) - P^{a}_{ij}\left(E\right)
```

```math
P^{a}_{ij} \left( E \right) = P^{r}_{ji} \left( -E \right)^{*}
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

Where only rocket-fft is a less standard package. It enables the use of scipy.fft/numpy.fft inside numba though using the PocketFFT implementation. It limits the usable Python version.

## Folder Structure
```bash
.
├── initialization
│   └── gf_init.py
├── __init__.py
├── kernel
│   ├── g2p_cpu.py
│   ├── g2p_gpu.py
│   └── README.md
├── README.md
├── test_gold.py
└── test_mpi.py
```
The following folder:
- `kernel/`: CPU and GPU implementations
- `initialization/`: The script to random initialize the Green's Functions
The following scripts:
- `test_gold.py`: Script to test the different single-node implementations
- `test_mpi.py`: Script to test the MPI/CUDA implementation


## Testing 
The resulting code can be tested against a gold solution coming from a MATLAB implementation. In addition, the dense Python implementation of Ouyang Runsheng can be also tested against the gold solution, but higher RAM capacity and patience are needed.

The path to the gold solution has to be provided and its contents should match how the solution is read in `gold_solution/`
Beforehand, the solution has to be processed with `gold_solution/changeFormat.m`.
The Matlab file adjusts the sparse matrix for every energy point to contain non-zero elements at the same places.
This is done by filling missing elements with epsilon $~10^{-300}$.
This is done for uniformity. The overhead from a few more non-zero elements is vanishing.
In addition, the format got changed from a list of sparse matrices to saving once the row/column indexes and a large 2D array of size number of non-zero elements times the number of energy points.

The test against the gold solution can be done for CPU, GPU, and multi-GPU code.
```bash
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft_inlined
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_conv
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_dense
python test_gold.py -t gpu_fft
python test_gold.py -t gpu_conv
mpiexec -n 4 -f pinning.run python test_mpi.py
```
where `NUMBA_NUM_THREADS` defines the number of threads used. 
If the NUMBA_NUM_THREADS variable is not specified, numba will take the max number of threads possible.
With the `-n` flag of mpiexec the number of ranks is specified. In addition, with the `-f` flag a configuration on how the ranks are pinned can be given.

The different solutions are the same if the matrix norm differences are of the order 1e-16 until 1e-12 and the asserts do not fail.

With the `-f` flag is possible to change the solution path to test against.
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
There are benchmarking scripts in the `docs/` folder and corresponding plotting scripts. 
They enable the creation of strong and weak scaling plots for the different implementations. 
Please check the code to see which inputs are possible.
As an example of a strong scaling plot for the FFT implementation with artificial data:
```bash
python docs/bench_cpu_strong.py -t pol_fft_cpu -ne 200 -nnz 500
python docs/plot_cpu_strong.py -t pol_fft_cpu
```
In addition, there is the `docs/bench_cpu_parts.py` script which benchmarks the different parts of 
the FFT implementation. This file can be extended if other parts should be benchmarked.

The -OO flag removes doc strings and asserts.
Asserts should be removed when random data is used.

The weak scaling benchmarking files are examples of how to use the randomly generated Green's functions inputs.
These inputs are not fully random, but fulfill the physical identities stated above.
The functions for generating the random inputs are inside `initialization/gf_init.py`.

## Profiling
First, the Cuda toolkit has to be added to the path. This can be done by adding the following lines to your .tcshrc file:
```
setenv PATH /usr/ela/local/linux-local/cuda-11.8/bin:$PATH
setenv PATH /usr/ela/local/linux-local/cuda-11.8:$PATH
```

Then an Nvidia nsight report can be created in the following way:
```bash
NUMBA_NUM_THREADS=1 nsys profile --output=name_report python test_gold.py gpu
NUMBA_NUM_THREADS=1 nsys profile --output=name_report python -OO timing.py gpu
```
This report can be analyzed on your local machine with [NVIDIA nsight systems](https://developer.nvidia.com/nsight-systems/get-started).

## Open Problems
- For input data, the following identity does not hold. This is due to noise. 
$$G^{>}_{ij}\left(E\right) - G^{<}_{ij}\left(E\right) = G^{r}_{ij}\left(E\right) - G^{a}_{ij}\left(E\right)$$

Better to test:
$$max(G^{>}\left(E\right) - G^{<}\left(E\right) - (G^{r}\left(E\right) - G^{a}\left(E\right)))/max(G^{r}\left(E\right) - G^{a}\left(E\right))$$

$$max(G^{>}\left(E\right) - G^{<}\left(E\right) - (G^{r}\left(E\right) - G^{a}\left(E\right)))/max(G^{>}\left(E\right) - G^{<}\left(E\right))$$

Both should be close to zero

- GPU conv implementation is not performing
- DaCe conv implementation is not performing
- FFT DaCe implementation
- Investigate why the CPU FFT scaling is suboptimal.

