# Bechmarking & Plotting
Contains scripts to benchmark the different GW implementations. 
Either strong or weak scaling plots can be created.
The amount of numba threads is varied.

# Examples
## Strong Scaling
```bash
python bench_cpu_strong.py -t p_fft_cpu -ne 200 -nnz 500 -th 10 -r 20
python plot_cpu_strong.py -t p_fft_cpu 
python bench_cpu_strong.py -t s_fft_cpu -ne 200 -nnz 500 -th 10 -r 20
python plot_cpu_strong.py -t s_fft_cpu 
```
The `-t` flag specifies the kernel type. The number of energy points and the non-zero elements can be decided with the `-ne` and `-nnz` flags.
The `th` flag specifies the number of threads to use.
The `-r`flag specifies the number of runs per data point.
The other input options can be found with:
```bash
python bench_cpu_strong.py --help
python plot_cpu_strong.py --help
```

## Weak Scaling
```bash
python bench_cpu_weak.py -t s_cpu_fft -d nnz -ne 200 -nnz 2000 -th 10 -r 20
python plot_cpu_weak.py -t s_fft_cpu -d nnz
```
The `-t` flag specifies the kernel type. 
The initial number of energy points and the non-zero elements can be decided with the `-ne` and `-nnz` flags.
The `th` flag specifies the number of threads to use.
The `-r`flag specifies the number of runs per data point.
The `-d` flag specifies over which dimension of the 2D array to scale.
This makes a difference since the run time does scale linearly in nnz, but nonlinear for ne depending on the kernel.
The other input options can be found with:
```bash
python bench_cpu_weak.py --help
python plot_cpu_weak.py --help
```

## GPU Timing
```bash
python bench_gpu.py -t gpu_fft -d energy -r 20 -ne 400 -nnz 2500 -m 20
python plot_gpu.py -t s_fft_cpu -d nnz
```
This would time a GPU kernel with linear scaling 2D matrix size.
The `-t` flag specifies the kernel type. 
The initial number of energy points and the non-zero elements can be decided with the `-ne` and `-nnz` flags.
The `-r`flag specifies the number of runs per data point.
The `-d` flag specifies over which dimension of the 2D array to scale.
In contrast to the weak scaling plot, here the scaling is always linear.
The `-m` flag specifies how much the initial size should be scaled
The other input options can be found with:
```bash
python bench_gpu.py --help
python plot_gpu.py --help
```

## Part Strong Scaling
Creates a strong scaling plot for different parts of the `cpu_fft` implementation.
Would have to be manually extended to benchmark other parts.
```bash
python bench_cpu_parts.py -ne 200 -nnz 400 -th 10 -r 20
python plot_cpu_parts.py
```
The initial number of energy points and the non-zero elements can be decided with the `-ne` and `-nnz` flags.
The `th` flag specifies the number of threads to use.
The `-r`flag specifies the number of runs per data point.

The other input options can be found with:
```bash
python bench_cpu_parts.py --help
python plot_cpu_parts.py --help
```

## MPI Benchmark
Runs an MPI script with a varying number of ranks.
```bash
python bench_mpi.py -pp pin_path.run -sp test_mpi.py -r 4
```
The `-pp` flag specifies the path to the rank pinning file.
The `-sp` flag specifies the path to the python mpi4py script to run.
The `-r` flag specifies the max number of ranks to use.

The measured times are with all the possible overhead included.

The other input options can be found with:
```bash
python test_mpi.py --help
```

# Open Problems
- Add support for P->W and H->G as both need MKL threads.
- Add support for DaCe kernels as they use OMP threads.
- Extend with newly created kernels
- Improve the MPI benchmark
