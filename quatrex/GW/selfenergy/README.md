# Description
In this folder, the calculation of the GW self-energies are implemented.
The formula for the self-energies and the derivation for the used transformations can be found in `selfenergy/kernel/README.md`. 

## Packages Required
The following packages are required:
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [cupy](https://cupy.dev/)
- [numba](https://numba.pydata.org/)
- [rocket-fft](https://github.com/styfenschaer/rocket-fft)
- [h5py](https://github.com/h5py/h5py/m)

Where only rocket-fft is a less standard package. It enables the use of scipy.fft/numpy.fft inside numba though using the PocketFFT implementation. It limits the use usable python version.

## Folder Structure
```bash
.
├── kernel
│   ├── gw2s_cpu.py
│   ├── gw2s_gpu.py
│   └── README.md
├── README.md
└── test_gold.py
```

## Testing 
```bash
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft
NUMBA_NUM_THREADS=1 python test_gold.py -t gpu_fft
```
With `NUMBA_NUM_THREADS` the amount of used threads can be controlled.
This does not have an impact on the gpu implementation.

In addition, with the `-f` flag it is possible to change the solution path to test against.
```bash
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft -f path-to-solution.mat
```

## Open Problems
- Possibility to reduce amount of FFT since both self-energy and polarization need the FFT of the green's function.
- Think about using some identities to reduce the amount of IFFT
- Dace implementation would be nice
- Better documentation is a plus
- Make an inlined cpu fft implementation (Numba does not support repeat with the axis argument)