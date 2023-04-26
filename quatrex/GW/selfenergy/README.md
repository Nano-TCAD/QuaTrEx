## Authors
- Dr. Alexandros Nikolaos Ziogas, Postdoctoral Researcher at ETH Zurich
- Leonard Deutschle, PHD student in Electrical Engineering and Information Technology at ETH Zurich
- Alexander Maeder, Master's Student in Electrical Engineering and Information Technology at ETH Zurich

## Description
In this folder, the calculation of the GW self-energies are implemented.
The formula for the self-energies and the derivation for the used transformations can be found in `docs/derivation_selfenergy.pdf`. 

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
├── docs
│   └── derivation_selfenergy.pdf
├── README.md
├── sparse
│   └── gw2s_sparse.py
└── test_gold.py
```

As mentioned `docs/` contain the derivation of the used formula.
Inside `sparse` is just the implementation

## Testing 
```bash
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft
NUMBA_NUM_THREADS=1 python test_gold.py -t gpu_fft
```
With `NUMBA_NUM_THREADS` the amount of used threads can be controlled.
This does not have an impact on the gpu implementation.

In addition, the `-f` flag is possible to change the solution path to test against.
```bash
NUMBA_NUM_THREADS=1 python test_gold.py -t cpu_fft -f path-to-solution.mat
```

## Open Problems
- Possibility to reduce amount of fft since both self-energy and polarization need the fft of the green's function.
- Dace implementation would be nice
- Better documentation is a plus
- Maybe add benchmarking for this new but similar functions (look at polarization)
- Make an inlined cpu fft implementation (Numba does not support repeat with the axis argument)