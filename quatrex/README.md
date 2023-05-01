## Authors
- Dr. Alexandros Nikolaos Ziogas, Postdoctoral Researcher at ETH Zurich
- Leonard Deutschle, PHD student in Electrical Engineering and Information Technology at ETH Zurich
- Ouyang Runsheng, Master's Student in Quantum Engineering at ETH Zurich
- Alexander Maeder, Master's Student in Electrical Engineering and Information Technology at ETH Zurich

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