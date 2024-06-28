#!/bin/bash
export PATH=/home/cc/.local/bin:$PATH
export LD_LIBRARY_PATH=/home/cc/.local/lib:$LD_LIBRARY_PATH
export OMPI_CC=hipcc
env MPICC=mpicc python -m pip install mpi4py