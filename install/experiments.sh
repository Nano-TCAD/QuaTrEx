#!/bin/bash
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
mpirun -n 1 --mca pml ucx python gw_sc24_test.py -f small -i 10 -e 32 -w 32 -r 1 -o small_e32_n01.csv
mpirun -n 1 --mca pml ucx python gw_sc24_test.py -f large -i 10 -e 16 -w 4 -r 1 -o large_e16_n01.csv
