#!/bin/bash
git clone https://github.com/openucx/ucx.git ucx
cd ucx
git submodule update --init --recursive
./autogen.sh
mkdir build
cd build
../configure --prefix=/home/cc/.local --with-rocm=/opt/rocm --without-knem
make -j
make install
cd ../../
git clone https://github.com/open-mpi/ompi.git
cd ompi
git submodule update --init --recursive
./autogen.pl
mkdir build
cd build
../configure -prefix=/home/cc/.local --with-rocm=/opt/rocm --with-ucx=/home/cc/.local
make -j
make install
cd ../../
export PATH=/home/cc/.local/bin:$PATH
export LD_LIBRARY_PATH=/home/cc/.local/lib:$LD_LIBRARY_PATH
export OMPI_CC=hipcc