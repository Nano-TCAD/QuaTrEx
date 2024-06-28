#!/bin/bash
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm
export HCC_AMDGPU_TARGET=gfx908
export CFLAGS="-I /opt/rocm/include"
python -m pip install cupy==12.3.0