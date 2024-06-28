#!/bin/bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh
source /home/cc/.bashrc
conda install -y numpy==1.26.4 scipy=1.11.4 matplotlib seaborn