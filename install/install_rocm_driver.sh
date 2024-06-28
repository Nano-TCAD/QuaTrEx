#!/bin/bash
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/5.7.1/ubuntu jammy main' \
    | sudo tee /etc/apt/sources.list.d/amdgpu.list
sudo apt update
sudo apt -y install amdgpu-dkms
sudo mv /etc/default/grub /etc/default/grub.old
sudo cp grub /etc/default/grub
sudo update-grub
sudo reboot