"""
Utils for gpu
"""
import subprocess


def gpu_avail(rank=None) -> bool:
    """Check if a gpu is available

    Returns:
        bool: True if gpu available, False otherwise
    """
    avail = False
    try:
        # Check for availability of NVIDIA GPU with nvidia-smi
        _ = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")
        avail = True
        if rank:
            if rank == 0:
                print("nvidia-smi found, NVIDIA GPU available", flush=True)
        else:
            print("nvidia-smi found, NVIDIA GPU available", flush=True)
    except FileNotFoundError:
        avail = False
    if not avail:
        try:
            # Check for availability of AMD GPU with rocm-smi
            _ = subprocess.run(["rocm-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")
            avail = True
            if rank:
                if rank == 0:
                    print("rocm-smi found, AMD GPU available", flush=True)
            else:
                print("rocm-smi found, AMD GPU available", flush=True)
        except FileNotFoundError:
            avail = False
    if not avail:
        print("Neither nvidia-smi, nor rocm-smi found, no GPU available")
    return avail
