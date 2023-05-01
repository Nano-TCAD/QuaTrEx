"""
Utils for gpu
"""
import subprocess

def gpu_avail() -> bool:
    """Check if a gpu is available

    Returns:
        bool: True if gpu available, False otherwise
    """
    avail = False
    try:
        # throug running nvidia-smi check if a gpu available
        _ = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")
        avail = True
    except FileNotFoundError:
        avail = False
    return avail
