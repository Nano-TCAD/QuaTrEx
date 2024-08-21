from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

arr_size = 15

if rank == 0:
    sendbuf = np.arange(2*arr_size).reshape(arr_size, 2)
else:
    sendbuf = None

# count: the size of each sub-task
ave, res = divmod(arr_size, nprocs)
count = [ave + 1 if p < res else ave for p in range(nprocs)]
count = np.array(count)

# displacement: the starting index of each sub-task
displ = [sum(count[:p]) for p in range(nprocs)]
displ = np.array(displ)

# initialize recvbuf on all processes
recvbuf = np.zeros((count[rank], 2))

comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=0)

print('After Scatterv, process {} has data:'.format(rank), recvbuf)