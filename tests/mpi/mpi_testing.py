from mpi4py import MPI
import numpy as np
import numpy.typing as npt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data_shape = np.array([4, 2], dtype=np.int32)

# split nnz/energy per rank
data_per_rank = data_shape // size

# create array with energy size distribution
count = np.repeat(data_per_rank.reshape(-1, 1), size, axis=1)
count[:, size-1] += data_shape % size
print(f"count: {count}")

# displacements in nnz/energy
disp = data_per_rank.reshape(-1, 1) * np.arange(size)
print(f"disp: {disp}")

BASE_TYPE = MPI.Datatype(MPI.DOUBLE_COMPLEX)
base_size = np.dtype(np.complex128).itemsize

# column type of orginal matrix
COLUMN = BASE_TYPE.Create_vector(data_shape[0], 1, data_shape[1])
COLUMN_RIZ = COLUMN.Create_resized(0, base_size)
MPI.Datatype.Commit(COLUMN_RIZ)
MPI.Datatype.Commit(COLUMN)

# row type of original transposed matrix
ROW = BASE_TYPE.Create_vector(data_shape[1], 1, data_shape[0])
ROW_RIZ = ROW.Create_resized(0, base_size)
MPI.Datatype.Commit(ROW_RIZ)
MPI.Datatype.Commit(ROW)

def gather_master_checkpoint(inp: npt.NDArray[np.complex128], outp: npt.NDArray[np.complex128]):
    comm.Gatherv(inp, [outp, count[0, :], disp[0, :], ROW_RIZ], root=0)


def scatter_master(inp: npt.NDArray[np.complex128], outp: npt.NDArray[np.complex128]):
    comm.Scatterv([inp, count[0, :], disp[0, :], ROW_RIZ], outp, root=0)


gg_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
print(f"gg_g2p.shape: {gg_g2p.shape}")

if rank == 0:
    gg_full = np.arange(data_shape[0] * data_shape[1], dtype=np.complex128).reshape(data_shape)
    print(f"gg_full.shape: {gg_full.shape}")
else:
    gg_full = None

scatter_master(gg_full, gg_g2p)
print(f"rank: {rank}, gg_g2p: {gg_g2p}")

if rank == 0:
    gg_save = np.empty(data_shape, dtype=np.complex128, order="C")
else:
    gg_save = None
gather_master_checkpoint(gg_g2p, gg_save)
