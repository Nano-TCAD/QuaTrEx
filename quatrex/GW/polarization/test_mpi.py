
"""
Implemented multi-node GPU polarization computations.
Includes all MPI communication needed for the whole chain.
Assumes every node has a GPU connected.
"""
import mpi4py
import sys
import numpy as np
import numpy.typing as npt
import os
import argparse

mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False    # do not finalize MPI automatically
from mpi4py import MPI

# ghetto solution from ghetto coder
main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", ".."))
sys.path.append(parent_path)

from GW.polarization.kernel import g2p_gpu
from GW.gold_solution import read_solution
from utils import change_format

if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # number of energy points and nnz elements
    data_shape: np.ndarray = np.array([0, 0])

    # energy step
    pre_factor = np.array([0.0], dtype=np.complex128)

    # load data on master-------------------------------------------------------
    if rank == 0:
        # read gold solution
        solution_path = os.path.join("/scratch/quatrex_data", "data_GPWS_old.mat")

        parser = argparse.ArgumentParser(
            description="Tests the mpi implementation of the polarization calculation"
        )
        parser.add_argument("-f", "--file", default=solution_path, required=False)

        args = parser.parse_args()

        # load greens function
        energy, rows, columns, gg_gold, gl_gold, gr_gold    = read_solution.load_x(args.file, "g")
        # load polarization
        _, _, _, pg_gold, pl_gold, pr_gold                  = read_solution.load_x(args.file, "p")

        ij2ji:      npt.NDArray[np.int32]   = change_format.find_idx_transposed(rows, columns)
        denergy:    npt.NDArray[np.double]  = np.array([energy[1] - energy[0]], dtype=np.double)
        ne:         np.int32                = np.int32(energy.shape[0])
        no:         np.int32                = np.int32(columns.shape[0])
        pre_factor: np.complex128           = -1.0j * denergy / (np.pi)

        data_shape = np.array(gg_gold.shape)

        # print size of data
        print(f"#Energy: {data_shape[1]} #nnz: {data_shape[0]}")


    # broadcast needed quantities-----------------------------------------------

    # exchange size of array(nnz,#energy)
    comm.Bcast(data_shape, root=0)

    # transposing vector buffer
    if rank != 0:
        ij2ji = np.empty(data_shape[0], dtype=np.int32)



    # broadcast energy step
    comm.Bcast(pre_factor, root=0)

    # broadcast transposing vector
    comm.Bcast(ij2ji, root=0)

    # calculation of data distribution per rank---------------------------------

    # split nnz/energy per rank
    data_per_rank = data_shape // size

    # create array with energy size distribution
    count = np.repeat(data_per_rank.reshape(-1, 1), size, axis=1)
    count[:, size-1] += data_shape % size

    # displacements in nnz/energy
    disp = data_per_rank.reshape(-1, 1) * np.arange(size)

    # print rank distribution
    print(
    f"Rank: {rank} #Energy/rank: {count[1,rank]} #nnz/rank: {count[0,rank]}", 
    name)

    # create needed data types--------------------------------------------------

    # arrays filled with complex doubles
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

    # send type g2p
    # column type of split up in nnz
    G2P_S = BASE_TYPE.Create_vector(count[1, rank], 1, data_shape[0])
    G2P_S_RIZ = G2P_S.Create_resized(0, base_size)
    MPI.Datatype.Commit(G2P_S)
    MPI.Datatype.Commit(G2P_S_RIZ)

    # receive types g2p
    # vector of size of #ranks
    # multi column data type for every rank size #energy not divisible
    G2P_R = np.array([BASE_TYPE.Create_vector(
        count[0, rank], count[1, i], data_shape[1]) for i in range(size)])
    G2P_R_RIZ = np.empty_like(G2P_R)
    for i in range(size):
        G2P_R_RIZ[i] = G2P_R[i].Create_resized(0, base_size)
        MPI.Datatype.Commit(G2P_R[i])
        MPI.Datatype.Commit(G2P_R_RIZ[i])

    # send type p2g
    # column type of split up in energy
    P2G_S = BASE_TYPE.Create_vector(count[0, rank], 1, data_shape[1])
    P2G_S_RIZ = P2G_S.Create_resized(0, base_size)
    MPI.Datatype.Commit(P2G_S)
    MPI.Datatype.Commit(P2G_S_RIZ)

    # receive types p2g
    # vector of size of #ranks
    # multi column data type for every rank size #nnz not divisible
    P2G_R = np.array([BASE_TYPE.Create_vector(
        count[1, rank], count[0, i], data_shape[0]) for i in range(size)])
    P2G_R_RIZ = np.empty_like(P2G_R)
    for i in range(size):
        P2G_R_RIZ[i] = P2G_R[i].Create_resized(0, base_size)
        MPI.Datatype.Commit(P2G_R[i])
        MPI.Datatype.Commit(P2G_R_RIZ[i])


    # define helper communication functions-------------------------------------
    # captures all variables from the outside (comm/count/disp/rank/size/types)

    def scatter_master(inp: npt.NDArray[np.complex128],
                       outp: npt.NDArray[np.complex128]):
        comm.Scatterv([inp, count[1, :], disp[1, :], COLUMN_RIZ], outp, root=0)

    def gather_master(inp: npt.NDArray[np.complex128],
                      outp: npt.NDArray[np.complex128]):
        comm.Gatherv(inp, [outp, count[1, :], disp[1, :], COLUMN_RIZ], root=0)

    def alltoall_g2p(inp: npt.NDArray[np.complex128],
                     outp: npt.NDArray[np.complex128]):
        comm.Alltoallw(
        [inp, count[0, :], disp[0, :]*base_size, np.repeat(G2P_S_RIZ, size)],
        [outp, np.repeat([1], size), disp[1, :]*base_size, G2P_R_RIZ])

    def alltoall_p2g(inp: npt.NDArray[np.complex128],
                     outp: npt.NDArray[np.complex128]):
        comm.Alltoallw(
        [inp, count[1, :], disp[1, :]*base_size, np.repeat(P2G_S_RIZ, size)],
        [outp, np.repeat([1], size), disp[0, :]*base_size, P2G_R_RIZ])


    # distribute greens function according to RGF step--------------------------

    # create local buffers
    gg_rgf = np.empty((count[1, rank], data_shape[0]),
                      dtype=np.complex128, order="C")
    gl_rgf = np.empty((count[1, rank], data_shape[0]),
                      dtype=np.complex128, order="C")
    gr_rgf = np.empty((count[1, rank], data_shape[0]),
                      dtype=np.complex128, order="C")

    # communicate from master to all other rank
    if rank == 0:
        scatter_master(gg_gold, gg_rgf)
        scatter_master(gl_gold, gl_rgf)
        scatter_master(gr_gold, gr_rgf)
    else:
        scatter_master(None, gg_rgf)
        scatter_master(None, gl_rgf)
        scatter_master(None, gr_rgf)


    # transpose of gl at every rank---------------------------------------------

    gl_trans_rgf = np.copy(gl_rgf[:,ij2ji], order="C")


    # distribute according to g2p step------------------------------------------

    # create local buffers
    gg_g2p = np.empty((count[0, rank], data_shape[1]),
                      dtype=np.complex128, order="C")
    gl_g2p = np.empty((count[0, rank], data_shape[1]),
                      dtype=np.complex128, order="C")
    gr_g2p = np.empty((count[0, rank], data_shape[1]),
                      dtype=np.complex128, order="C")
    gl_trans_g2p = np.empty((count[0, rank], data_shape[1]),
                      dtype=np.complex128, order="C")

    # use of all to all w since not divisible
    alltoall_g2p(gg_rgf, gg_g2p)
    alltoall_g2p(gl_rgf, gl_g2p)
    alltoall_g2p(gr_rgf, gr_g2p)
    alltoall_g2p(gl_trans_rgf, gl_trans_g2p)

    # calculate the polarization at every rank----------------------------------

    pg_g2p, pl_g2p, pr_g2p = g2p_gpu.g2p_fft_mpi_gpu(pre_factor[0],
                                                        gg_g2p,
                                                        gl_g2p,
                                                        gr_g2p,
                                                        gl_trans_g2p)



    # distribute polarization function according to RGF step--------------------

    # create local buffers
    pg_p2g = np.empty((count[1, rank], data_shape[0]),
                      dtype=np.complex128, order="C")
    pl_p2g = np.empty((count[1, rank], data_shape[0]),
                      dtype=np.complex128, order="C")
    pr_p2g = np.empty((count[1, rank], data_shape[0]),
                      dtype=np.complex128, order="C")

    # use of all to all w since not divisible
    alltoall_p2g(pg_g2p, pg_p2g)
    alltoall_p2g(pl_g2p, pl_p2g)
    alltoall_p2g(pr_g2p, pr_p2g)


    # culminate on master again-------------------------------------------------

    if rank == 0:
        # create buffers at master
        pg_mpi = np.empty_like(gg_gold)
        pl_mpi = np.empty_like(gg_gold)
        pr_mpi = np.empty_like(gg_gold)

        gather_master(pg_p2g, pg_mpi)
        gather_master(pl_p2g, pl_mpi)
        gather_master(pr_p2g, pr_mpi)
    else:
        gather_master(pg_p2g, None)
        gather_master(pl_p2g, None)
        gather_master(pr_p2g, None)


    # test against gold solution------------------------------------------------

    if rank == 0:
        # print difference to given solution
        # use Frobenius norm
        diff_g = np.linalg.norm(pg_gold - pg_mpi)
        diff_l = np.linalg.norm(pl_gold - pl_mpi)
        diff_r = np.linalg.norm(pr_gold - pr_mpi)
        print(f"Differences to Gold Solution g/l/r:  {diff_g:.4f}, {diff_l:.4f}, {diff_r:.4f}")

        # assert solution close to real solution
        abstol = 1e-12
        reltol = 1e-6
        assert diff_g <= abstol + reltol * np.max(np.abs(pg_gold))
        assert diff_l <= abstol + reltol * np.max(np.abs(pl_gold))
        assert diff_r <= abstol + reltol * np.max(np.abs(pr_gold))
        assert np.allclose(pg_gold, pg_mpi)
        assert np.allclose(pl_gold, pl_mpi)
        assert np.allclose(pr_gold, pr_mpi)
        print("The mpi implementation is correct.")


    # free datatypes------------------------------------------------------------

    MPI.Datatype.Free(COLUMN_RIZ)
    MPI.Datatype.Free(COLUMN)
    MPI.Datatype.Free(ROW_RIZ)
    MPI.Datatype.Free(ROW)
    MPI.Datatype.Free(G2P_S_RIZ)
    MPI.Datatype.Free(G2P_S)
    MPI.Datatype.Free(P2G_S_RIZ)
    MPI.Datatype.Free(P2G_S)
    for i in range(size):
        MPI.Datatype.Free(G2P_R_RIZ[i])
        MPI.Datatype.Free(G2P_R[i])
        MPI.Datatype.Free(P2G_R_RIZ[i])
        MPI.Datatype.Free(P2G_R[i])


    # finalize
    MPI.Finalize()
