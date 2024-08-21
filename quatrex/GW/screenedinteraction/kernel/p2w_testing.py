import concurrent.futures
from itertools import repeat
import numpy as np
import mkl
import typing
import numpy.typing as npt
from quatrex.utilities import matrix_creation
from quatrex.utilities.matrix_creation import homogenize_matrix_Rnosym, extract_small_matrix_blocks
from quatrex.block_tri_solvers import rgf_W


def p2w_pool_mpi_cpu_kpoint(
    coulomb_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    new: npt.NDArray[np.complex128],
    npw: npt.NDArray[np.complex128],
    idx_k: npt.NDArray[np.int32],
    idx_e: npt.NDArray[np.int32],
    factor: npt.NDArray[np.float64],
    comm,
    rank,
    size,
    nbc,
    homogenize: bool = False,
    NCpSC: int = 1,
    mkl_threads: int = 1,
    worker_num: int = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128], int, int]:
    """Calculates the screened interaction on the cpu.
    Uses mkl threading and pool threads.

    k-points included for this function.

    Args:
        energy (npt.NDArray[np.float64]): energy points
        pg (npt.NDArray[np.complex128]): Greater polarization, vector of sparse matrices
        pl (npt.NDArray[np.complex128]): Lesser polarization, vector of sparse matrices
        coulomb_obj (npt.NDArray[np.complex128]): Class containing the Coulomb integral information
        dosw (npt.NDArray[np.complex128]): density of state
        new (npt.NDArray[np.complex128]): density of state
        npw (npt.NDArray[np.complex128]): density of state
        factor (npt.NDArray[np.float64]): Smoothing factor
        mkl_threads (int, optional): Number of mkl threads used. Defaults to 1.
        worker_num(int, optional): Number of pool workers used. Defaults to 1.

    Returns:
        typing.Tuple[npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    int, int ]:
    Diagonal/Upper block tensor (#blocks, blocksize, blocksize) of greater, lesser, retarded screened interaction.
    Number of blocks and block size after matrix multiplication
    """
    # number of energy points
    ne = energy.shape[0]

    # number of blocks
    nb = coulomb_obj.NBlock
    # start and end index of each block in python indexing
    bmax = coulomb_obj.Bmax
    bmin = coulomb_obj.Bmin
    if bmin[0] == 1:
        bmin = bmin - 1
        bmax = bmax - 1

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # in block format
    wr_diag, wr_upper, wl_diag, wl_upper, wg_diag, wg_upper = matrix_creation.initialize_block_G(ne, nb_mm, lb_max_mm)
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)

    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    for ie in range(ne):
        # Have to check that these identities are correct for k-points
        # Anti-Hermitian symmetrizing of PL and PG
        pl[ie] = (pl[ie] - pl[ie].conj().T) / 2
        pg[ie] = (pg[ie] - pg[ie].conj().T) / 2

        # PR has to be derived from PL and PG and then has to be symmetrized
        # pr[ie] = 1j * np.imag(pr[ie])  # (pg[ie] - pl[ie]) / 2
        pr[ie] = (pg[ie] - pl[ie]) / 2
        # pr[ie] = np.real(pr[ie])

        if homogenize:
            (PR00, PR01, PR10, _) = extract_small_matrix_blocks(pr[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pr[ie][bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1],
                                                                pr[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pr[ie] = homogenize_matrix_Rnosym(PR00,
                                              PR01,
                                              PR10, len(bmax))
            (PL00, PL01, PL10, _) = extract_small_matrix_blocks(pl[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pl[ie][bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1],
                                                                pl[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pl[ie] = homogenize_matrix_Rnosym(PL00,
                                              PL01,
                                              PL10,
                                              len(bmax))
            (PG00, PG01, PG10, _) = extract_small_matrix_blocks(pg[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pg[ie][bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1],
                                                                pg[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pg[ie] = homogenize_matrix_Rnosym(PG00,
                                              PG01,
                                              PG10,
                                              len(bmax))

    # Here I need a generator as for the calculation of the retarded Green's function
    rgf_Coul = generator_rgf_Coulomb(idx_k, coulomb_obj)
    # Create a process pool with num_worker workers
    ref_flag = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        results = executor.map(
                    rgf_W.rgf_w_opt,
                    rgf_Coul,
                    pg, pl, pr,
                    repeat(bmax), repeat(bmin),
                    wg_diag, wg_upper,
                    wl_diag, wl_upper,
                    wr_diag, wr_upper,
                    xr_diag, dosw, new, npw, repeat(nbc),
                    idx_e, factor,
                    repeat(NCpSC),
                    repeat(block_inv),
                    repeat(use_dace),
                    repeat(validate_dace),
                    repeat(ref_flag))
        for res in results:
           assert isinstance(res, np.ndarray)

    # Calculate F1, F2, which are the relative errors of WR-WA = WG-WL
    F1 = np.max(np.abs(dosw - (new + npw)) / (np.abs(dosw) + 1e-6), axis=1)
    F2 = np.max(np.abs(dosw - (new + npw)) / (np.abs(new + npw) + 1e-6), axis=1)

    buf_recv_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_recv_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    if size > 1:
        if rank == 0:
            buf_send_r[:] = dosw[ne - 1, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = dosw[ne - 1, :]
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1)

    # Remove individual peaks (To-Do: improve this part by sending boundary elements to the next process)
    if size == 1:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    elif rank == 0:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))
    elif rank == size - 1:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    else:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) | ((dDOSm > 5) & (dDOSp > 5)))[0]

    # Remove the identified peaks and errors
    for index in ind_zeros:
        wr_diag[index, :, :, :] = 0
        wr_upper[index, :, :, :] = 0
        wl_diag[index, :, :, :] = 0
        wl_upper[index, :, :, :] = 0
        wg_diag[index, :, :, :] = 0
        wg_upper[index, :, :, :] = 0

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm, ind_zeros
