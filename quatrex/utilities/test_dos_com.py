"""
Tests the communication inside p2w for filtering the peaks
"""
import numpy as np
from mpi4py import MPI
from numpy.random import default_rng

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # create array with energy size distribution
    nes = [17, 20, 27, 112, 223]
    lbs = [1, 3, 5, 7, 9, 11, 12, 20]
    rng = default_rng(42)

    for ne in nes:
        data_per_rank = ne // size
        count = np.repeat(data_per_rank, size)
        count[size - 1] += ne % size
        disp = data_per_rank * np.arange(size)

        for lb in lbs:
            dos_global = rng.standard_normal((ne, lb))
            dDOSm_gold_global = np.concatenate(
                ([0], np.max(np.abs(dos_global[1:ne - 1, :] / (dos_global[0:ne - 2, :] + 1)),
                             axis=1), [np.max(np.abs(dos_global[ne - 1, :] / (dos_global[ne - 2, :] + 1)))]))
            dDOSp_gold_global = np.concatenate(([np.max(np.abs(dos_global[0, :] / (dos_global[1, :] + 1)))],
                                                np.max(np.abs(dos_global[1:ne - 1, :] / (dos_global[2:ne, :] + 1)),
                                                       axis=1), [0]))

            dos_loc = np.copy(dos_global[disp[rank]:disp[rank] + count[rank], :])
            buf_recv_r = np.empty((dos_loc.shape[1]), dtype=np.complex128)
            buf_send_r = np.empty((dos_loc.shape[1]), dtype=np.complex128)
            buf_recv_l = np.empty((dos_loc.shape[1]), dtype=np.complex128)
            buf_send_l = np.empty((dos_loc.shape[1]), dtype=np.complex128)
            ne_loc = dos_loc.shape[0]

            if size > 1:
                if rank == 0:
                    buf_send_r[:] = dos_loc[ne_loc - 1, :]
                    comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1)
                    assert np.allclose(buf_recv_r, dos_global[disp[rank] + count[rank], :])

                elif rank == size - 1:
                    buf_send_l[:] = dos_loc[0, :]
                    comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1)
                    assert np.allclose(buf_recv_l, dos_global[disp[rank] - 1, :])

                else:
                    buf_send_r[:] = dos_loc[ne_loc - 1, :]
                    buf_send_l[:] = dos_loc[0, :]
                    comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1)
                    comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1)
                    assert np.allclose(buf_recv_r, dos_global[disp[rank] + count[rank], :])
                    assert np.allclose(buf_recv_l, dos_global[disp[rank] - 1, :])

            if size == 1:
                dDOSm = np.concatenate(
                    ([0], np.max(np.abs(dos_loc[1:ne_loc - 1, :] / (dos_loc[0:ne_loc - 2, :] + 1)),
                                 axis=1), [np.max(np.abs(dos_loc[ne_loc - 1, :] / (dos_loc[ne_loc - 2, :] + 1)))]))
                dDOSp = np.concatenate(([np.max(np.abs(dos_loc[0, :] / (dos_loc[1, :] + 1)))],
                                        np.max(np.abs(dos_loc[1:ne_loc - 1, :] / (dos_loc[2:ne_loc, :] + 1)),
                                               axis=1), [0]))
            elif rank == 0:
                dDOSm = np.concatenate(
                    ([0], np.max(np.abs(dos_loc[1:ne_loc - 1, :] / (dos_loc[0:ne_loc - 2, :] + 1)),
                                 axis=1), [np.max(np.abs(dos_loc[ne_loc - 1, :] / (dos_loc[ne_loc - 2, :] + 1)))]))
                dDOSp = np.concatenate(([np.max(np.abs(dos_loc[0, :] / (dos_loc[1, :] + 1)))],
                                        np.max(np.abs(dos_loc[1:ne_loc - 1, :] / (dos_loc[2:ne_loc, :] + 1)),
                                               axis=1), [np.max(np.abs(dos_loc[ne_loc - 1, :] / (buf_recv_r + 1)))]))
            elif rank == size - 1:
                dDOSm = np.concatenate(
                    ([np.max(np.abs(dos_loc[0, :] / (buf_recv_l + 1)))],
                     np.max(np.abs(dos_loc[1:ne_loc - 1, :] / (dos_loc[0:ne_loc - 2, :] + 1)),
                            axis=1), [np.max(np.abs(dos_loc[ne_loc - 1, :] / (dos_loc[ne_loc - 2, :] + 1)))]))
                dDOSp = np.concatenate(([np.max(np.abs(dos_loc[0, :] / (dos_loc[1, :] + 1)))],
                                        np.max(np.abs(dos_loc[1:ne_loc - 1, :] / (dos_loc[2:ne_loc, :] + 1)),
                                               axis=1), [0]))
            else:
                dDOSm = np.concatenate(
                    ([np.max(np.abs(dos_loc[0, :] / (buf_recv_l + 1)))],
                     np.max(np.abs(dos_loc[1:ne_loc - 1, :] / (dos_loc[0:ne_loc - 2, :] + 1)),
                            axis=1), [np.max(np.abs(dos_loc[ne_loc - 1, :] / (dos_loc[ne_loc - 2, :] + 1)))]))
                dDOSp = np.concatenate(([np.max(np.abs(dos_loc[0, :] / (dos_loc[1, :] + 1)))],
                                        np.max(np.abs(dos_loc[1:ne_loc - 1, :] / (dos_loc[2:ne_loc, :] + 1)),
                                               axis=1), [np.max(np.abs(dos_loc[ne_loc - 1, :] / (buf_recv_r + 1)))]))

            assert np.allclose(dDOSm, dDOSm_gold_global[disp[rank]:disp[rank] + count[rank]])
            assert np.allclose(dDOSp, dDOSp_gold_global[disp[rank]:disp[rank] + count[rank]])
