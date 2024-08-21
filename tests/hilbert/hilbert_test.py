from quatrex.utilities.hilbert import HilbertTransform
import numpy as np

if __name__ == '__main__':
    scratch_path = '/usr/scratch/bucaramanga/awinka/quatrex_results/'
    pl = np.load(scratch_path + 'pl_g2p.npy')
    pg = np.load(scratch_path + 'pg_g2p.npy')
    energy = np.linspace(-10, 5+15, 1024, endpoint=True, dtype=float)
    ne = len(energy)
    nkpts = int(pl.shape[1]/len(energy))

    pr_hilbert = np.zeros_like(pl)

    eta = 1e-6

    ht = HilbertTransform(energy, eta=eta, quatrex=False)

    S = 1j * (pg - pl)

    for kp in range(nkpts):
        pr_hilbert[:, kp*ne:(kp+1)*ne] = ht(S[:, kp*ne:(kp+1)*ne].T).T
    np.save(scratch_path + 'pr_hilbert.npy', pr_hilbert)


