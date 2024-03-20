"""This file is based on/copied from the GPAW hilbert transform implementation. Which is based on the paper: 
    M. Shishkin and G. Kresse, Implementation and performance of the frequency-dependent GW method within the PAW framework, Phys. Rev. B 74, 035101 (2006).
    """
import numpy as np

class HilbertTransform:
    def __init__(self, E, eta=1e-5):
        self.E = E
        self.hilbert = self._hilbert(E)

    def _hilbert(self, E):
        """Create the Hilbert transform matrix for a given number of points."""
        N = len(E)
        M = np.zeros((N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                if i != j:
                    M[i, j] = (E[i] - E[j]) / (E[i] - E[j] + 1j)
                else:
                    M[i, j] = 0.0
        return M