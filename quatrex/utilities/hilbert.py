"""This file is based on/copied from the GPAW hilbert transform implementation. Which is based on the paper: 
    M. Shishkin and G. Kresse, Implementation and performance of the frequency-dependent GW method within the PAW framework, Phys. Rev. B 74, 035101 (2006).

    See also abinit implementation: https://github.com/abinit/abinit/blob/master/src/70_gw/m_chi0tk.F90
    """
import numpy as np


class HilbertTransform:
    def __init__(self, omega_w, eta, timeordered=False, gw=False,
                 blocksize=500):
        """Analytic Hilbert transformation using linear interpolation.

        Hilbert transform::

           oo
          /           1                1
          |dw' (-------------- - --------------) S(w').
          /     w - w' + i eta   w + w' + i eta
          0

        With timeordered=True, you get::

           oo
          /           1                1
          |dw' (-------------- - --------------) S(w').
          /     w - w' - i eta   w + w' + i eta
          0

        With gw=True, you get::

           oo
          /           1                1
          |dw' (-------------- + --------------) S(w').
          /     w - w' + i eta   w + w' + i eta
          0

        """

        self.blocksize = blocksize

        if timeordered:
            self.H_ww = self.H(omega_w, -eta) + self.H(omega_w, -eta, -1)
        elif gw:
            self.H_ww = self.H(omega_w, eta) - self.H(omega_w, -eta, -1)
        else:
            self.H_ww = self.H(omega_w, eta) + self.H(omega_w, -eta, -1)

    def H(self, o_w, eta, sign=1):
        """Calculate transformation matrix.

        With s=sign (+1 or -1)::

                        oo
                       /       dw'
          X (w, eta) = | ---------------- S(w').
           s           / s w - w' + i eta
                       0

        Returns H_ij so that X_i = np.dot(H_ij, S_j), where::

            X_i = X (omega_w[i]) and S_j = S(omega_w[j])
                   s
        """

        nw = len(o_w)
        H_ij = np.zeros((nw, nw), complex)
        do_j = o_w[1:] - o_w[:-1]
        for i, o in enumerate(o_w):
            d_j = o_w - o * sign
            y_j = 1j * np.arctan(d_j / eta) + 0.5 * np.log(d_j**2 + eta**2)
            y_j = (y_j[1:] - y_j[:-1]) / do_j
            H_ij[i, :-1] = 1 - (d_j[1:] - 1j * eta) * y_j
            H_ij[i, 1:] -= 1 - (d_j[:-1] - 1j * eta) * y_j
        return H_ij

    def __call__(self, S_wx):
        """Inplace transform"""
        B_wx = S_wx.reshape((len(S_wx), -1))
        nw, nx = B_wx.shape
        tmp_wx = np.zeros((nw, min(nx, self.blocksize)), complex)
        for x in range(0, nx, self.blocksize):
            b_wx = B_wx[:, x:x + self.blocksize]
            c_wx = tmp_wx[:, :b_wx.shape[1]]
            mmm(1.0, self.H_ww, 'N', b_wx, 'N', 0.0, c_wx)
            b_wx[:] = c_wx


class GWHilbertTransforms:
    """Helper class which wraps two transforms using contiguous array.

    (This slightly speeds up things.)"""
    def __init__(self, omega_w, eta):
        self.htp = htp = HilbertTransform(omega_w, eta, gw=True)
        self.htm = htm = HilbertTransform(omega_w, -eta, gw=True)
        self._stacked_H_nww = np.array([htp.H_ww, htm.H_ww])

    def __call__(self, A_wGG):
        # (Note: This effectively duplicates the Hilbert call)
        nw = len(A_wGG)
        H_xw = self._stacked_H_nww.reshape(-1, nw)
        A_wy = A_wGG.reshape(nw, -1)
        tmp_xy = np.zeros((H_xw.shape[0], A_wy.shape[1]), complex)
        # gemm(1.0, A_wy, H_xw, 0.0, tmp_xy)
        mmm(1.0, H_xw, 'N', A_wy, 'N', 0.0, tmp_xy)
        return tmp_xy.reshape((2, *A_wGG.shape))
