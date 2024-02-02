# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved."""
"""
This script is an attempt to create a class with similar functionality as the OMENHamClass.py, but
it is not based on OMEN output files. Instead, dedicated python scripts are used to create the
required matrices. This is done to avoid the need to run OMEN...
"""

import numpy as np
from scipy import sparse
import numpy.typing as npt
import matplotlib.pylab as plt
import pickle

from quatrex.utils.read_utils import read_file_to_float_ndarray


def find_block_sizes(matrix: sparse.spmatrix) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Finds the block sizes of a matrix. The algorithm is very crude and assumes that
    the number of off-diagonal elements for each row don't decrease with more than 1 as you
    go to the next row.

    It also only checks the upper triangular part of the matrix, but the same methodology can be used to
    study the lower triangular part.

    Parameters
    ----------
    matrix : spmatrix
        Matrix you want to find the block structure for.

    Returns
    -------
    Bmin : ndarray
        List of indices for the start of each block
    Bmax : ndarray
        List of indices for the end of each block. Stupid, I know
    """
    size = matrix.shape[0]
    # row and column indices of nonzero elements
    rows, cols = _nz_indices(matrix)
    # index (for column) of last nonzero element for each row (bandwidth). Don't really need all...
    inds = _indices_of_value_change(rows)
    # Block indices
    Binds = [0]
    # Create one block at the time
    while Binds[-1] < size - 1:
        Binds.append(cols[inds[Binds[-1]]] + 1)
    Bmin = np.array(Binds[:-1])
    Bmax = np.array(Binds[1:]) - 1
    return Bmin, Bmax


def _indices_of_value_change(vector: npt.NDArray) -> npt.NDArray:
    """
    Returns the last index before the value changes in the 'vector'.

    Parameters
    ----------
    vector : ndarray
        A 1D-array with values

    Returns
    -------
     : ndarray
        A 1D-array with indices where the next value in the vector changes
    """
    return np.where(vector[:-1] != vector[1:])[0]


def _nz_indices(sparse_matrix: sparse.spmatrix) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Returns a tuple of arrays containing the indices of the nozero elements of
    the sparse matrix.
    """
    return sparse_matrix.nonzero()


class Matrices:
    # Class variables
    rows: npt.NDArray = None
    columns: npt.NDArray = None

    Bmin: npt.NDArray = None
    Bmax: npt.NDArray = None

    kp: npt.NDArray = None

    def __init__(self, sim_folder, Nk, Vappl=0.0, bias_point=0, potential_type='linear', rank=0):
        """
        Parameters
        ----------
        sim_folder : str
            Path to the simulation folder, which is where the Hamiltonian, Coulomb integral and wannier positions are stored.
        Nk : ndarray
            Number of k-points. Nk[0]-transport, Nk[1]-confined, Nk[2]-periodic
        Vappl : float, optional
            Applied bias in eV. The default is 0.0. Used for linear potential drop(?)
        bias_point : int, optional
            Bias point read from self.Vbias(?). The default is 0.
        potential_type : str, optional
            Type of potential. The default is 'linear'. Possible values are 'linear', 'unit_cell', 'atomic'.
        rank : int, optional
            MPI rank. The default is 0.
        """

        if not rank:
            self.sim_folder = sim_folder
            self.Vappl = Vappl
            self.bias_point = bias_point

            # Read the Hamiltonian and Coulomb matrix, which are dictionaries with device cell shifts as keys.
            with open(sim_folder + '/Hamiltonian.pkl', 'rb') as f:
                self.Hamiltonian = pickle.load(f)
            with open(sim_folder + '/Coulomb_matrix.pkl', 'rb') as f:
                self.Coulomb_matrix = pickle.load(f)
            # Read positions of the Wannier functions
            self.position_vector = np.load(sim_folder + 'position_vector.npy')

            # Size
            self.size = self.Hamiltonian[(0, 0, 0)].shape[0]
            # S matrix. Currently only a identity matrix because the Wannier functions are orthonormal.
            self.Overlap = {}
            self.Overlap[(0, 0, 0)] = sparse.identity(self.size, format='csr')

            # Hartree potentials
            if potential_type == 'linear':
                self.Vpot = self.get_linear_potential_drop()
            elif potential_type == 'unit_cell':
                self.Vbias = read_file_to_float_ndarray(
                    sim_folder + '/Vpot.dat', ",")
                self.Vpot = self.get_unit_cell_potential()
            elif potential_type == 'atomic':
                self.Vatom = read_file_to_float_ndarray(
                    sim_folder + '/Vatom.dat', ",")
                self.Vpot = self.get_atomic_potential()
            # Adds the potential to the Hamiltonian
            self.add_potential()

            # Set kpoints (MP grid)
            self.kp = self.k_points(Nk)
            # Number of kpoints
            self.nkpts = np.prod(Nk)
            # create k-dependent Hamiltonian
            self.k_Hamiltonian = self.create_k_matrix(self.Hamiltonian)
            # create k-dependent Coulomb matrix
            self.k_Coulomb_matrix = self.create_k_matrix(self.Coulomb_matrix)

            # Prepare block properties. Should be done from the k_Hamiltonian
            self.NBlocks, self.Bmin, self.Bmax = self.prepare_block_properties(self.k_Hamiltonian[(0, 0, 0)])
            assert self.size == self.Bmax[-1] - self.Bmin[0] + 1, f"Size of the Ham.: ({self.size}) does not match Bmax[-1] - Bmin[0] +1: ({self.Bmax[-1] - self.Bmin[0]+1})"

            # returns the sparse indices of the k_Hamiltonian
            self.columns, self.rows = self.map_sparse_indices(self.k_Hamiltonian[(0, 0, 0)])

    def spy_hamiltonian(self):
        """
        Spy plot of the Hamiltonian.
        """
        fig, axes = plt.subplots(nrows=1, ncols=len(self.keys))
        fig.set_figheight = 3
        fig.set_figwith = 10
        for i, key in enumerate(self.Hamiltonian.keys()):
            axes[i].set_title(key)
            axes[i].spy(self.Hamiltonian[key])
            axes[i].grid(True)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def spy_hamiltonian_grid(self, grid):
        """
        Function that plots the Hamiltonian on a grid(?)
        """
        fig, axes = plt.subplots()
        fig.set_figheight = 3
        fig.set_figwith = 3
        axes.set_title('H[(0,0,0)]')
        axes.spy(self.Hamiltonian[(0, 0, 0)])
        axes.set_xticks(grid, minor=False)
        axes.set_yticks(grid, minor=True)
        axes.xaxis.grid(True, which='major')
        axes.yaxis.grid(True, which='minor')
        axes.set_axisbelow(False)
        plt.show()

    def write_hamiltonian(self, write_dir='.'):
        """
        This method has not yet been updated. But it could be interesting to have...
        """
        for i in np.arange(0, len(self.keys)):
            H = self.Hamiltonian[self.keys[i]]
            write_file = write_dir + '/' + self.keys[i] + '.bin'
            h = H.tocoo()
            blob2 = np.zeros((h.size, 4))
            head2 = np.array([h.shape[0], h.nnz, 0.0], dtype=float)
            blob2[:, 0] = h.col
            blob2[:, 1] = h.row
            blob2[:, 2] = np.real(h.data)
            blob2[:, 3] = np.imag(h.data)
            M2 = np.reshape(blob2, (blob2.size, 1))
            np.concatenate((np.reshape(head2, (3, 1)), M2)).astype(
                'double').tofile(write_file)

    def prepare_block_properties(self, matrix):
        """
        Function that prepares the block properties of the Hamiltonian. Not sure where it is needed.

        Based on the matrix_assembler function called find_block_sizes.
        Parameters
        ----------
        matrix : sparse matrix
            Hamiltonian or Coulomb matrix
        Returns
        -------
        Bmin,Bmax : ndarray of length NBlocks (number of blocks) of type uint
            Arrays with start and end orbital indices of all blocks
        """
        Bmin, Bmax = find_block_sizes(matrix)
        NBlocks = len(Bmin)  # or len(Bmax)
        return NBlocks, Bmin, Bmax

    def map_sparse_indices(self, k_Matrix):
        """
        This functions generates the content of the rows and cols field.
        The meaning of rows and cols are the non-zero indices of the sparse system matrices,
        Using these indices, the transformation between sparse, block and energy contigous formats can be done.

        This should be calculated from the k_Hamiltonian.

        Parameters
        ----------
        k_Matrix : sparse matrix
            Hamiltonian or Coulomb matrix
        Returns
        -------
        columns, rows : ndarray
            Non-zero indices of the sparse system matrices
        """
        columns, rows = _nz_indices(k_Matrix)
        return columns, rows

    def get_linear_potential_drop(self, position_vector=None, Vappl=None):
        """
        This function returns the linear potential drop for the current system.
        The potential drop is calculated from the difference in the applied bias in the left and right reservoirs.

        Parameters
        ----------
        position_vector : ndarray, optional
            Positions of the Wannoier centers. The default is None.
        Returns
        -------
        V : ndarray
            Linear potential drop in eV.
        """
        if position_vector is None:
            position_vector = self.position_vector
        if Vappl is None:
            Vappl = self.Vappl
        # Only interested in the x-coordinate (transport direction)
        x = position_vector[:, 0]
        xmin = np.min(x)
        xmax = np.max(x)
        Vpot = - Vappl * (x - xmin) / (xmin - xmax)
        return Vpot

    def _get_unit_cell_potential(self,):
        """
        This function extracts the potential for a particular bias from the OMEN solver with unit cell resolution

        Not working yet...

        Returns
        -------
        Vpot : ndarray
            Potential of each atom in eV

        """
        self.NA = self.LM.shape[0]

        orb_per_at_loc = self.no_orb[self.LM[:, 3].astype(int)-1]

        Vpercell = self.Vbias[:, self.bias_point]
        no_of_cells = Vpercell.shape[0]
        no_at_per_cell = int(self.NA/no_of_cells)

        V = np.zeros(self.NA)

        for IC in range(no_of_cells):
            V[IC:IC+no_at_per_cell] = Vpercell[IC]

        Vpot = np.zeros(np.sum(orb_per_at_loc))

        ind = 0
        for IA in range(self.NA):
            Vpot[ind:ind+orb_per_at_loc[IA]] = V[IA]
            ind += orb_per_at_loc[IA]

        return Vpot

    def _get_atomic_potential(self,):
        """
        This function extracts the potential for a particular bias from the OMEN solver with atomistic resolution.

        Not working yet... Need a orb_per_at_loc for this to work. Have code for it somewhere...

        Returns
        -------
        Vpot : float
            Potential at each atom in eV

        """
        self.NA = self.LM.shape[0]
        orb_per_at_loc = self.no_orb[self.LM[:, 3].astype(int)-1]

        Vpot = np.zeros(np.sum(orb_per_at_loc))

        ind = 0
        for IA in range(self.NA):
            Vpot[ind:ind+orb_per_at_loc[IA]] = self.Vatom[IA, self.bias_point]
            ind += orb_per_at_loc[IA]

        return Vpot

    def add_potential(self, ):
        """
        This function adds the linear potential drop to the Hamiltonian.

        Returns
        -------
        None.

        """
        Vpot = self.Vpot
        indi, indj = np.nonzero(self.Overlap[(0, 0, 0)])
        for IP in range(len(indi)):
            self.Hamiltonian[(0, 0, 0)][indi[IP], indj[IP]] += (Vpot[indi[IP]] +
                                                                Vpot[indj[IP]]) * self.Overlap[(0, 0, 0)][indi[IP], indj[IP]] / 2.0

    def k_points(self, Nk=None, mode='grid'):
        """
        Creates the k-point vector.

        The grid always includes the origin (gamma-point). Should grid be shifted to band gap?

        Parameters
        ----------
        Nk : ndarray
            number of k-points. Nk[0]-transport, Nk[1]-confined, Nk[2]-periodic
        mode : str, optional
            k-point mode (either 'line' or 'grid'). Line could be used for a bandstructure calculation

        Returns
        -------
        k : ndarray
            k-points
        """

        if Nk is None:
            Nk = self.Nk

        if mode == 'grid':
            kx = (np.arange(Nk[0]) - int(Nk[0]/2)) / Nk[0]
            ky = (np.arange(Nk[1]) - int(Nk[1]/2)) / Nk[1]
            kz = (np.arange(Nk[2]) - int(Nk[2]/2)) / Nk[2]
            k_mesh = np.meshgrid(kx, ky, kz, indexing='ij')
            k = np.zeros((np.prod(Nk), 3))
            k[:, 0] = k_mesh[0].ravel()
            k[:, 1] = k_mesh[1].ravel()
            k[:, 2] = k_mesh[2].ravel()
        else:
            raise ValueError(
                'Only "grid" is yet a valid mode. "line" is comming...')
        return k

    def create_k_matrix(self, int_mat=None, kp=None):
        """
        Constructs the full matrix from an interaction matrix. Here, it only works for
        H_3, H_4, and H_5.

        ## Parameters
        int_mat: dict
            interactions (Hamiltonian or Coulomb)
        kp: ndarray
            k-points

        ## Returns
        Mk: dict
            k-dependent Hamiltonian/Coulomb integral
        """

        if int_mat is None:
            int_mat = self.Hamiltonian
        if kp is None:
            kp = self.kp

        Nk = kp.shape[0]
        Mk = {}

        # Tile the matrix with the appropriate blocks of int_mat
        for ik in range(Nk):
            k_key = tuple(kp[ik])
            for key in int_mat.keys():
                rp = np.array(key)
                if k_key not in Mk.keys():
                    Mk[k_key] = np.exp(2 * np.pi * 1j * rp @
                                       kp[ik]) * int_mat[key]
                # Not sure this is needed...
                else:
                    Mk[k_key] += np.exp(2 * np.pi * 1j * rp @
                                        kp[ik]) * int_mat[key]
        return Mk
