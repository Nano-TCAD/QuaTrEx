# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved."""

import numpy as np
import glob
from scipy import sparse
from scipy.sparse import identity
import matplotlib.pylab as plt
from scipy.interpolate import griddata

# TODO: Import only what is needed. If many methods are needed, import the whole module under a shorter name.
from quatrex.utilities.read_utils import *
from quatrex.utilities.matrix_creation import extract_small_matrix_blocks, homogenize_matrix_Rnosym


def matlab_fread(fid, nelements, dtype):
    """Equivalent to Matlab fread function"""

    if dtype is np.str_:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array


# Hamiltonians from binary files, contains all the OMEN block properties.
class Hamiltonian:
    # Class variables
    LM = None
    LM_map = None
    rows = None
    columns = None
    Smin = None

    Bmin = None
    Bmax = None

    no_orb = None
    NBlock = None
    orb_per_at = None

    NH = None
    NA = None
    NB = None
    TB = None

    kp = None
    
    
    def __init__(self,sim_folder, no_orb, Nk, Vappl = 0.0, bias_point = 0, potential_type = 'linear', layer_matrix = '/Layer_Matrix.dat', homogenize=False, rank = 0):
        if(not rank):
            self.no_orb = no_orb # Number orbitals per atom type
            self.sim_folder = sim_folder
            self.blocks = glob.glob(self.sim_folder + '/H*.bin')
            self.Sblocks = glob.glob(self.sim_folder + '/S*.bin')

            self.keys = [i.rsplit('/')[-1].rsplit('.bin')[0] for i in self.blocks]

            self.Vappl = Vappl
            self.bias_point = bias_point
            #self.Hamiltonian = []
            #for p in self.blocks:
            #   self.Hamiltonian.append(read_sparse_matrix(p))

            self.Hamiltonian = {}
            self.Overlap = {}

            #If orthogonal basis set, define S as diagonal.
            if not self.Sblocks:
                for ii in range(0, len(self.blocks)):
                    self.Hamiltonian[self.keys[ii]] = self.read_sparse_matrix(self.blocks[ii])
                    #self.NH[self.keys[ii]] = self.Hamiltonian[self.keys[ii]].shape[0]
                    self.NH = self.Hamiltonian[self.keys[ii]].shape[0]
                    self.Overlap[self.keys[ii]] = sparse.identity(self.Hamiltonian[self.keys[ii]].shape[0],
                                                                  dtype=np.cfloat,
                                                                  format='csr')
            #Otherwise read from file
            else:
                for ii in range(0, len(self.blocks)):
                    self.Hamiltonian[self.keys[ii]] = self.read_sparse_matrix(self.blocks[ii])
                    self.Overlap[self.keys[ii]] = self.read_sparse_matrix(self.Sblocks[ii])

                if np.abs(self.Overlap[self.keys[ii]][0, 0] + 1) < 1e-6:
                    self.Overlap[self.keys[ii]] = sparse.identity(self.Hamiltonian[self.keys[ii]].shape[0],
                                                                  dtype=np.cfloat,
                                                                  format='csr')

            #Check that hamiltonian is hermitean
            self.hermitean = self.check_hermitivity(tol=1e-6)
            #self.hermitean = True

            # Set kpoints
            self.kp = self.k_points(Nk)
            self.nkpts = np.prod(Nk)
            # create k-dependent Hamiltonian
            self.k_Hamiltonian = self.create_k_matrix()

            #Read Block Properties
            self.LM = read_file_to_float_ndarray(sim_folder + layer_matrix)
            self.NA = self.LM.shape[0]  # Number of atoms
            self.NB = self.LM.shape[1] - 4 # Number of Neighboors
            self.TB = np.max(self.no_orb)  # Max number of orbitals per atom
            self.Smin = read_file_to_int_ndarray(sim_folder + '/Smin_dat')
            self.Smin = self.Smin.reshape((self.Smin.shape[0], )).astype(int)
            self.prepare_block_properties()
            self.map_neighbor_indices()
            self.map_sparse_indices()
            if(homogenize):
                self.homogenize(NCpSC)
            if(potential_type == 'linear'):
                self.Vpot = self.get_linear_potential_drop()
            elif (potential_type == 'unit_cell'):
                self.Vbias = read_file_to_float_ndarray(sim_folder + '/Vpot.dat', ",")
                self.Vpot = self.get_unit_cell_potential()
            elif (potential_type == 'atomic'):
                self.Vatom = read_file_to_float_ndarray(sim_folder + '/Vatom.dat', ",")
                self.Vpot = self.get_atomic_potential()
            self.add_potential()

    #Helper function to initialise all hamiltonians
    def read_sparse_matrix(self, fname='./H_4.bin'):

        fid = open(fname, 'rb')

        #Read the Header head[2]: Contains the format 0 if indexing starts at 0, 1 if indexing starts at 1
        #                head[1]: number of non-zero elements
        #                head[0]: matrix size
        head = matlab_fread(fid, 3, 'double')

        #Read the data and reshape it
        M = matlab_fread(fid, 4 * int(head[1]), 'double')
        blob = np.reshape(M, (int(head[1]), 4))
        fid.close()

        #Matrix indexes start with zero
        if head[2] == 0:
            H = sparse.csc_matrix((blob[:, 2] + 1j * blob[:, 3], (blob[:, 0], blob[:, 1])),
                                  shape=(int(head[0]), int(head[0])))
        #Matrix indexes start with one
        else:
            H = sparse.csc_matrix((blob[:, 2] + 1j * blob[:, 3], (blob[:, 0] - 1, blob[:, 1] - 1)),
                                  shape=(int(head[0]), int(head[0])))

        return H

    def write_sparse_matrix(self, H, write_file='./H_4.bin'):

        h = H.tocoo()
        blob2 = np.zeros((h.size, 4))
        head2 = np.array([h.shape[0], h.nnz, 0.0], dtype=float)

        blob2[:, 0] = h.col
        blob2[:, 1] = h.row
        blob2[:, 2] = np.real(h.data)
        blob2[:, 3] = np.imag(h.data)

        M2 = np.reshape(blob2, (blob2.size, 1))

        np.concatenate((np.reshape(head2, (3, 1)), M2)).astype('double').tofile(write_file)

    def spy_hamiltonian(self):

        fig, axes = plt.subplots(nrows=1, ncols=len(self.keys))
        fig.set_figheight = 3
        fig.set_figwith = 10

        for i in np.arange(0, len(self.keys)):
            axes[i].set_title(self.keys[i])
            axes[i].spy(self.Hamiltonian[self.keys[i]])
            axes[i].grid(True)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def spy_hamiltonian_grid(self, grid):

        fig, axes = plt.subplots()
        fig.set_figheight = 3
        fig.set_figwith = 3
        axes.set_title('H_4')
        axes.spy(self.Hamiltonian[self.keys[1]])

        axes.set_xticks(grid, minor=False)
        axes.set_yticks(grid, minor=True)
        axes.xaxis.grid(True, which='major')
        axes.yaxis.grid(True, which='minor')
        axes.set_axisbelow(False)
        plt.show()

    def check_hermitivity(self, tol=1e-6):
        #Check if the hamiltonian is hermitean.
        #
        # H = H_4 + H_3*e^{ikL} + H_5*e^{-ikL}
        #
        # H-H^{h} = 0 --> (H_4-H_4^{h}) + (H_3 - H_5^{h})*e^{ikL} + (H_5 - H_3^{h})*e^{-ikL} = 0
        #
        # and therefore each braket must be zero.

        full_filled = True
        #err35 = np.max(np.absolute(self.Hamiltonian['H_3'] - self.Hamiltonian['H_5'].H))
        err44 = np.max(np.absolute(self.Hamiltonian['H_4'] - self.Hamiltonian['H_4'].H))

        #print(type(err35))
        #print(err35.max().min())
        if err44 > tol:
            full_filled = False
            print("Error 44: " + str(err44))

        return full_filled

    def cut_hamiltonian(self, R):
        for ii in range(0, len(self.blocks)):
            H = self.Hamiltonian[self.keys[ii]]
            self.Hamiltonian[self.keys[ii]] = R * H * R.transpose()

    def write_hamiltonian(self, write_dir='.'):
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

            np.concatenate((np.reshape(head2, (3, 1)), M2)).astype('double').tofile(write_file)

    def prepare_block_properties(self, ):
        """

        Parameters
        ----------
        no_orb : list of ints
            Number of Orbitals per atom type
        LM : OMEN Layer Matrix Format
            First three columns are atom coordinates, fourth column is atom type remaining columns are neighbor indices
        Smin : ndarray of int64
            First element is number of blocks, remaining elements are
            starting atom index of each block, last element indicates the end of last layer.

        Returns
        -------
        orb_per_at : ndarray of type uint
            Orbital starting index of each atom in structure
        Bmin,Bmax : ndarray of length NBlock (number of blocks) of type uint
            Arrays with start and end orbital indices of all blocks

        """
        NA = self.LM.shape[0]
        orb = self.no_orb[self.LM[:, 3].astype(int) - 1]

        self.orb_per_at = np.zeros((NA + 1, ), dtype=int)  #array with orbital index per atom
        self.orb_per_at[0] = 1

        for i in range(NA):
            self.orb_per_at[i + 1] = self.orb_per_at[i] + orb[i]

        self.NBlock = self.Smin[0]
        self.Bmin = self.orb_per_at[self.Smin[1:-1] - 1]
        # Shouldn't Bmin and Bmax match? 
        self.Bmax = np.append(self.Bmin[1:] - 1, np.max(self.orb_per_at) - 1).astype(int)

    def map_neighbor_indices(self, ):
        """
        This functions generates the content of the field LM_map.
        Assume atom i and atom j are neighbors. Using the layer matrix it holds that
        LM(i,4+IB_i)=j. However it doesn't hold LM(j,4+IB_i)=i, instead we 
        find the reciprocal neighbor with LM(j,4+IB_j)=i. 
        
        LM_map has the following property: LM_map(i,4+IB_i)=IB_j and LM_map(j,4+IB_j)=IB_i

        Returns
        -------
        None.

        """
        self.LM_map = np.copy(self.LM)

        for IA in range(self.NA):
            for IB1 in range(self.NB):
                if self.LM[IA, 4 + IB1] > 0:
                    neigh = self.LM[IA, 4 + IB1].astype(int)

                    for IB2 in range(self.NB):

                        if self.LM[neigh - 1, 4 + IB2].astype(int) == IA + 1:
                            self.LM_map[IA, 4 + IB1] = IB2 + 1
                            break

    def map_sparse_indices(self, ):
        """
        This functions generates the content of the rows and cols field.
        The meaning of rows and cols are the non-zero indices of the sparse system matrices,
        they are derived from the neighbor information in the layer matrix.
        Using these indices, the transformation between sparse, block and energy contigous formats can be done.
        """
        self.NH = self.Bmax[-1]
        indI = np.zeros((self.NH * (self.NB + 1) * self.TB, ), dtype=int)
        indJ = np.zeros((self.NH * (self.NB + 1) * self.TB, ), dtype=int)
        ind = 0

        indA = 0

        for IA in range(self.NA):
            indR = self.orb_per_at[IA] - 1
            orbA = self.orb_per_at[IA + 1] - self.orb_per_at[IA]

            for IB in range(self.NB + 1):
                add_element = 1

                if IB == 0:
                    indC = indR
                    orbB = orbA
                else:
                    if self.LM[IA, 4 + IB - 1] > 0:
                        neigh = int(self.LM[IA, 4 + IB - 1])
                        indC = self.orb_per_at[neigh - 1] - 1
                        orbB = self.orb_per_at[neigh] - self.orb_per_at[neigh - 1]
                    else:
                        add_element = 0

                if add_element:
                    if (np.max(self.no_orb) == 1):
                        indI[ind:ind + orbA * orbB] = np.sort(
                            np.reshape(np.outer(np.arange(indR, indR + orbA), np.ones((1, orbB))), (1, orbA * orbB)))
                        indJ[ind:ind + orbA * orbB] = np.sort(
                            np.reshape(np.outer(np.ones((orbA, 1)), np.arange(indC, indC + orbB)), (1, orbA * orbB)))
                    else:
                        indI[ind:ind + orbA * orbB] = np.reshape(
                            np.outer(np.arange(indR, indR + orbA), np.ones((1, orbB))), (1, orbA * orbB))
                        indJ[ind:ind + orbA * orbB] = np.reshape(
                            np.outer(np.ones((orbA, 1)), np.arange(indC, indC + orbB)), (1, orbA * orbB))
                    ind += orbA * orbB
            if (np.max(self.no_orb) == 1):
                indI[indA:ind] = np.sort(indI[indA:ind])
                indJ[indA:ind] = np.sort(indJ[indA:ind])
            indA = ind
        self.columns = indI[:ind].astype('int32')
        self.rows = indJ[:ind].astype('int32')

    def get_linear_potential_drop(self, ):
        """
        This function returns the linear potential drop for the current system.
        The potential drop is calculated from the difference in the applied bias in the left and right reservoirs.

        Returns
        -------
        V : float
            Linear potential drop in eV.

        """
        self.NBlock = self.Smin[0]
        ABmin = self.Smin[1:]
        self.NA = self.LM.shape[0]

        x = self.LM[:, 0].astype(float)
        orb_per_at_loc = self.no_orb[self.LM[:, 3].astype(int) - 1]
        indL = np.arange(0, ABmin[2] - 1)
        indR = np.arange(ABmin[self.NBlock - 2] - 1, self.NA)
        indC = np.arange(ABmin[2] - 1, ABmin[self.NBlock - 2] - 1)

        V = np.zeros(self.NA)

        V[indL] = 0
        V[indR] = -self.Vappl
        V[indC] = -self.Vappl * (x[indC] - x[indC[0]]) / (x[indC[-1]] - x[indC[0]])

        Vpot = np.zeros(np.sum(orb_per_at_loc))

        ind = 0
        for IA in range(self.NA):
            Vpot[ind:ind + orb_per_at_loc[IA]] = V[IA]
            ind += orb_per_at_loc[IA]

        return Vpot
    
    def get_unit_cell_potential(self,):
        """
        This function extracts the potential for a particular bias from the OMEN solver with unit cell resolution
        Returns
        -------
        Vpot : float
            Potential of each atom in eV 

        """
        self.NBlock = self.Smin[0]
        self.NA = self.LM.shape[0]

        orb_per_at_loc = self.no_orb[self.LM[:,3].astype(int)-1]

        Vpercell = self.Vbias[:,self.bias_point]
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
    
    def get_atomic_potential(self,):
        """
        This function extracts the potential for a particular bias from the OMEN solver with atomistic resolution
        Returns
        -------
        Vpot : float
            Potential at each atom in eV 

        """
        self.NBlock = self.Smin[0]
        self.NA = self.LM.shape[0]
        orb_per_at_loc = self.no_orb[self.LM[:,3].astype(int)-1]

        Vpot = np.zeros(np.sum(orb_per_at_loc))

        ind = 0
        for IA in range(self.NA):
            Vpot[ind:ind+orb_per_at_loc[IA]] = self.Vatom[IA,self.bias_point]
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

        indi, indj = np.nonzero(self.Overlap['H_4'])

        for IP in range(len(indi)):
            self.Hamiltonian['H_4'][indi[IP],
                                    indj[IP]] += (Vpot[indi[IP]] + Vpot[indj[IP]]) * self.Overlap['H_4'][indi[IP],
                                                                                                         indj[IP]] / 2.0
    def homogenize(self, NCpSC):
        """
        This function homogenizes the Hamiltonian and Overlap matrices.
        """
        
        (H00, H01, H10, _) = extract_small_matrix_blocks(self.Hamiltonian['H_4'][self.Bmin[0] - 1:self.Bmax[0], self.Bmin[0] - 1:self.Bmax[0]],\
                                                                      self.Hamiltonian['H_4'][self.Bmin[0] - 1:self.Bmax[0], self.Bmin[1] - 1:self.Bmax[1]],\
                                                                      self.Hamiltonian['H_4'][self.Bmin[1] - 1:self.Bmax[1], self.Bmin[0] - 1:self.Bmax[0]], NCpSC, 'L')
        
        self.Hamiltonian['H_4'] = homogenize_matrix_Rnosym(H00, H01, H10, len(self.Bmin))


        (S00, S01, S10, _) = extract_small_matrix_blocks(self.Overlap['H_4'][self.Bmin[0] - 1:self.Bmax[0], self.Bmin[0] - 1:self.Bmax[0]],\
                                                                      self.Overlap['H_4'][self.Bmin[0] - 1:self.Bmax[0], self.Bmin[1] - 1:self.Bmax[1]],\
                                                                      self.Overlap['H_4'][self.Bmin[1] - 1:self.Bmax[1], self.Bmin[0] - 1:self.Bmax[0]], NCpSC, 'L')
        self.Overlap['H_4'] = homogenize_matrix_Rnosym(S00, S01, S10, len(self.Bmin)) 

    def k_points(self, Nk=None, mode='grid'):
        """
        Creates the k-point vector.

        The grid always includes the origin (gamma-point).

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

        # Tile the matrix with the appropriate blocks of Hblocks
        for ik in range(Nk):
            k_key = tuple(kp[ik])
            for key in int_mat.keys():
                if key == 'H_3':
                    rp = np.array([0, -1, 0])
                elif key == 'H_4':
                    rp = np.array([0, 0, 0])
                elif key == 'H_5':
                    rp = np.array([0, 1, 0])
                else:
                    raise KeyError(f'{key} is not yet a valid key')
                if k_key not in Mk.keys():
                    Mk[k_key] = np.exp(2 * np.pi * 1j * rp @ kp[ik]) * int_mat[key]
                else:
                    Mk[k_key] += np.exp(2 * np.pi * 1j * np.array(rp) @ kp[ik]) * int_mat[key]
        return Mk

    def construct_coulomb_matrix(self, eps_r, eps0, e, diag=False, orb_uniform=False, lattice_vectors=None, int_mat=None):
        """
        Assembles the Coulomb matrix. 

        Parameters
        ----------
        eps_r : float
            Relative Dielectric permittivity
        eps0 : float
            Dielectric permittivity of vacuum
        e   : float
            Elementary charge
        lattice_vectors : ndarray, optional
            Lattice vectors. The default is None.
        int_mat : dict, optional
            Coulomb matrix elements. The default is None.
        """
        if lattice_vectors is not None:
            self.lattice_vectors = lattice_vectors
        else:
            lattice_vectors = np.eye(3)
            print('No lattice vectors defined.')

        # This is for knowing how to shift. Not very nice
        if int_mat is None:
            int_mat = self.Hamiltonian

        MR = {}

        # Tile the matrix with the appropriate blocks 
        for key in int_mat.keys():
            if key == 'H_3':
                shift = np.array([0, -1, 0]) @ lattice_vectors
            elif key == 'H_4':
                shift = np.array([0, 0, 0]) @ lattice_vectors
            elif key == 'H_5':
                shift = np.array([0, 1, 0]) @ lattice_vectors
            else:
                raise KeyError(f'{key} is not yet a valid key')
            MR[key] = self.calculate_coulomb_matrix_elements(eps_r, eps0, e, shift, diag, orb_uniform)
        return MR

    def calculate_coulomb_matrix_elements(self, eps_r, eps0, e, shift=None, diag=False, orb_uniform=False):
        """
        This function computes a placeholder for the 2-index Coulomb matrix. It
        assumes that the atomic orbitals are point charges and computes their
        coulomb repulsion based on their mutual distance. The coulomb matrix
        should be computed using the localized basis on a spacial grid. The interaction
        length is the same as for the Hamiltonian.

        Note:
        Need to read the lattice vectors from somewhere! Now it is passed as an argument.

        The way the shifts are made is not great. It reads it from the Hamiltonian dictionary.

        Parameters
        ----------
        epsR : float
            Relative Dielectric permittivity
        eps0 : float
            Dielectric permittivity of vacuum
        e   : float
            Elementary charge
            
        Returns
        -------
        V_Col : scipy.sparse.csc matix of type cfloat, same dimension as
        Hamiltonian Matrix in Hamiltonian class. (n_orbs x n_orbs)
            The coulomb Matrix

        """
        if shift is None:
            shift = np.array([0, 0, 0])

        factor = e / (4 * np.pi * eps0 * eps_r) * 1e9
        NA = self.NA  # Number of atoms
        NB = self.NB  # Number of neighbouring atoms
        TB = self.TB  # Number of orbitals per atom (only max valueG)
        V_atomic = np.zeros((NA, NB + 1, TB, TB), dtype=np.cfloat)
        SF = np.outer(np.arange(1, -0.1, -0.1), np.arange(1, -0.1, -0.1))
        Vmax = float(0.0)
        LM = self.LM  # Layer matrix
 
        for ia in range(NA):
            orbA = self.orb_per_at[ia + 1] - self.orb_per_at[ia]
            for ib in range(NB):
                if LM[ia, 4 + ib] > 0:  # or not equal to 0
                    neigh = int(LM[ia, 4 + ib] - 1)
                    orbB = self.orb_per_at[neigh + 1] - self.orb_per_at[neigh]

                    dist = np.linalg.norm(LM[neigh, 0:3] - LM[ia, 0:3] + shift)

                    if abs(dist) < 1e-24:
                        print(ia)
                        print(LM[ia, 4 + ib])
                        print(dist)

                    Vact = factor / dist

                    if Vact > Vmax:
                        Vmax = Vact
                    
                    if (orb_uniform):
                        V_atomic[ia, ib + 1, 0:orbA, 0:orbB] = Vact * np.ones((orbA, orbB), dtype = np.cfloat)
                    else:
                        V_atomic[ia, ib + 1, 0:orbA, 0:orbB] = Vact * SF[0:orbA, 0:orbB]
        
        # This second loop is for orbitals belonging to the same atom.
        for ia in range(NA):
            orbA = self.orb_per_at[ia+1] - self.orb_per_at[ia]
            if diag:
                if orb_uniform:
                    #V_atomic[ia,0, 0:orbA, 0:orbA] = 1.5 * Vmax * (np.ones((orbA, orbA), dtype = np.cfloat) - np.eye(int(orbA), dtype = np.cfloat))
                    V_atomic[ia, 0, 0:orbA, 0:orbA] = 1.5 * Vmax * (np.ones((orbA, orbA), dtype = np.cfloat))
                else:
                    V_atomic[ia, 0, 0:orbA, 0:orbA] = 1.5 * Vmax * SF[0:orbA, 0:orbA]
            elif orbA > 1:
                if orb_uniform:
                    V_atomic[ia, 0, 0:orbA, 0:orbA] = Vmax * (np.ones((orbA, orbA), dtype = np.cfloat) - np.eye(int(orbA), dtype = np.cfloat))
                else:
                    pass #not changing this as it will break the test unfortunately.   
            #V_atomic[ia,0, :orbA, :orbA] = 1.5 * Vmax * np.eye(int(orbA), dtype = np.cfloat)

        return map_4D_to_sparse(V_atomic, DH)

def map_4D_to_sparse(V_atomic, DH):
    """
    Parameters
    ----------
    V_atomic : 4-D cfloat array of coulomb elements
        First dimension specifies atom index, second dimension specifies the
        selected neighbor. The remaining two dimensions are of size of the TB order 
        (maximum number of orbitals over atoms in the structure), this means
        one can select all possible orbital combinations (for each atom and its
                                                          selected neighbor)
    DH : Device_Hamiltonian
        OMEN Hamiltonian Class with Block Properties

    Returns
    -------
    V_Col : scipy.sparse.csc matix of type complex128, same dimension as
    Hamiltonian Matrix in Hamiltonian class. (n_orbs x n_orbs)
        The coulomb Matrix

    """
    indI = np.zeros((DH.NA * (DH.NB + 1) * DH.TB * DH.TB, ), dtype=int)
    indJ = np.zeros((DH.NA * (DH.NB + 1) * DH.TB * DH.TB, ), dtype=int)
    NNZ = np.zeros((DH.NA * (DH.NB + 1) * DH.TB * DH.TB, ), dtype=complex)

    ind = 0

    for IA in range(DH.NA):

        indR = DH.orb_per_at[IA]
        orbA = DH.orb_per_at[IA + 1] - DH.orb_per_at[IA]

        for IB in range(DH.NB + 1):

            add_element = 1

            if IB == 0:
                indC = indR
                orbB = orbA

            else:
                if DH.LM[IA, 4 + IB - 1] > 0:

                    neigh = int(DH.LM[IA, 4 + IB - 1] - 1)

                    indC = DH.orb_per_at[neigh]
                    orbB = DH.orb_per_at[neigh + 1] - DH.orb_per_at[neigh]

                else:
                    add_element = 0

            if add_element:
                indI[ind:ind + orbA * orbB] = np.reshape(np.outer(np.ones((1, orbB)), np.arange(indR, indR + orbA)),
                                                         (1, orbA * orbB))
                indJ[ind:ind + orbA * orbB] = np.reshape(np.outer(np.arange(indC, indC + orbB), np.ones((orbA, 1))),
                                                         (1, orbA * orbB))
                NNZ[ind:ind + orbA * orbB] = np.reshape(np.squeeze(V_atomic[IA, IB, 0:orbA, 0:orbB]), (1, orbA * orbB))

                ind = ind + orbA * orbB

    sparse_shape = np.max(DH.orb_per_at) - DH.orb_per_at[0]
    indI_sparse = indI[:ind] - 1
    indJ_sparse = indJ[:ind] - 1
    NNZ_sparse = NNZ[:ind]

    return sparse.csr_matrix((NNZ_sparse, (indI_sparse, indJ_sparse)), shape=(sparse_shape, sparse_shape))
