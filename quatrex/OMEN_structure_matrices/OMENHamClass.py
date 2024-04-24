# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved."""

import numpy as np
import glob
from scipy import sparse
from scipy.sparse import identity
import matplotlib.pylab as plt
from scipy.interpolate import griddata

# TODO: Import only what is needed. If many methods are needed, import the whole module under a shorter name.
from quatrex.utils.read_utils import *
from quatrex.utils.matrix_creation import extract_small_matrix_blocks, homogenize_matrix_Rnosym


def matlab_fread(fid, nelements, dtype):
    """Equivalent to Matlab fread function"""

    if dtype is np.str_:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array


#Hamiltonians from binary files, contains all the OMEN block properties.
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
    
    
    def __init__(self,sim_folder, no_orb, Vappl = 0.0, bias_point = 0, potential_type = 'linear', layer_matrix = '/Layer_Matrix.dat', homogenize = False, NCpSC = 1, rank = 0):
        if(not rank):
            self.no_orb = no_orb
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

            #Read Block Properties
            self.LM = read_file_to_float_ndarray(sim_folder + layer_matrix)
            self.NA = self.LM.shape[0]
            self.NB = self.LM.shape[1] - 4
            self.TB = np.max(self.no_orb)
            self.Smin = read_file_to_int_ndarray(sim_folder + '/Smin_dat')
            self.Smin = self.Smin.reshape((self.Smin.shape[0], )).astype(int)
            self.prepare_block_properties()
            self.map_neighbor_indices()
            self.map_sparse_indices()
            if(homogenize):
                self.homogenize(NCpSC)
            if(potential_type == 'read_in_diag'):
                self.Vpot = np.loadtxt(self.sim_folder + 'Vpot_diag.npy')
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

        #Read the Header head[0]: Contains the format 0 if indexing starts at 0, 1 if indexing starts at 1
        #                head[1]: number of non-zero elements
        #                head[2]: matrix size
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
        if (ABmin.shape[0] > 5):
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