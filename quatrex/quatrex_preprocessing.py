# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.


from quatrex.simulation_parameters import SimulationParameters

from bsparse.bdia import BDIA
from bsparse.sparse import CSR as BCSR

import numpy as np
import scipy.sparse as sp
import os
import toml






class QuatrexPreprocessing:
    def __init__(
        self,
        path_to_parameters_file: str,
    ):
        self._simulation_parameters = self._read_parameters_from_file(path_to_parameters_file)
        
        csr_hamiltonian: sp.csr_matrix = self._read_binnary_matrix(self._simulation_parameters.path_to_hamiltonian)	
        
        """ bcsr_hamiltonian = BCSR.from_sparray(csr_hamiltonian)
        
        import matplotlib.pyplot as plt
        plt.plot(np.real(bcsr_hamiltonian.toarray()))
        plt.show() """
        
        
        """ import matplotlib.pyplot as plt
        plt.matshow(np.real(csr_hamiltonian.todense()))
        plt.show() """
        
        
        # mat_size = 3456, let's use 16 blocks as test -> blocksize = 216
        blocksize = 216
        
        blocksizes_array = [[blocksize for i in range(16)], [blocksize for i in range(16)]]
        
        self.bdia_hamiltonian = BDIA.from_sparray(csr_hamiltonian, blocksizes_array)
        #self.bdia_hamiltonian = bcsr_hamiltonian.todia()
        
        # csr_overlap_matrix: sp.csr_matrix = self._read_binnary_matrix(self._simulation_parameters.path_to_overlap_matrix)
        # self.bsparse_overlap_matrix: bsparse = bsparse(csr_overlap_matrix)
        
        # csr_coulomb_matrix: sp.csr_matrix = self._read_binnary_matrix(self._simulation_parameters.path_to_coulom_matrix)
        # self.bsparse_coulomb_matrix: bsparse = bsparse(csr_coulomb_matrix)
        
        # self.position_per_orbital: [list, list, list] = self._compute_position_per_orbitals_from_atoms_positions(self._simulation_parameters.path_to_atoms_positions, self._simulation_parameters.path_to_position_per_orbitals)
        # self.neighboring_matrix_indices: np.ndarray = self._compute_neighboring_matrix(position_per_orbital)
        
        # self.energy_array: np.ndarray = self._compute_energy_array(self._simulation_parameters.energy_grid)
        # self.fermi_levels: [float, float]  = self._compute_fermi_level(self._simulation_parameters.fermi_levels, self._simulation_parameters.applied_voltage)

    
    def get_hamiltonian(
        self
    ) -> BDIA:
        
        return self.bdia_hamiltonian
    
    
    def to_quatrex(
      self  
    ):
        pass
        
        
        """ return {
            self.bdia_hamiltonian,
            self.bsparse_overlap_matrix,
            self.bsparse_coulomb_matrix,
            self.neighboring_matrix_indices,
            self.energy_array,
            self.fermi_levels,
            self._simulation_parameters.conduction_band_energy,
            self._simulation_parameters.temperature,
            self._simulation_parameters.solver_mode
        } """
    
    
    def _compute_position_per_orbitals_from_atoms_positions(
        self,
        path_to_atoms_positions: str, 
        path_to_position_per_orbitals: str
    ) -> np.ndarray:
        pass
    
    
    def _compute_neighboring_matrix(
        self,
        position_per_orbital: np.ndarray
    ) -> np.ndarray:
        pass
    
    
    def _compute_energy_array(
        self,
        energy_grid: np.ndarray
    ) -> np.ndarray:
        pass
    
    
    def _compute_fermi_level(
        self,
        fermi_levels: np.ndarray,
        applied_voltage: float
    ) -> [float, float]:
        pass
    
    
    def _read_parameters_from_file(
        self,
        path_to_parameters_file: str
    ) -> SimulationParameters:
        
        with open(path_to_parameters_file, "r") as file:
            parameters_from_file = toml.load(file)
        
        simulation_parameters: SimulationParameters
        
        try:
            simulation_parameters = SimulationParameters(**parameters_from_file)
        except Exception as e:
            print(f'Validation error: {e}')
        else:
            return simulation_parameters
        
        
    def _read_binnary_matrix(
        self,
        path: os.PathLike
    ) -> sp.csr_matrix:
        """Parses an OMEN binary sparse matrix file.

        Parameters
        ----------
        path : str or Path
            Path to the binary sparse matrix from OMEN.

        Returns
        -------
        matrix : csr_matrix
            The matrix stored in the file.

        """
        with open(path, "rb") as f:
            bin = np.fromfile(f, dtype=np.double)

        dim, size, one_indexed = tuple(map(int, bin[:3]))
        data = np.reshape(bin[3:], (size, 4))
        row_inds, col_inds, real, imag = data.T

        if one_indexed:
            row_inds, col_inds = row_inds - 1, col_inds - 1

        matrix = sp.csr_matrix(
            (real + 1j * imag, (row_inds, col_inds)),
            shape=(dim, dim),
            dtype=np.complex128,
        )
        
        return matrix
        
        
        
    _simulation_parameters: SimulationParameters
    
    bdia_hamiltonian: BDIA
    """ bsparse_overlap_matrix: bsparse
    bsparse_coulomb_matrix: bsparse """
    
    position_per_orbital: [list, list, list]
    neighboring_matrix_indices: np.ndarray
    
    energy_array: np.ndarray
    fermi_levels: [float, float]