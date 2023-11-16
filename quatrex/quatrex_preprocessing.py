# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.


from quatrex.quatrex_solver import QuatrexSolver
from quatrex.simulation_parameters import SimulationParameters

import bsparse

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel
import toml



class QuatrexPreprocessing:
    def __init__(
        self,
        path_to_parameters_file: str,
    ):
        self._simulation_parameters = self._read_parameters_from_file(path_to_parameters_file)
        
        print(self._simulation_parameters)
        
        # csr_hamiltonian: sp.csr_matrix = self._read_binnary_matrix(self._simulation_parameters.path_to_hamiltonian)	
        # bsparse_hamiltonian: bsparse = bsparse(csr_hamiltonian)
        
        # csr_overlap_matrix: sp.csr_matrix = self._read_binnary_matrix(self._simulation_parameters.path_to_overlap_matrix)
        # bsparse_overlap_matrix: bsparse = bsparse(csr_overlap_matrix)
        
        # csr_coulomb_matrix: sp.csr_matrix = self._read_binnary_matrix(self._simulation_parameters.path_to_coulom_matrix)
        # bsparse_coulomb_matrix: bsparse = bsparse(csr_coulomb_matrix)
        
        # position_per_orbital: [list, list, list] = self._compute_position_per_orbitals_from_atoms_positions(self._simulation_parameters.path_to_atoms_positions, self._simulation_parameters.path_to_position_per_orbitals)
        # Neighboring_matrix_indices: np.ndarray = self._compute_neighboring_matrix(position_per_orbital)
        
        # energy_array: np.ndarray = self._compute_energy_array(self._simulation_parameters.energy_grid)
        # fermi_levels: [float, float]  = self._compute_fermi_level(self._simulation_parameters.fermi_levels, self._simulation_parameters.applied_voltage)
        
        # quatrex = Quatrex_solver(bsparse_hamiltonian, 
        #                          bsparse_overlap_matrix, 
        #                          bsparse_coulomb_matrix, 
        #                          Neighboring_matrix_indices, 
        #                          energy_array, 
        #                          fermi_levels, 
        #                          self._simulation_parameters.conduction_band_edge, 
        #                          self._simulation_parameters.temperature, 
        #                          self._simulation_parameters.solver_mode)

        
    def solve(
        self
    ) -> int:
        """ Start the self-consistency solver.

        Returns:
            int: status returned by the solver.
        """
        pass
        #status = QuatrexSolver.self_consistency_solve()
        
        #return status
    
    
    def get_observables(
        self
    ):
        pass
    
        
    # Private methods    
    def _read_binnary_matrix(
        self,
        path_to_matrix: str
    ) -> sp.csr_matrix:
        pass

    
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
        
        
        
    _simulation_parameters: SimulationParameters