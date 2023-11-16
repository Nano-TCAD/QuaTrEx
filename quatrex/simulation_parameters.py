# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from pydantic import BaseModel
import toml

class SimulationParameters(BaseModel):
    # Path to the system-matrices
    #   The system-matrices are given in complex-number float128 binary format.
    path_to_hamiltonian: str
    path_to_overlap_matrix: str
    path_to_coulom_matrix: str
    
    # Blocksize of the system-matrices
    blocksize: int
    
    # Path to the atoms positions and the position per orbitals matrices.
    #   The atoms positions and positions per orbitals are given in the xyz format. 
    #   If atoms positions are given, intern function will deduce the position per
    #   orbitals from the atoms positions.
    path_to_atoms_positions: str
    path_to_position_per_orbitals: str
    
    # Conduction band energy [eV]
    conduction_band_energy: float
    
    # Temperature in Kelvin [K]
    temperature: float
    
    # Solver solver
    #   - "gf" for Green's Function computation
    #   - "gw" for electron-electron GW computation
    solver_mode: str
    
    # Energy grid
    #   The energy array will be computed using the left energy range and the
    #   energy step. Stepping up to the right energy range.
    energy_grid: dict
    
    # Fermi levels
    #  Data validation will be performed to ensure that the given fermi levels
    #  and applied voltage are consistent with each other. If one is set to 
    # "None", it will be computed from the other two.
    fermi_levels: dict
    
    
    