# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.



from quatrex.quatrex_interface import QuatrexInterface



if __name__ == "__main__":
    path_to_hamiltonian = "data/hamiltonian.bin"
    path_to_overlap_matrix = "data/overlap_matrix.bin"
    path_to_coulom_matrix = "data/coulomb_matrix.bin"
    path_to_parameters_file = "simulation_parameters.toml"
    path_to_atoms_positions = "data/atoms_positions.xyz"
    path_to_position_per_orbitals = "data/position_per_orbitals.yaml"
    
    quatrex = QuatrexInterface(path_to_hamiltonian, path_to_overlap_matrix, path_to_coulom_matrix, path_to_parameters_file, path_to_atoms_positions, path_to_position_per_orbitals)
    
    
