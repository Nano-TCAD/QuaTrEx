# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from quatrex.quatrex_preprocessing import QuatrexPreprocessing
from quatrex.quatrex_solver import QuatrexSolver

from bsparse.bdia import BDIA


if __name__ == "__main__":
    path_to_parameters_file = "simulation_parameters.toml"
    
    #quatrex = QuatrexPreprocessing(path_to_parameters_file)
    
    quatrex_data = QuatrexPreprocessing(path_to_parameters_file)
    quatrex_data.to_file("quatrex_data.hdf5")
    
    bdia_hamiltonian = quatrex_data.get_hamiltonian()
    
    print(bdia_hamiltonian[0, 5])
    
    """ import matplotlib.pyplot as plt
    plt.plot(bdia_hamiltonian[0, 0])
    plt.show() """
    
    
    
    quatrex_solver = QuatrexSolver("quatrex_data/")
    
    
    #print(quatrex_data)
    
    #quatrex.solve()
    
    #observables = quatrex.get_observables()
    
