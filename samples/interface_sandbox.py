# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from quatrex.quatrex_preprocessing import QuatrexPreprocessing
from quatrex.quatrex_solver import QuatrexSolver


if __name__ == "__main__":
    path_to_parameters_file = "simulation_parameters.toml"
    
    #quatrex = QuatrexPreprocessing(path_to_parameters_file)
    
    quatrex_data = QuatrexPreprocessing(path_to_parameters_file)
    
    
    quatrex_data.to_quatrex()
    
    
    
    #quatrex_solver = QuatrexSolver(quatrex_data.to_quatrex())
    
    
    #print(quatrex_data)
    
    #quatrex.solve()
    
    #observables = quatrex.get_observables()
    
