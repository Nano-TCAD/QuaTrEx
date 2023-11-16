# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from quatrex.quatrex_preprocessing import QuatrexPreprocessing


if __name__ == "__main__":
    path_to_parameters_file = "simulation_parameters.toml"
    
    quatrex = QuatrexPreprocessing(path_to_parameters_file)
    
    quatrex.solve()
    
    observables = quatrex.get_observables()
    
