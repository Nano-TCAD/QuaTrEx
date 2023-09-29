# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np



# gw_solver.py
def gw_solver(
    G_retarded,
    G_lesser,
    G_greater,
    Screened_interactions,
    GW_self_energy,
    coulomb_potential
):

    # Artifact Filtering need -> __compute_observables()
            
    # Polarisation = compute_polarisation(Gr, G<, G>)
        
    # Screened_interactions = compute_screened_interactions(Polarisation, Screened_interactions)
        
    # Artifact Filtering need -> __compute_observables()

    # GW_self_energy = compute_gw_self_energy(Screened_interactions, GW_self_energy)
    
    #return GW_self_energy
    
    pass 
    
    
def compute_polarisation(
    G_retarded,
    G_lesser,
    G_greater
):  
    pass    
    
    
def compute_screened_interactions(
    Polarisation
):
    pass   