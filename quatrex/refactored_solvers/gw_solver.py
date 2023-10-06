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

    # prepare_greens_function_for_polarisation(G_retarded, G_lesser, G_greater)
            
    # Polarisation = compute_polarisation(G_retarded, G_lesser, G_greater)
    
    # Symettrizing polarization
        
    # Screened_interactions = compute_screened_interactions(Polarisation, Screened_interactions)
        
    # prepare_greens_function_for_gw_self_energy(Screened_interactions)

    # GW_self_energy = compute_gw_self_energy(Screened_interactions, GW_self_energy)
    
    #return GW_self_energy
    
    pass 
    
    
def prepare_greens_function_for_polarisation(
    G_retarded,
    G_lesser,
    G_greater
):
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


def prepare_greens_function_for_gw_self_energy(
    Screened_interactions
):
    pass


def compute_gw_self_energy(
    Screened_interactions,
    GW_self_energy
):
    pass   