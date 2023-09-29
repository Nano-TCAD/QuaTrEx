# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from quatrex.refactored_solvers import greens_function_solver
import numpy as np

    
class Solver:
    
    def __init__(
        self,
        energy_range : [float, float],
        number_of_energy_point : int,
        current_convergence_threshold : float,
    ):
        # __init_greens_function_storage()
        # __compute_energy_array(energy_range, number_of_energy_point)
        
        pass
    
    # ----- Interface -----
    def ballistic_solve(
        self,
        Hamiltonian : np.ndarray,
        Overlap_matrix : np.ndarray,
    ):
        
        G_retarded, G_lesser, G_greater = greens_function_solver.greens_function_solver(Hamiltonian, Overlap_matrix, 0)
        
        # __compute_current()
        # __compute_electron_density()
        # __compute_hole_density()
        # __compute_density_of_states()
        
        # return status
    
    def self_consistence_solve(
        self,
        Hamiltonian : np.ndarray,
        Overlap_matrix : np.ndarray,
        max_iterations : int
    ):  
        pass
        
        # __init_memory()
        
        # if gw:
        #   Pre-computation related to GW
    
        # Self_energy : np.ndarray
    
        # current_iteration = 0
        # While(not __current_converged() or current_iteration >= max_iterations):
            # Gr, G<, G> = compute_greens_functions(Hamiltonian, Overlap_matrix, Self_energy)
            
            # if gw:
            #   if iteration == 0:  
            #       coulomb_potential: np.ndarray
            #       GW_self_energy : np.ndarray
            #       Screened_interactions : np.ndarray
            #
            #   Self_energy += compute_gw_self_energy(Gr, 
            #                                         G<, 
            #                                         G>, 
            #                                         Screened_interactions, 
            #                                         GW_self_energy,
            #                                         coulomb_potential,) 
            # if photon:
            #   Self_energy += compute_photon_self_energy()
            # if phonon:
            #   Self_energy += compute_phonon_self_energy()
            
        # __compute_current()
        # __compute_electron_density()
        # __compute_hole_density()
        # __compute_density_of_states()
        
        # return status
        
        
    # Getters for computed observables
    def get_current(
        self
    ) -> np.ndarray:
        
        current : np.ndarray
    
        # Compute current
    
        return current
    
    
    def get_electron_density(
        self
    ):
        pass    
    
    
    def get_hole_density(
        self
    ):
        pass
    
    
    def get_density_of_states(
        self
    ):
        pass
    
    
    
    # ----- Private methods -----
    def __set_communications(
        self
    ):
        pass
    
    def __compute_current(
        self
    ):
        pass
    
    def __compute_electron_density(
        self
    ):
        pass
    
    def __compute_hole_density(
        self
    ):
        pass
    
    def __compute_density_of_states(
        self
    ):
        pass
    
    def __current_converged(
        self,
        current_convergence_threshold : float
    ) -> bool:
        
        #current_left_contact : np.ndarray
        #current_right_contact : np.ndarray
        # for energy in energy_array:
            # Compute current_left_contact(energy)
            # Compute current_right_contact(energy)
        
        #total_current_left_contact(current_left_contact)
        #total_current_right_contact(current_right_contact)
        # if abs(total_current_left_contact - total_current_right_contact) < current_convergence_threshold:
            # return True
        # return False
        pass
    
    def __init_greens_function_storage(
        self
    ):
        pass
    
    def __compute_energy_array(
        self,
        energy_range : [float, float],
        number_of_energy_point : int
    ):
        # Compute energy_array
        # Log to user: energy resoltion
        pass
    
    
    # ----- Private attributes -----
    G_retarded : np.ndarray
    G_lesser : np.ndarray
    G_greater : np.ndarray
    
    energy_array : np.ndarray
    