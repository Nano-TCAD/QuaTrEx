# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from pydantic import BaseModel


class SolverParameters(BaseModel):
    self_energy_stepping_factor: float = 0.5
    screened_interaction_stepping_factor: float = 0.1

    self_consistency_loop_max_iterations: int = 100
    check_convergence_every_n_iterations: int = 10
    current_convergence_threshold: float = 1e-2

    sancho_rubio_max_iterations: int = 100
    G_beyn_contour_integration_radius: float = 1000.0
    G_beyn_imaginary_limit: float = 5e-4
    Screened_interaction_beyn_contour_integration_radius: float = 1e6
    Screened_interaction_beyn_imaginary_limit: float = 1e-4


solver_parameters = SolverParameters()
