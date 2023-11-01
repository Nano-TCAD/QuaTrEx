# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from pydantic import BaseModel

class SolverParameters(BaseModel):
    self_energy_stepping_factor: float = 0.5
    screened_interaction_stepping_factor: float = 0.1
    self_consistency_loop_max_iterations: int = 100
    current_convergence_threshold: float = 1e-2

    sancho_rubio_max_iterations: int = 100
    beyn_contour_integration_radius: float = 1000.0
    beyn_imaginary_limit: float = 5e-4


solver_parameters = SolverParameters()
