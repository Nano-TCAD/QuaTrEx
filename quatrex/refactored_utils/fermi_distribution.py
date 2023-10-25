# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np

def fermi_distribution(
    energy_in_eV : np.ndarray,
    fermi_energy_in_eV : float,
    elementary_charge : float,
    boltzmann_constant : float,
    temperature_in_kelvin : float
):
    return 1 / (1 + np.exp((energy_in_eV - fermi_energy_in_eV) * elementary_charge
                           / (boltzmann_constant * temperature_in_kelvin)))
