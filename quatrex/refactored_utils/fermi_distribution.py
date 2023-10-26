# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from quatrex.refactored_utils.constants import e, k


def fermi_distribution(
    energy_in_eV : np.ndarray,
    fermi_energy_in_eV : float,
    temperature_in_kelvin : float
):
    return 1 / (1 + np.exp((energy_in_eV - fermi_energy_in_eV) * e
                           / (k * temperature_in_kelvin)))
