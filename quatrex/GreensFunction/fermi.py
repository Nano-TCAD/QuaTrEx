# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np


def fermi_function(E, Ef, UT):
    return 1 / (1 + np.exp((E - Ef) / UT))
