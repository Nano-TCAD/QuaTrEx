#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:58:38 2023

@author: dleonard
"""
import numpy as np
from scipy import sparse

def read_hamiltonian(Path):
    """
    
    Parameters
    ----------
    Path : String
        Path to the binary files

    Returns
    -------
    H : scipy sparse CSR matrix of float64 with size tot_orbs x tot_orbs
        Hamiltonian Matrix of the structure in sparse format
    S : scipy sparse CSR matrix of float64 with size tot_orbs x tot_orbs
        Overlap Matrix in case of non-orthogonal basis set

    """
    H = sparse.random(3120, 3120, 0.5).tocsr()
    S = sparse.random(3120, 3120, 0.01).tocsr()
    return(H, S)