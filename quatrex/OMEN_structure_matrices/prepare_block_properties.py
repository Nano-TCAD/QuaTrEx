#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:27:36 2023

@author: dleonard
"""
import numpy as np

def prepare_block_properties(no_orb, LM, Smin):
    """

    Parameters
    ----------
    no_orb : list of ints
        Number of Orbitals per atom type
    LM : OMEN Layer Matrix Format
        First three columns are atom coordinates, fourth column is atom type remaining columns are neighbor indices
    Smin : ndarray of int64
        First element is number of blocks, remaining elements are
        starting atom index of each block, last element indicates the end of last layer.

    Returns
    -------
    orb_per_at : ndarray of type uint
        Orbital starting index of each atom in structure
    Bmin,Bmax : ndarray of length NBlock (number of blocks) of type uint
        Arrays with start and end orbital indices of all blocks

    """
    NA = LM.shape[0]
    orb = np.array(no_orb)[LM[:,3].astype(np.uint)-1]
    
    orb_per_at = np.zeros((NA+1,), dtype = np.uint) #array with orbital index per atom
    orb_per_at[0] = 1
    
    for i in range(NA):
        orb_per_at[i+1] = orb_per_at[i] + orb[i]
        
    NBlock = Smin[0]
    Bmin = orb_per_at[Smin[1:-1]-1]
    Bmax = np.append(Bmin[1:]-1, np.max(orb_per_at)-1).astype(np.uint)

    return(NBlock, orb_per_at, Bmin, Bmax)