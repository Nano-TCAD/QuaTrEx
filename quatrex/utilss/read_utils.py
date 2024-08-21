#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 19:55:13 2023

@author: dleonard
"""

import numpy as np


def read_file_to_int_ndarray(filename, delimiter=" "):

    def is_int(string):
        """ True if given string is float else False"""
        try:
            return int(string)
        except ValueError:
            return False

    data = []
    with open(filename, 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(delimiter)
            data.append([int(i) if is_int(i) else i for i in k])
    data = np.array(data, dtype='int')

    return (data)


def read_file_to_float_ndarray(filename, delimiter=" "):

    def is_float(string):
        """ True if given string is float else False"""
        try:
            return float(string)
        except ValueError:
            return False

    data = []
    with open(filename, 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(delimiter)
            data.append([float(i) if is_float(i) else i for i in k])
    data = np.array(data, dtype='float64')

    return (data)

def read_lattice_dat(path):
    """
    Reads the lattice_dat file and returns the lattice vectors and the number of atoms

    Args:
        filename (str): Name of the file to be read

    Returns:
        lattice_vectors (np.ndarray): Lattice vectors
        num_atoms (int): Number of atoms
    """

    filename = path + '/lattice_dat'
    print(filename)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        lattice_vectors = np.array([list(map(float, line.split())) for line in lines[2:5]])
        num_atoms = int(lines[0].split()[0])

    return lattice_vectors*0.1, num_atoms  # Convert to nm
