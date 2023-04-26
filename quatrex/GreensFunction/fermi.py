#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:18:06 2023

@author: dleonard
"""
import numpy as np

def fermi_function(E, Ef, UT):
    return 1 / (1 + np.exp((E - Ef)/UT))