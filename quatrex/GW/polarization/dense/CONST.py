"""Class to save all used physical constants

Raises:
    ValueError: If tried to change a constant
    TypeError: All constants named in upper case
"""
# Copyright Runsheng Ouyang, 2023

from scipy import constants
import copy
import sys

class CONST(object):
    '''
    DEFAULT
        energy  eV
        mass    eV s^2 / Å^2
        length  Å
        time    s
        temp    K
        charge  e

    how to use
        from Polarization.Dense import CONST
        pi = CONST.PI
    '''
    
    def __init__(self):
        self.PI = copy.deepcopy(constants.pi)
        self.HBAR = copy.deepcopy(
            constants.hbar/constants.eV
        ) # 6.582119569509067e-16 eV s
        self.K = copy.deepcopy(
            constants.Boltzmann/constants.eV
        ) # 8.617333262145179e-05 eV K^-1
        self.M_E = copy.deepcopy(
            constants.electron_mass*1e-20/constants.eV
        ) # 5.685630103565723e-32 eV s^2 Å^-2, hbar^2/m_e^2 ~ O(1)
        self.E = 1.0 # elementary charge
        self.EPS0 = copy.deepcopy(
            constants.epsilon_0*1e-10/constants.elementary_charge
        ) # 5.526349358057110e-03 e^2 eV^-1 Å^-1
        

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ValueError('CONST can not be changed!')
        if not name.isupper():
            raise TypeError('CONST should be all UPPER_CASE!')
        self.__dict__[name] = value

sys.modules[__name__] = CONST() # create an instance when calling this module
