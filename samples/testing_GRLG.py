
import numpy as np
import numpy.typing as npt
import typing
import sys
import os
import numba

import matplotlib.pyplot as plt

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)

from quatrex.utils import linalg_cpu

gr = np.loadtxt("gr.dat").view(np.complex128).reshape((-1,))
gl = np.loadtxt("gl.dat").view(np.complex128).reshape((-1,))
gg = np.loadtxt("gg.dat").view(np.complex128).reshape((-1,))
energy = np.loadtxt("energy.dat")

# Using the principal value integral method for yet another sigma_r
NE = len(energy)
dE = energy[1] - energy[0]
Evec = np.linspace(0, (NE-1)*dE, NE, endpoint = True, dtype = float)

one_div_by_E = np.concatenate((-1.0/(Evec[-1:0:-1]), np.array([0.0], dtype = float), 1/(Evec[1:]), np.array([1/(Evec[-1] + dE)], dtype = float)))
one_div_by_E_t = np.fft.fft(one_div_by_E)


D1 = np.fft.fft(1j*np.imag(gg-gl), 2*NE)
gr_t_principale = np.multiply(D1, one_div_by_E_t)

gr_principale = 1j / (2*np.pi) * np.fft.ifft(gr_t_principale)[NE:] * dE

plt.figure(0, figsize=(20,10))
plt.plot(np.real(gr), label = 'reference real gr', linewidth = 4, linestyle = '--')
plt.plot(np.real(gr_principale), label = 'principal value real gr', linestyle = ':')
#plt.xlim((5000, 12500))
plt.ylim((-1, 1))
plt.legend()
plt.savefig('gr_comparison_test.png')

print('done')