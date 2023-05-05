import sys
sys.path.insert(0, '../')
from GW.gold_solution import read_solution

import numpy as np


energy, rows, columns, gg_gold, gl_gold, gr_gold = read_solution.load_x("/home/vmaillou/Documents/Playground/data_GPWS.mat", "g")


# GG Matrix for energy n0
N_energy = 0
if(len(sys.argv) > 1 and int(sys.argv[-1]) > 0 and int(sys.argv[-1]) < len(energy)):
    N_energy = int(sys.argv[-1])

gg_0 = gg_gold[:, N_energy]
gl_0 = gl_gold[:, N_energy]
gr_0 = gr_gold[:, N_energy]

# Take only the real part of G
gg_real = np.real(gg_0)
gl_real = np.real(gl_0)
gr_real = np.real(gr_0)

# Take only the imaginary part of G
gg_imag = np.imag(gg_0)
gl_imag = np.imag(gl_0)
gr_imag = np.imag(gr_0)



# Show in 3D graphs the sparsity pattern of the Green's functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig.suptitle('Green\'s functions at energy: {}'.format(N_energy), fontsize=16)


# Plots real part
ax = fig.add_subplot(2,3,1, projection='3d')
ax.scatter(rows, columns, gg_real, c='r', marker='o')
ax.title.set_text('Gg_real')

ax = fig.add_subplot(2,3,2, projection='3d')
ax.scatter(rows, columns, gl_real, c='b', marker='o')
ax.title.set_text('Gl_real')

ax = fig.add_subplot(2,3,3, projection='3d')
ax.scatter(rows, columns, gr_real, c='g', marker='o')
ax.title.set_text('Gr_real')


# Plots imaginary part
ax = fig.add_subplot(2,3,4, projection='3d')
ax.scatter(rows, columns, gg_real, c='r', marker='o')
ax.title.set_text('Gg_imag')

ax = fig.add_subplot(2,3,5, projection='3d')
ax.scatter(rows, columns, gl_real, c='b', marker='o')
ax.title.set_text('Gl_imag')

ax = fig.add_subplot(2,3,6, projection='3d')
ax.scatter(rows, columns, gr_real, c='g', marker='o')
ax.title.set_text('Gr_imag')


plt.show()