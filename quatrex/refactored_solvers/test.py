
import sympy as sy
# Write down a sympy block matrix multiplication using symbolic variables and 3*3 matrices

from sympy import Matrix, symbols

# Define symbolic variables
zero = sy.MatrixSymbol('0', 3, 3)

#zero = sy.zeros(3, 3)

V_00 = sy.MatrixSymbol('V_{N-2,N-2}', 3, 3)
V_01 = sy.MatrixSymbol('V_{N-2,N-1}', 3, 3)
V_10 = sy.MatrixSymbol('V_{N-1,N-2}', 3, 3)

V_11 = sy.MatrixSymbol('V_{N-1,N-1}', 3, 3)
V_12 = sy.MatrixSymbol('V_{N-1,N}', 3, 3)
V_21 = sy.MatrixSymbol('V_{N,N-1}', 3, 3)

V_22 = sy.MatrixSymbol('V_{N,N}', 3, 3)
#V_23 = sy.MatrixSymbol('V_{23}', 3, 3)
#V_32 = sy.MatrixSymbol('V_{32}', 3, 3)

# Define 3x3 matrices
V = sy.BlockMatrix([[V_00, V_01,    zero,    zero,    zero], 
                    [V_10, V_11, V_12,    zero,    zero], 
                    [   zero, V_21, V_22, V_12,    zero],
                    [   zero,    zero, V_21, V_22, V_12],
                    [   zero,    zero,    zero, V_21, V_22]])

V_T = sy.BlockMatrix([[V_00, V_10,    zero,    zero,    zero], 
                      [V_01, V_11, V_21,    zero,    zero], 
                      [   zero, V_12, V_22, V_21,    zero],
                      [   zero,    zero, V_12, V_22, V_21],
                      [   zero,    zero,    zero, V_12, V_22]])


V2 = sy.BlockMatrix([[V_00, V_01,zero,    zero,    zero], 
                    [V_10, V_11, V_12,    zero,    zero], 
                    [   zero, V_21, V_22, zero,    zero],
                    [   zero,    zero,    zero, zero, zero],
                    [   zero,    zero,    zero, zero, zero]])

V2_T = sy.BlockMatrix([[V_00, V_10,    zero,    zero,    zero], 
                      [V_01, V_11, V_21,    zero,    zero], 
                      [zero, V_12, V_22, zero,    zero],
                      [zero, zero, zero, zero, zero],
                      [zero, zero,    zero, zero, zero]])


P_00 = sy.MatrixSymbol('P_{N-2,N-2}', 3, 3)
P_01 = sy.MatrixSymbol('P_{N-2,N-1}', 3, 3)
P_10 = sy.MatrixSymbol('P_{N-1,N-2}', 3, 3)

P_11 = sy.MatrixSymbol('P_{N-1,N-1}', 3, 3)
P_12 = sy.MatrixSymbol('P_{N-1,N}', 3, 3)
P_21 = sy.MatrixSymbol('P_{N,N-1}', 3, 3)

P_22 = sy.MatrixSymbol('P_{N,N}', 3, 3)

P = sy.BlockMatrix([[P_00, P_01,    zero,    zero,    zero], 
                    [P_10, P_11, P_12,    zero,    zero], 
                    [   zero, P_21, P_22, P_12,    zero],
                    [   zero,    zero, P_21, P_22, P_12],
                    [   zero,    zero,    zero, P_21, P_22]])

P2 = sy.BlockMatrix([[P_00, P_01,    zero,    zero,    zero], 
                    [P_10, P_11, P_12,    zero,    zero], 
                    [   zero, P_21, P_22, zero,    zero],
                    [   zero,    zero, zero, zero, zero],
                    [   zero,    zero,    zero, zero, zero]])

# Perform block matrix multiplication
Z = V @ P @ V_T - V2 @ P2 @ V2_T 

Z = sy.block_collapse(Z)

# Print the result
sy.print_latex(sy.simplify(Z.blocks[2, 2]))