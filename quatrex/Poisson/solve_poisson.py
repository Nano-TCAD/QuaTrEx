import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csc_array, csc_matrix

import time

import cupy as cp
from cupyx.scipy.sparse.linalg import spilu as cupy_spilu
from cupyx.scipy.sparse.linalg import cg as cupy_cg
from cupyx.scipy.sparse.linalg import LinearOperator as cupy_LinearOperator

from mpi4py import MPI

def solve_poisson(DH, n_at, p_at, dn_at, dp_at, Temp, comm, rank):

    if rank == 0:
        q = 1.6022e-19
        kB = 1.38e-23
        Eps0 = 8.854e-12

        gate_index = DH.poisson_gate_index
        Vg = DH.poisson_Vg
        P = DH.stiffness_mat
        atom_index = DH.poisson_atom_index.reshape((DH.poisson_atom_index.shape[0],))
        grid_index = DH.grid_index
        Vold = DH.Vpoiss
        doping = DH.poisson_doping

        drho = 1e-8
        criterion = 1e-3
        min_tol = 1e-6
        max_iter = 500
        beta = (q * 1.0e9) / Eps0

        NG = P.shape[1]
        NP = P.shape[0]

        # Vold = Vold.reshape((NG, 1))
        # doping = doping.reshape((NG, 1))

        rho_at = (n_at - p_at).T.astype(float)
        srho_at = np.sign(rho_at).astype(float)
        NC = np.max(np.abs(rho_at))

        #Vnew = Vold.copy()
        Vnew = np.copy(Vold)
        Vnew[gate_index] = Vg

        Ef = Vnew[atom_index] + srho_at * kB * Temp / q * np.log(np.exp((rho_at + srho_at * drho) / (srho_at * NC)) - 1)

        condition = np.inf
        cond_0 = np.inf
        ICmax = 20

        rho = np.zeros(NG)
        drho_dV = np.zeros(NG)

        RP = P[:, grid_index]

        IC = 0
        while condition > criterion and IC < ICmax:

            rho[atom_index] = srho_at * NC * np.log(np.exp(srho_at * (Ef - Vnew[atom_index]) / (kB * Temp / q)) + 1)
            drho_dV[atom_index] = -NC * q / (kB * Temp) / (np.exp(srho_at * (Vnew[atom_index] - Ef) / (kB * Temp / q)) + 1)

            res = P @ Vnew + beta * (doping[grid_index] - rho[grid_index])

            diag_indices = np.arange(NP)
            RP[diag_indices, diag_indices] -= beta * drho_dV[grid_index]

            problem_size = RP.shape[0]
            rows_loc = np.arange(problem_size)
            cols_loc = np.arange(problem_size)

            
            #RP_GPU = cp.sparse.csr_matrix(RP)
            #res_gpu = cp.asarray(res)

            # ilu_decomp = cupy_spilu(RP_GPU, drop_tol=0.01)
            # M = cupy_LinearOperator(RP_GPU.shape, ilu_decomp.solve)
            # delta_V_GPU, _ = cupy_cg(RP_GPU, res_gpu, tol=min_tol, maxiter=max_iter, M=M)
            # delta_V = cp.asnumpy(delta_V_GPU)

            middle_b = time.perf_counter()

            ilu_a = time.perf_counter()

            # Incomplete LU factorization
            # ilu_decomp = spilu(RP, drop_tol=0.01)
            diagonal_e = (1/ np.array(RP[rows_loc, cols_loc])).reshape((-1,))
            M = csc_array((diagonal_e, (rows_loc, cols_loc)))
            #M_GPU = cp.sparse.csc_matrix(M)

            ilu_b = time.perf_counter()
            print(f'  Time ILU: {IC + 1}: {ilu_b - ilu_a}')

            operator_a = time.perf_counter()

            # Prepare linear operator
            #M = LinearOperator(RP.shape, ilu_decomp.solve)


            operator_b = time.perf_counter()
            print(f'  Time operator: {IC + 1}: {operator_b - operator_a}')

            conjugate_a = time.perf_counter()

            # Solve linear system with BIConjugate Gradient STABilized
            delta_V, _ = bicgstab(RP, res, tol=min_tol, maxiter=max_iter, M=M)
            #delta_V_GPU, _ = cupy_cg(RP_GPU, res_gpu, tol=min_tol, maxiter=max_iter, M=M_GPU)


            conjugate_b = time.perf_counter()
            print(f'  Time conjugate: {IC + 1}: {conjugate_b - conjugate_a}')

            middle_a = time.perf_counter()

            print(f'  Time solve: {IC + 1}: {middle_a - middle_b}')


            Vnew[grid_index] -= delta_V

            condition = np.max(np.abs(delta_V))

            print(f'Poisson Inner Loop : {IC+1}: {condition}')

            if IC == 0:
                cond_0 = condition

            IC += 1

        comm.Bcast([Vnew, MPI.DOUBLE], root=0)
    else:
        bcast_size = DH.Vpoiss.shape[0]
        Vnew = np.zeros((bcast_size,), dtype=float)
        cond_0 = np.inf
        comm.Bcast([Vnew, MPI.DOUBLE], root=0)
    DH.Vpoiss = Vnew

    return Vnew, cond_0


def solve_poisson_gpu(DH, n_at, p_at, dn_at, dp_at, Temp, comm, rank):

    if rank == 0:
        q = 1.6022e-19
        kB = 1.38e-23
        Eps0 = 8.854e-12

        gate_index = DH.poisson_gate_index
        Vg = DH.poisson_Vg
        P = DH.stiffness_mat
        atom_index = DH.poisson_atom_index.reshape((DH.poisson_atom_index.shape[0],))
        grid_index = DH.grid_index
        Vold = DH.Vpoiss
        doping = DH.poisson_doping

        drho = 1e-8
        criterion = 1e-3
        min_tol = 1e-6
        max_iter = 500
        beta = (q * 1.0e9) / Eps0

        NG = P.shape[1]
        NP = P.shape[0]

        # Vold = Vold.reshape((NG, 1))
        # doping = doping.reshape((NG, 1))

        rho_at = (n_at - p_at).T.astype(float)
        srho_at = np.sign(rho_at).astype(float)
        srho_at_GPU = cp.asarray(srho_at)
        NC = np.max(np.abs(rho_at))

        #Vnew = Vold.copy()
        Vnew = np.copy(Vold)
        Vnew[gate_index] = Vg

        Ef = Vnew[atom_index] + srho_at * kB * Temp / q * np.log(np.exp((rho_at + srho_at * drho) / (srho_at * NC)) - 1)
        Ef_GPU = cp.asarray(Ef)

        condition = np.inf
        cond_0 = np.inf
        ICmax = 20

        rho = np.zeros(NG)
        drho_dV = np.zeros(NG)

        RP = P[:, grid_index]

        grid_index_GPU = cp.asarray(grid_index)

        P_GPU = cp.sparse.csr_matrix(P)
        RP_GPU = P_GPU[:, grid_index_GPU]
        Vnew_GPU = cp.asarray(Vnew)
        doping_GPU = cp.asarray(doping)
        rho_GPU = cp.zeros(NG)
        drho_dV_GPU = cp.zeros(NG)
        atom_index_GPU = cp.asarray(atom_index)


        IC = 0
        while condition > criterion and IC < ICmax:

            evalrho_a = time.perf_counter()

            rho_GPU[atom_index_GPU] = srho_at_GPU * NC * cp.log(cp.exp(srho_at_GPU * (Ef_GPU - Vnew_GPU[atom_index_GPU]) / (kB * Temp / q)) + 1)
            drho_dV_GPU[atom_index_GPU] = -NC * q / (kB * Temp) / (cp.exp(srho_at_GPU * (Vnew_GPU[atom_index_GPU] - Ef_GPU) / (kB * Temp / q)) + 1)

            evalrho_b = time.perf_counter()
            print(f'  Time evalrho: {IC + 1}: {evalrho_b - evalrho_a}')

            evalres_a = time.perf_counter()

            res = P_GPU @ Vnew_GPU + beta * (doping_GPU[grid_index_GPU] - rho_GPU[grid_index_GPU])

            evalres_b = time.perf_counter()
            print(f'  Time evalres: {IC + 1}: {evalres_b - evalres_a}')

            correctmat_a = time.perf_counter()

            # Double indexing of RP_GPU is very slow, therefore copy back to host. To-Do: Look into it.
            diag_indices = np.arange(NP)
            RP = RP_GPU.get()
            values = beta * drho_dV_GPU[grid_index_GPU].get()
            RP[diag_indices, diag_indices] -= values
            #RP_GPU[diag_indices, diag_indices] -= beta * drho_dV_GPU[grid_index_GPU]
            RP_GPU = cp.sparse.csr_matrix(RP)

            problem_size = RP_GPU.shape[0]
            rows_loc = cp.arange(problem_size)
            cols_loc = cp.arange(problem_size)

            correctmat_b = time.perf_counter()
            print(f'  Time correctmat: {IC + 1}: {correctmat_b - correctmat_a}')


            # ilu_decomp = cupy_spilu(RP_GPU, drop_tol=0.01)
            # M = cupy_LinearOperator(RP_GPU.shape, ilu_decomp.solve)
            # delta_V_GPU, _ = cupy_cg(RP_GPU, res_gpu, tol=min_tol, maxiter=max_iter, M=M)
            # delta_V = cp.asnumpy(delta_V_GPU)

            middle_b = time.perf_counter()

            ilu_a = time.perf_counter()

            # Incomplete LU factorization
            # ilu_decomp = spilu(RP, drop_tol=0.01)
            diagonal_e = (1/ cp.array(RP_GPU[rows_loc, cols_loc])).reshape((-1,))
            M_GPU = cp.sparse.csc_matrix((diagonal_e, (rows_loc, cols_loc)))
            #M_GPU = cp.sparse.csc_matrix(M)

            ilu_b = time.perf_counter()
            print(f'  Time ILU: {IC + 1}: {ilu_b - ilu_a}')

            operator_a = time.perf_counter()

            # Prepare linear operator
            #M = LinearOperator(RP.shape, ilu_decomp.solve)


            operator_b = time.perf_counter()
            print(f'  Time operator: {IC + 1}: {operator_b - operator_a}')

            conjugate_a = time.perf_counter()

            # Solve linear system with BIConjugate Gradient STABilized
            #delta_V, _ = bicgstab(RP, res, tol=min_tol, maxiter=max_iter, M=M)
            delta_V_GPU, _ = cupy_cg(RP_GPU, res, tol=min_tol, maxiter=max_iter, M=M_GPU)


            conjugate_b = time.perf_counter()
            print(f'  Time conjugate: {IC + 1}: {conjugate_b - conjugate_a}')

            middle_a = time.perf_counter()

            print(f'  Time solve: {IC + 1}: {middle_a - middle_b}')


            Vnew_GPU[grid_index_GPU] -= delta_V_GPU

            condition = cp.max(cp.abs(delta_V_GPU))

            print(f'Poisson Inner Loop : {IC+1}: {condition}')

            if IC == 0:
                cond_0 = condition

            IC += 1
        Vnew = cp.asnumpy(Vnew_GPU)
        comm.Bcast([Vnew, MPI.DOUBLE], root=0)
    else:
        bcast_size = DH.Vpoiss.shape[0]
        Vnew = np.zeros((bcast_size,), dtype=float)
        cond_0 = np.inf
        comm.Bcast([Vnew, MPI.DOUBLE], root=0)
    DH.Vpoiss = Vnew

    return Vnew, cond_0
