import cupy as cp
from cupy.linalg import svd



cp.random.seed(0)

def beyn(M00, M01, M10, imag_lim, R, type_side):
    
    #cp.seterr(divide='ignore', invalid='ignore')

    theta_min = 0
    theta_max = 2*cp.pi
    NT = 51
    eps_lim = 1e-8
    ref_iteration = 2                                   
    cond = 0
    min_dEk = 1e8

    N = M00.shape[0]

    theta = cp.linspace(theta_min, theta_max, NT)
    dtheta = cp.hstack((theta[1]-theta[0], theta[2:]-theta[:-2], theta[-1]-theta[-2]))/2

    c = 0
    r1 = 3.0
    r2 = 1/R

    zC1 = c + r1*cp.exp(1j*theta)
    zC2 = c + r2*cp.exp(1j*theta)
    z = cp.hstack((zC1, zC2))

    dzC1_dtheta = 1j*r1*cp.exp(1j*theta)
    dzC2_dtheta = 1j*r2*cp.exp(1j*theta)
    dz_dtheta = cp.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta = cp.hstack((dtheta, dtheta))

    if N < 100:
        NM = round(3*N/4)
    else:
        NM = round(N/2)

    Y = cp.random.rand(N, NM)
    #Y = cp.loadtxt('CNT_newwannier/' + 'ymatrix.dat')

    P0 = cp.zeros((N, N), dtype=cp.complex128)
    P1 = cp.zeros((N, N), dtype=cp.complex128)

    for I in range(len(z)):

        if type_side == 'L':
            T = M00 + M01/z[I] + M10*z[I]
        else:
            T = M00 + M01*z[I] + M10/z[I]

        iT = cp.linalg.inv(T)

        P0 += iT*dz_dtheta[I]*dtheta[I]/(2*cp.pi*1j)
        P1 += iT*z[I]*dz_dtheta[I]*dtheta[I]/(2*cp.pi*1j)

    LP0 = P0@Y
    LP1 = P1@Y

    RP0 = Y.T@P0
    RP1 = Y.T@P1

    LV, LS, LW = svd(LP0, full_matrices=False)
    Lind = cp.where(abs(cp.diag(LS)) > eps_lim)[0]

    RV, RS, RW = svd(RP0, full_matrices=True)
    Rind = cp.where(abs(cp.diag(RS)) > eps_lim)[0]

    if len(Lind) == 0:

        cond = cp.nan
        ksurf = None
        Sigma = None
        gR = None

    else:

        LV = LV[:, Lind]
        LS = LS[Lind]
        LW = cp.conj(LW).T[:, Lind]
        # eig does not exist in cupy
        Llambda, Lu = cp.linalg.eig(cp.conj(LV).T@LP1@LW@cp.linalg.inv(cp.diag(LS)))
        #Llambda = cp.diag(Llambda)
        phiL = LV@Lu

        RV = RV[:, Rind]
        RS = RS[Rind]
        RW = cp.conj(RW).T[:, Rind]
        # eig does not exist in cupy
        Rlambda, Ru = cp.linalg.eig(cp.linalg.inv(cp.diag(RS))@cp.conj(RV).T@RP1@RW)
        #Rlambda = cp.diag(Rlambda)
        phiR = cp.linalg.solve(Ru, cp.conj(RW).T)

        if type_side == 'L':
            kL = 1j * cp.log(Llambda)
            kR = 1j * cp.log(Rlambda)
        else:
            kL = -1j * cp.log(Llambda)
            kR = -1j * cp.log(Rlambda)

        ind_sort_kL = cp.argsort(abs(cp.imag(kL)))
        k = kL[ind_sort_kL]
        phiL = phiL[:, ind_sort_kL]

        if type_side == 'L':
            ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, 1.0)
            gR = Vsurf @ cp.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M10 @ Vsurf @ cp.diag(cp.exp(-1j * ksurf))) @ Vsurf.T
            for IC in range(ref_iteration):
                gR = cp.linalg.inv(M00 - M10 @ gR @ M01)
            Sigma = M10 @ gR @ M01
        else:
            ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, -1.0)
            gR = Vsurf @ cp.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M01 @ Vsurf @ cp.diag(cp.exp(1j * ksurf))) @ Vsurf.T
            for IC in range(ref_iteration):
                gR = cp.linalg.inv(M00 - M01 @ gR @ M10)
            Sigma = M01 @ gR @ M10

        ind = cp.where(abs(dEk_dk))
        if len(ind[0]) > 0:
            min_dEk = cp.min(abs(dEk_dk[ind]))
    return ksurf, cond, gR, Sigma, min_dEk

def check_imag_cond(k, kR, phiR, phiL, M10, M01, max_imag):
    imag_cond = cp.zeros(len(k))
    dEk_dk = cp.zeros(len(k), dtype =  cp.cfloat)

    ind = cp.where(cp.abs(cp.imag(k)) < cp.max(cp.array([0.5, max_imag])))[0]
    Ikmax = len(ind)

    if Ikmax % 2 == 1:
        Ikmax += 1

    for Ik in range(Ikmax):
        ind_kR = cp.argmin(cp.abs(cp.ones(len(kR)) * k[Ik] - kR))

        dEk_dk[Ik] = -(phiR[ind_kR, :] @ (-1j * M10 * cp.exp(-1j * k[Ik]) + 1j * M01 * cp.exp(1j * k[Ik])) @ phiL[:, Ik]) / \
            (phiR[ind_kR, :] @ phiL[:, Ik])

    for Ik in range(Ikmax):
        if not imag_cond[Ik]:
            ind_neigh = cp.argmin(cp.abs(dEk_dk + dEk_dk[Ik] * cp.ones(len(k))))

            k1 = k[Ik]
            k2 = k[ind_neigh]
            dEk1 = dEk_dk[Ik]
            dEk2 = dEk_dk[ind_neigh]

            cond1 = cp.abs(dEk1 + dEk2) / (cp.abs(dEk1) + 1e-10) < 0.25
            cond2 = cp.abs(k1 + k2) / (cp.abs(k1) + 1e-10) < 0.25
            cond3 = (cp.abs(cp.imag(k1)) + cp.abs(cp.imag(k2))) / 2.0 < 1.5 * max_imag
            cond4 = cp.sign(cp.imag(k1)) == cp.sign(cp.imag(k2))

            if cond1 and cond2 and (cond3 or (not cond3 and cond4)):
                if not imag_cond[ind_neigh]:
                    imag_cond[Ik] = 1
                    imag_cond[ind_neigh] = 1

    return imag_cond, dEk_dk



def sort_k(k, kR, phiL, phiR, M01, M10, imag_limit, factor):
    Nk = len(k)
    NT = len(phiL[:,1])

    ksurf = cp.zeros(Nk, dtype = cp.cfloat)
    Vsurf = cp.zeros((NT, Nk), dtype = cp.cfloat)

    imag_cond, dEk_dk = check_imag_cond(k, kR, phiR, phiL, M10, M01, imag_limit)

    Nref = 0

    for Ik in range(Nk):

        if not imag_cond[Ik] and abs(cp.imag(k[Ik])) > 1e-6:

            if factor*cp.imag(k[Ik]) < 0:

                Nref += 1

                ksurf[Nref-1] = k[Ik]
                Vsurf[:,Nref-1] = phiL[:,Ik]

        else:
        
            cond1 = (abs(cp.real(dEk_dk[Ik])) < abs(cp.imag(dEk_dk[Ik]))/100) and (factor*cp.imag(k[Ik]) < 0)
            cond2 = (abs(cp.real(dEk_dk[Ik])) >= abs(cp.imag(dEk_dk[Ik]))/100) and (factor*cp.real(dEk_dk[Ik]) < 0)

            if cond1 or cond2:

                Nref += 1

                ksurf[Nref-1] = k[Ik]
                Vsurf[:,Nref-1] = phiL[:,Ik]
    ksurf = ksurf[0:Nref]
    Vsurf = Vsurf[:,0:Nref]
    return ksurf, Vsurf, dEk_dk