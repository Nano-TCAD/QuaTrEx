# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import cupy as cp
import cupy_backends as cpb
import numpy as np

# compute_stream = cp.cuda.Stream(non_blocking=True)

# cublas_handle = cp.cuda.device.get_cublas_handle()
# cpb.cuda.libs.cublas.setStream(cublas_handle, compute_stream.ptr)
# rocblas_handle = cp.cuda.device.get_rocblas_handle()
# cpb.cuda.libs.rocblas.setStream(rocblas_handle, compute_stream.ptr)



def _rgf(ham_diag, ham_upper, ham_lower,  # Input Hamiltonian + Boundary Conditions
         sg_diag, sg_upper, sg_lower,  # Input Greater Self-Energies
         sl_diag, sl_upper, sl_lower,  # Input Lesser Self-Energies
         SigGBR, SigLBR,  # Input ???
         GR, GRnn1,  # Output Retarded Green's Functions (unused)
         GL, GLnn1,  # Output Lesser Green's Functions
         GG, GGnn1,  # Output Greater Green's Functions
         DOS, nE, nP, idE,  # Output Observables
         Bmin_fi, Bmax_fi  # Indices
        ):
    
    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    Bsize = max(Bmax - Bmin + 1)
    NB = len(Bmin)


# def _rgf_batched(ham_diag, ham_upper, ham_lower,  # Input Hamiltonian + Boundary Conditions
#                  sg_diag, sg_upper, sg_lower,  # Input Greater Self-Energies
#                  sl_diag, sl_upper, sl_lower,  # Input Lesser Self-Energies
#                  SigGBR, SigLBR,  # Input ???
#                  GR, GRnn1,  # Output Retarded Green's Functions (unused)
#                  GL, GLnn1,  # Output Lesser Green's Functions
#                  GG, GGnn1,  # Output Greater Green's Functions
#                  DOS, nE, nP, idE,  # Output Observables
#                  Bmin_fi, Bmax_fi  # Indices
#                 ):

#     # Sizes
#     # Why are subtracing by 1 every time? Fix 0-based indexing
#     Bmax = Bmax_fi - 1
#     Bmin = Bmin_fi - 1
#     Bsize = max(Bmax - Bmin + 1)
#     NB = len(Bmin)
#     energy_batchsize = ham_diag.shape[1]

#     ham_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     ham_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     ham_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sl_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sl_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sl_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sg_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sg_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sg_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)

#     gR_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Retarded (right)
#     gL_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser (right)
#     gG_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater (right)
#     SigLB_gpu = cp.empty((NB - 1, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser boundary self-energy
#     SigGB_gpu = cp.empty((NB - 1, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater boundary self-energy

#     ham_upper_H_gpu = cp.empty((energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     gR_H_gpu = cp.empty((energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    
#     # Backward pass

#     # First iteration
#     IB = NB - 1
#     idx = IB % 2
#     NN = Bmax[-1] - Bmin[-1] + 1
#     # print(f"IB: {IB}, idx: {idx}, NN: {NN}", flush=True)

#     hd = ham_diag_gpu[idx]
#     sld = sl_diag_gpu[idx]
#     sgd = sg_diag_gpu[idx]

#     gr = gR_gpu[IB]
#     grh = gR_H_gpu
#     gl = gL_gpu[IB]
#     gg = gG_gpu[IB]

#     hd.set(ham_diag[IB])
#     sld.set(sl_diag[IB])
#     sgd.set(sg_diag[IB])
    
#     gr[:, 0:NN, 0:NN] = cp.linalg.inv(hd[:, 0:NN, 0:NN])
#     # Here, potentially write gR back to host to save memory
#     cp.conjugate(gr[:, 0:NN, 0:NN].transpose((0,2,1)), out=grh[:, 0:NN, 0:NN])
#     cp.matmul(gr[:, 0:NN, 0:NN] @ sld[:, 0:NN, 0:NN], grh[:, 0:NN, 0:NN], out=gl[:, 0:NN, 0:NN])
#     cp.matmul(gr[:, 0:NN, 0:NN] @ sgd[:, 0:NN, 0:NN], grh[:, 0:NN, 0:NN], out=gg[:, 0:NN, 0:NN])

#     # Rest iterations
#     for IB in range(NB - 2, -1, -1):

#         nIB = IB - 1
#         idx = IB % 2
#         NI = Bmax[IB] - Bmin[IB] + 1
#         NP = Bmax[IB + 1] - Bmin[IB + 1] + 1
#         # print(f"IB: {IB}, idx: {idx}, pidx: {pidx}, NI: {NI}, NP: {NP}", flush=True)

#         hd = ham_diag_gpu[idx]
#         hu = ham_upper_gpu[idx]
#         hl = ham_lower_gpu[idx]
#         sld = sl_diag_gpu[idx]
#         sll = sl_lower_gpu[idx]
#         sgd = sg_diag_gpu[idx]
#         sgl = sg_lower_gpu[idx]

#         huh = ham_upper_H_gpu
#         gr = gR_gpu[IB]
#         pgr = gR_gpu[IB + 1]
#         grh = gR_H_gpu
#         gl = gL_gpu[IB]
#         pgl = gL_gpu[IB + 1]
#         gg = gG_gpu[IB]
#         pgg = gG_gpu[IB + 1]
#         slb = SigLB_gpu[IB]
#         sgb = SigGB_gpu[IB]


#         hd.set(ham_diag[IB])
#         hu.set(ham_upper[IB])
#         hl.set(ham_lower[IB])
#         sld.set(sl_diag[IB])
#         sll.set(sl_lower[IB])
#         sgd.set(sg_diag[IB])
#         sgl.set(sg_lower[IB])

#         cp.conjugate(hu[:, 0:NI, 0:NP].transpose((0,2,1)), out=huh[:, 0:NP, 0:NI])
#         gr[:, 0:NI, 0:NI] = cp.linalg.inv(hd[:, 0:NI, 0:NI] -
#                                         hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ hl[:, :NP, 0:NI])
#         cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)), out=grh[:, 0:NI, 0:NI])
#         al = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ sll[:, 0:NP, 0:NI]
#         slb[:, 0:NI, 0:NI] = hu[:, 0:NI, 0:NP] @ pgl[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI] - (al - cp.conjugate(al.transpose((0,2,1))))
#         gl[:, 0:NI, 0:NI] = gr[:, 0:NI, 0:NI] @ (sld[:, 0:NI, 0:NI] + slb[:, 0:NI, 0:NI]) @ grh[:, 0:NI, 0:NI]
#         ag = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ sgl[:, 0:NP, 0:NI]
#         sgb[:, 0:NI, 0:NI] = hu[:, 0:NI, 0:NP] @ pgg[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI] - (ag - cp.conjugate(ag.transpose((0,2,1))))
#         gg[:, 0:NI, 0:NI] = gr[:, 0:NI, 0:NI] @ (sgd[:, 0:NI, 0:NI] + sgb[:, 0:NI, 0:NI]) @ grh[:, 0:NI, 0:NI]

#     gR_gpu.get(out=GR)
#     gL_gpu.get(out=GL)
#     gG_gpu.get(out=GG)
#     cp.cuda.Stream.null.synchronize()


def _rgf_batched(ham_diag, ham_upper, ham_lower,  # Input Hamiltonian + Boundary Conditions
                 sg_diag, sg_upper, sg_lower,  # Input Greater Self-Energies
                 sl_diag, sl_upper, sl_lower,  # Input Lesser Self-Energies
                 SigGBR, SigLBR,  # Input ???
                 GR_host, GRnn1_host,  # Output Retarded Green's Functions (unused)
                 GL_host, GLnn1_host,  # Output Lesser Green's Functions
                 GG_host, GGnn1_host,  # Output Greater Green's Functions
                 DOS, nE, nP, idE,  # Output Observables
                 Bmin_fi, Bmax_fi  # Indices
                ):

    # Sizes
    # Why are subtracing by 1 every time? Fix 0-based indexing
    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    Bsize = max(Bmax - Bmin + 1)
    NB = len(Bmin)
    energy_batchsize = ham_diag.shape[1]

    ham_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    ham_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    ham_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    sl_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    sl_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    sl_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    sg_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    sg_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    sg_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)

    # streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    # default_stream = cp.cuda.Stream.null
    comp_stream = cp.cuda.Stream.null
    comm_stream = cp.cuda.Stream(non_blocking=True)
    out_stream = cp.cuda.Stream(non_blocking=True)
    events = [cp.cuda.Event() for _ in range(2)]
    comp_event = cp.cuda.Event()
    out_events = [cp.cuda.Event() for _ in range(2)]
    # out_diag_event = cp.cuda.Event()
    # out_offdiag_event = cp.cuda.Event()

    gR_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Retarded (right)
    gL_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser (right)
    gG_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater (right)
    SigLB_gpu = cp.empty((NB-1, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser boundary self-energy
    SigGB_gpu = cp.empty((NB-1, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater boundary self-energy
    DOS_gpu = cp.empty((energy_batchsize, NB), dtype=ham_diag.dtype)
    nE_gpu = cp.empty((energy_batchsize, NB), dtype=ham_diag.dtype)
    nP_gpu = cp.empty((energy_batchsize, NB), dtype=ham_diag.dtype)
    idE_gpu = cp.empty((energy_batchsize, NB), dtype=ham_diag.dtype)

    GR_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Retarded (right)
    GRnn1_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Retarded (right)
    GL_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser (right)
    GLnn1_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser (right)
    GG_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater (right)
    GGnn1_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater (right)

    ham_upper_H_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    gR_H_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    
    # Backward pass

    # First iteration
    IB = NB - 1
    nIB = IB - 1
    idx = IB % 2
    nidx = nIB % 2
    NN = Bmax[-1] - Bmin[-1] + 1
    # print(f"IB: {IB}, idx: {idx}, NN: {NN}", flush=True)

    hd = ham_diag_gpu[idx]
    sld = sl_diag_gpu[idx]
    sgd = sg_diag_gpu[idx]

    gr = gR_gpu[IB]
    grh = gR_H_gpu[idx]
    gl = gL_gpu[IB]
    gg = gG_gpu[IB]

    hd.set(ham_diag[IB])
    sld.set(sl_diag[IB])
    sgd.set(sg_diag[IB])

    if nIB >= 0:
        with comm_stream:

            nhd = ham_diag_gpu[nidx]
            nhu = ham_upper_gpu[nidx]
            nhl = ham_lower_gpu[nidx]
            nsld = sl_diag_gpu[nidx]
            nsll = sl_lower_gpu[nidx]
            nsgd = sg_diag_gpu[nidx]
            nsgl = sg_lower_gpu[nidx]

            nhd.set(ham_diag[nIB])
            nhu.set(ham_upper[nIB])
            nhl.set(ham_lower[nIB])
            nsld.set(sl_diag[nIB])
            nsll.set(sl_lower[nIB])
            nsgd.set(sg_diag[nIB])
            nsgl.set(sg_lower[nIB])
            events[idx].record(stream=comm_stream)
    
    gr[:, 0:NN, 0:NN] = cp.linalg.inv(hd[:, 0:NN, 0:NN])
    # Here, potentially write gR back to host to save memory
    cp.conjugate(gr[:, 0:NN, 0:NN].transpose((0,2,1)), out=grh[:, 0:NN, 0:NN])
    cp.matmul(gr[:, 0:NN, 0:NN] @ sld[:, 0:NN, 0:NN], grh[:, 0:NN, 0:NN], out=gl[:, 0:NN, 0:NN])
    cp.matmul(gr[:, 0:NN, 0:NN] @ sgd[:, 0:NN, 0:NN], grh[:, 0:NN, 0:NN], out=gg[:, 0:NN, 0:NN])

    # Rest iterations
    for IB in range(NB - 2, -1, -1):

        nIB = IB - 1
        idx = IB % 2
        nidx = nIB % 2
        pidx = (IB + 1) % 2
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1
        # print(f"IB: {IB}, idx: {idx}, pidx: {pidx}, NI: {NI}, NP: {NP}", flush=True)

        hd = ham_diag_gpu[idx]
        hu = ham_upper_gpu[idx]
        hl = ham_lower_gpu[idx]
        sld = sl_diag_gpu[idx]
        sll = sl_lower_gpu[idx]
        sgd = sg_diag_gpu[idx]
        sgl = sg_lower_gpu[idx]

        huh = ham_upper_H_gpu[idx]
        gr = gR_gpu[IB]
        pgr = gR_gpu[IB + 1]
        grh = gR_H_gpu[idx]
        gl = gL_gpu[IB]
        pgl = gL_gpu[IB + 1]
        gg = gG_gpu[IB]
        pgg = gG_gpu[IB + 1]
        slb = SigLB_gpu[IB]
        sgb = SigGB_gpu[IB]

        if IB == 0:
            gr = GR_gpu[0]
            gl = GL_gpu[0]
            gg = GG_gpu[0]

        with comm_stream:
                
            if nIB >= 0:

                nhd = ham_diag_gpu[nidx]
                nhu = ham_upper_gpu[nidx]
                nhl = ham_lower_gpu[nidx]
                nsld = sl_diag_gpu[nidx]
                nsll = sl_lower_gpu[nidx]
                nsgd = sg_diag_gpu[nidx]
                nsgl = sg_lower_gpu[nidx]

                nhd.set(ham_diag[nIB])
                nhu.set(ham_upper[nIB])
                nhl.set(ham_lower[nIB])
                nsld.set(sl_diag[nIB])
                nsll.set(sl_lower[nIB])
                nsgd.set(sg_diag[nIB])
                nsgl.set(sg_lower[nIB])
        
            else:  # nIB < 0

                nslu = sl_upper_gpu[idx]
                nsgu = sg_upper_gpu[idx]

                nslu.set(sl_upper[IB])
                nsgu.set(sg_upper[IB])

            events[idx].record(stream=comm_stream)

        comp_stream.wait_event(event=events[pidx])
        hupgr = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP]
        al = hupgr @ sll[:, 0:NP, 0:NI]
        ag = hupgr @ sgl[:, 0:NP, 0:NI]
        gr[:, 0:NI, 0:NI] = cp.linalg.inv(hd[:, 0:NI, 0:NI] - hupgr @ hl[:, :NP, 0:NI])
        cp.conjugate(hu[:, 0:NI, 0:NP].transpose((0,2,1)), out=huh[:, 0:NP, 0:NI])
        cp.subtract(hu[:, 0:NI, 0:NP] @ pgl[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI],
                    al - cp.conjugate(al.transpose((0,2,1))), out=slb[:, 0:NI, 0:NI])
        cp.subtract(hu[:, 0:NI, 0:NP] @ pgg[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI],
                    ag - cp.conjugate(ag.transpose((0,2,1))), out=sgb[:, 0:NI, 0:NI])
        # gr[:, 0:NI, 0:NI] = cp.linalg.inv(hd[:, 0:NI, 0:NI] - hupgr @ hl[:, :NP, 0:NI])
        cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)), out=grh[:, 0:NI, 0:NI])
        cp.matmul(gr[:, 0:NI, 0:NI] @ (sld[:, 0:NI, 0:NI] + slb[:, 0:NI, 0:NI]), grh[:, 0:NI, 0:NI], out=gl[:, 0:NI, 0:NI])
        cp.matmul(gr[:, 0:NI, 0:NI] @ (sgd[:, 0:NI, 0:NI] + sgb[:, 0:NI, 0:NI]), grh[:, 0:NI, 0:NI], out=gg[:, 0:NI, 0:NI])

        if IB == 0:
            comp_stream.synchronize()
            comp_event.record(stream=comp_stream)
            with out_stream:
                out_stream.wait_event(event=comp_event)
                # GR_gpu[0].get(out=GR_host[0])
                GL_gpu[0].get(out=GL_host[0])
                GG_gpu[0].get(out=GG_host[0])
            DOS_gpu[:, 0] = 1j * cp.trace(gr[:, 0:NI, 0:NI] - grh[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nE_gpu[:, 0] = -1j * cp.trace(gl[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nP_gpu[:, 0] = 1j * cp.trace(gg[:, 0:NI, 0:NI], axis1=1, axis2=2)
            idE_gpu[:, 0] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ gl[:, 0:NI, 0:NI] -
                                             gg[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))


    # gR_gpu.get(out=GR_host)
    # gL_gpu.get(out=GL_host)
    # gG_gpu.get(out=GG_host)
    # comp_stream.synchronize()
    # return

    # Forward pass

    # First iteration
    IB = 0
    nIB = IB + 1
    idx = IB % 2
    nidx = nIB % 2

    hu = ham_upper_gpu[idx]
    hl = ham_lower_gpu[idx]
    slu = sl_upper_gpu[idx]
    sgu = sg_upper_gpu[idx]

    GR = GR_gpu[IB]
    GRnn1 = GRnn1_gpu[idx]
    GL = GL_gpu[IB]
    GLnn1 = GLnn1_gpu[idx]
    GG = GG_gpu[IB]
    GGnn1 = GGnn1_gpu[idx]

    # NOTE: These were written directly to output in the last iteration of the backward pass
    gr = GR_gpu[IB]
    gl = GL_gpu[IB]
    gg = GG_gpu[IB]

    pgr = gR_gpu[nIB]
    pgl = gL_gpu[nIB]
    pgg = gG_gpu[nIB]
    grh = gR_H_gpu[idx]

    with comm_stream:
            
        if nIB < NB:

            nphu = ham_upper_gpu[nidx]
            nphl = ham_lower_gpu[nidx]
            npsll = sl_lower_gpu[nidx]
            npsgl = sg_lower_gpu[nidx]

            nphu.set(ham_upper[IB])
            nphl.set(ham_lower[IB])
            npsll.set(sl_lower[IB])
            npsgl.set(sg_lower[IB])

            if nIB < NB - 1:

                nhu = ham_diag_gpu[nidx]
                nhl = sl_diag_gpu[nidx]
                nslu = sl_upper_gpu[nidx]
                nsgu = sg_upper_gpu[nidx]

                nhu.set(ham_upper[nIB])
                nhl.set(ham_lower[nIB])
                nslu.set(sl_upper[nIB])
                nsgu.set(sg_upper[nIB])

            events[nidx].record(stream=comm_stream)

    comp_stream.wait_event(event=events[idx])
    cp.conjugate(pgr[:, 0:NP, 0:NP].transpose((0,2,1)), out=grh[:, 0:NP, 0:NP])
    hlh = cp.conjugate(hl[:, 0:NP, 0:NI].transpose((0,2,1)))
    grhu = gr[:, 0:NI, 0:NI] @ hu[:, 0:NI, 0:NP]
    hlhgrh = hlh @ grh[:, 0:NP, 0:NP]
    # NOTE: These were written in the last iteration of the backward pass
    # GR[:] = gr
    # GL[:] = gl
    # GG[:] = gg
    cp.negative(grhu @ pgr[:, 0:NP, 0:NP], out=GRnn1[:, 0:NI, 0:NP])
    cp.subtract(gr[:, 0:NI, 0:NI] @ slu[:, 0:NI, 0:NP] @ grh[:, 0:NP, 0:NP] - grhu @ pgl[:, 0:NP, 0:NP], gl[:, 0:NI, 0:NI] @ hlhgrh, out=GLnn1[:, 0:NI, 0:NP])
    cp.subtract(gr[:, 0:NI, 0:NI] @ sgu[:, 0:NI, 0:NP] @ grh[:, 0:NP, 0:NP] - grhu @ pgg[:, 0:NP, 0:NP], gg[:, 0:NI, 0:NI] @ hlhgrh, out=GGnn1[:, 0:NI, 0:NP])

    comp_stream.synchronize()
    comp_event.record(stream=comp_stream)
    with out_stream:
        out_stream.wait_event(event=comp_event)
        # GRnn1.get(out=GRnn1_host[0])
        GLnn1.get(out=GLnn1_host[0])
        GGnn1.get(out=GGnn1_host[0])
        out_events[idx].record(stream=out_stream)

    # GR_gpu.get(out=GR_host)
    # GL_gpu.get(out=GL_host)
    # GG_gpu.get(out=GG_host)
    # GRnn1_gpu.get(out=GRnn1_host)
    # GLnn1_gpu.get(out=GLnn1_host)
    # GGnn1_gpu.get(out=GGnn1_host)
    # comp_stream.synchronize()
    # return
    
    # TODO: Add observables

    # Rest iterations
    for IB in range(1, NB):

        pIB = IB - 1
        nIB = IB + 1
        idx = IB % 2
        pidx = pIB % 2
        nidx = nIB % 2
        NI = Bmax[IB] - Bmin[IB] + 1
        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        # print(f"IB: {IB}, idx: {idx}, pidx: {pidx}, NI: {NI}, NP: {NP}", flush=True)

        # hu = ham_upper_gpu[idx]
        # hl = ham_lower_gpu[idx]
        phu = ham_upper_gpu[pidx]
        phl = ham_lower_gpu[pidx]
        # sld = sl_diag_gpu[idx]
        # sll = sl_lower_gpu[idx]
        psll = sl_lower_gpu[pidx]
        # sgd = sg_diag_gpu[idx]
        # sgl = sg_lower_gpu[idx]
        psgl = sg_lower_gpu[pidx]

        GR = GR_gpu[idx]
        pGR = GR_gpu[pidx]
        GL = GL_gpu[idx]
        pGL = GL_gpu[pidx]
        GG = GG_gpu[idx]
        pGG = GG_gpu[pidx]

        gr = gR_gpu[IB]
        gl = gL_gpu[IB]
        gg = gG_gpu[IB]
        # slb = SigLB_gpu[IB]
        # sgb = SigGB_gpu[IB]

        with comm_stream:
                
            if nIB < NB:

                nphu = ham_upper_gpu[idx]
                nphl = ham_lower_gpu[idx]
                npsll = sl_lower_gpu[idx]
                npsgl = sg_lower_gpu[idx]

                # nphu.set(ham_upper[IB])
                # nphl.set(ham_lower[IB])
                # NOTE: These are already on the GPU from the previous iteration
                nphu[:] = ham_diag_gpu[idx]
                nphl[:] = sl_diag_gpu[idx]
                npsll.set(sl_lower[IB])
                npsgl.set(sg_lower[IB])


                if nIB < NB - 1:

                    nhu = ham_diag_gpu[nidx]
                    nhl = sl_diag_gpu[nidx]
                    nslu = sl_upper_gpu[nidx]
                    nsgu = sg_upper_gpu[nidx]

                    nhu.set(ham_upper[nIB])
                    nhl.set(ham_lower[nIB])
                    nslu.set(sl_upper[nIB])
                    nsgu.set(sg_upper[nIB])
                
                else:

                    sl_diag_gpu[nidx].set(SigLBR)
                    sg_diag_gpu[nidx].set(SigGBR)

            events[nidx].record(stream=comm_stream)

            # pGR.get(out=GR_host[pIB])
            # pGL.get(out=GL_host[pIB])
            # pGG.get(out=GG_host[pIB])

            # pGRnn1.get(out=GRnn1_host[pIB])
            # pGLnn1.get(out=GLnn1_host[pIB])
            # pGGnn1.get(out=GGnn1_host[pIB])

        comp_stream.wait_event(event=events[idx])
        pGRh = cp.conjugate(pGR[:, 0:NM, 0:NM].transpose((0,2,1)))
        phlh = cp.conjugate(phl[:, 0:NI, 0:NM].transpose((0,2,1)))
        grh = cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)))
        grphl = gr[:, 0:NI, 0:NI] @ phl[:, 0:NI, 0:NM]
        grphlpGRphu = grphl @ pGR[:, 0:NM, 0:NM] @ phu[:, 0:NM, 0:NI]
        phlhgrh = phlh @ grh
        pGRhphlhgrh = pGRh @ phlhgrh
        al = gr[:, 0:NI, 0:NI] @ psll[:, 0:NI, 0:NI] @ pGRhphlhgrh
        bl = grphlpGRphu @ gl[:, 0:NI, 0:NI]
        ag = gr[:, 0:NI, 0:NI] @ psgl[:, 0:NI, 0:NM] @ pGRhphlhgrh
        bg = grphlpGRphu @ gg[:, 0:NI, 0:NI]
        # comp_stream.synchronize()
        cp.add(gr[:, 0:NI, 0:NI], grphlpGRphu @ gr[:, 0:NI, 0:NI], out=GR[:, 0:NI, 0:NI])
        cp.subtract(gl[:, 0:NI, 0:NI] + grphl @ pGL[:, 0:NM, 0:NM] @ phlhgrh,
                    al - cp.conjugate(al.transpose((0,2,1))) - bl + cp.conjugate(bl.transpose((0,2,1))),
                    out=GL[:, 0:NI, 0:NI])
        cp.subtract(gg[:, 0:NI, 0:NI] + grphl @ pGG[:, 0:NM, 0:NM] @ phlhgrh,
                    ag - cp.conjugate(ag.transpose((0,2,1))) - bg + cp.conjugate(bg.transpose((0,2,1))),
                    out=GG[:, 0:NI, 0:NI])
    
        if IB < NB - 1:

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1 = GRnn1_gpu[idx]
            GLnn1 = GLnn1_gpu[idx]
            GGnn1 = GGnn1_gpu[idx]

            ngr = gR_gpu[nIB]
            ngl = gL_gpu[nIB]
            ngg = gG_gpu[nIB]

            hu = ham_diag_gpu[idx]
            hl = sl_diag_gpu[idx]
            slu = sl_upper_gpu[idx]
            sgu = sg_upper_gpu[idx]

            ngrh = cp.conjugate(ngr[:, 0:NP, 0:NP].transpose((0,2,1)))
            hlh = cp.conjugate(hl[:, 0:NP, 0:NI].transpose((0,2,1)))
            GRhu = GR[:, 0:NI, 0:NI] @ hu[:, 0:NI, 0:NP]
            hlhngrh = hlh @ ngrh
            # comp_stream.synchronize()
            # comp_stream.wait_event(event=out_events[pidx])
            cp.negative(GR[:, 0:NI, 0:NI] @ hu[:, 0:NI, 0:NP] @ ngr[:, 0:NP, 0:NP], out=GRnn1[:, 0:NI, 0:NP])
            cp.subtract(GR[:, 0:NI, 0:NI] @ slu[:, 0:NI, 0:NP] @ ngrh,
                        GRhu @ ngl[:, 0:NP, 0:NP] + GL[:, 0:NI, 0:NI] @ hlhngrh, out=GLnn1[:, 0:NI, 0:NP])
            cp.subtract(GR[:, 0:NI, 0:NI] @ sgu[:, 0:NI, 0:NP] @ ngrh,
                        GRhu @ ngg[:, 0:NP, 0:NP] + GG[:, 0:NI, 0:NI] @ hlhngrh, out=GGnn1[:, 0:NI, 0:NP])
        
            comp_stream.synchronize()
            comp_event.record(stream=comp_stream)
            with out_stream:
                out_stream.wait_event(event=comp_event)
                # GR.get(out=GR_host[IB])
                GL.get(out=GL_host[IB])
                GG.get(out=GG_host[IB])
                # GRnn1.get(out=GRnn1_host[IB])
                GLnn1.get(out=GLnn1_host[IB])
                GGnn1.get(out=GGnn1_host[IB])
                out_events[idx].record(stream=out_stream)

            slb = SigLB_gpu[IB]
            sgb = SigGB_gpu[IB]
            
            idE_gpu[:, IB] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ GL[:, 0:NI, 0:NI] -
                                              GG[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))
        
        DOS_gpu[:, IB] = 1j * cp.trace(GR[:, 0:NI, 0:NI] - cp.conjugate(GR[:, 0:NI, 0:NI].transpose(0,2,1)), axis1=1, axis2=2)
        nE_gpu[:, IB] = -1j * cp.trace(GL[:, 0:NI, 0:NI], axis1=1, axis2=2)
        nP_gpu[:, IB] = 1j * cp.trace(GG[:, 0:NI, 0:NI], axis1=1, axis2=2)

    nidx = (NB - 1) % 2
    slb = sl_diag_gpu[nidx]
    sgb = sg_diag_gpu[nidx]
    comp_stream.wait_event(event=events[nidx])
    idE_gpu[:, NB - 1] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ GL[:, 0:NI, 0:NI] -
                                          GG[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))

    GL.get(out=GL_host[-1])
    GG.get(out=GG_host[-1])
    DOS_gpu.get(out=DOS)
    nE_gpu.get(out=nE)
    nP_gpu.get(out=nP)
    idE_gpu.get(out=idE)
    out_stream.synchronize()
    comp_stream.synchronize()


# def _rgf_batched(ham_diag, ham_upper, ham_lower,  # Input Hamiltonian + Boundary Conditions
#                  sg_diag, sg_upper, sg_lower,  # Input Greater Self-Energies
#                  sl_diag, sl_upper, sl_lower,  # Input Lesser Self-Energies
#                  SigGBR, SigLBR,  # Input ???
#                  GR, GRnn1,  # Output Retarded Green's Functions (unused)
#                  GL, GLnn1,  # Output Lesser Green's Functions
#                  GG, GGnn1,  # Output Greater Green's Functions
#                  DOS, nE, nP, idE,  # Output Observables
#                  Bmin_fi, Bmax_fi  # Indices
#                 ):

#     # Sizes
#     # Why are subtracing by 1 every time? Fix 0-based indexing
#     Bmax = Bmax_fi - 1
#     Bmin = Bmin_fi - 1
#     Bsize = max(Bmax - Bmin + 1)
#     NB = len(Bmin)
#     energy_batchsize = ham_diag.shape[1]

#     ham_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     ham_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     ham_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sl_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sl_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sl_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sg_diag_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sg_upper_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     sg_lower_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)

#     streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
#     default_stream = cp.cuda.Stream.null
#     events = [cp.cuda.Event() for _ in range(2)]
#     inv_event = cp.cuda.Event()
#     # cublas_handle = cp.cuda.device.get_cublas_handle()
#     # cusolver_handle = cp.cuda.device.get_cusolver_handle()

#     gR_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Retarded (right)
#     gL_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser (right)
#     gG_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater (right)
#     SigLB_gpu = cp.empty((NB - 1, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser boundary self-energy
#     SigGB_gpu = cp.empty((NB - 1, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater boundary self-energy

#     ham_upper_H_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
#     gR_H_gpu = cp.empty((2, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    
#     # # Backward pass

#     # # First iteration
#     # IB = NB - 1
#     # nIB = IB - 1
#     # idx = IB % 2
#     # nidx = nIB % 2
#     # NN = Bmax[-1] - Bmin[-1] + 1
#     # # print(f"IB: {IB}, idx: {idx}, NN: {NN}", flush=True)

#     # with streams[idx] as current_stream:

#     #     hd = ham_diag_gpu[idx]
#     #     sld = sl_diag_gpu[idx]
#     #     sgd = sg_diag_gpu[idx]

#     #     hd.set(ham_diag[IB])
#     #     sld.set(sl_diag[IB])
#     #     sgd.set(sg_diag[IB])

#     #     events[idx].record(stream=current_stream)
    
#     # if nIB >= 0:

#     #     with streams[nidx] as current_stream:

#     #         hd = ham_diag_gpu[nidx]
#     #         hu = ham_upper_gpu[nidx]
#     #         hl = ham_lower_gpu[nidx]
#     #         sld = sl_diag_gpu[nidx]
#     #         sll = sl_lower_gpu[nidx]
#     #         sgd = sg_diag_gpu[nidx]
#     #         sgl = sg_lower_gpu[nidx]

#     #         hd.set(ham_diag[nIB])
#     #         hu.set(ham_upper[nIB])
#     #         hl.set(ham_lower[nIB])
#     #         sld.set(sl_diag[nIB])
#     #         sll.set(sl_lower[nIB])
#     #         sgd.set(sg_diag[nIB])
#     #         sgl.set(sg_lower[nIB])

#     #         events[nidx].record(stream=current_stream)
    
#     # hd = ham_diag_gpu[idx]
#     # sld = sl_diag_gpu[idx]
#     # sgd = sg_diag_gpu[idx]

#     # gr = gR_gpu[IB]
#     # grh = gR_H_gpu[idx]
#     # gl = gL_gpu[IB]
#     # gg = gG_gpu[IB]
    
#     # default_stream.wait_event(event=events[idx])

#     # gr[:, 0:NN, 0:NN] = cp.linalg.inv(hd[:, 0:NN, 0:NN])
#     # # Here, potentially write gR back to host to save memory
#     # cp.conjugate(gr[:, 0:NN, 0:NN].transpose((0,2,1)), out=grh[:, 0:NN, 0:NN])
#     # cp.matmul(gr[:, 0:NN, 0:NN] @ sld[:, 0:NN, 0:NN], grh[:, 0:NN, 0:NN], out=gl[:, 0:NN, 0:NN])
#     # cp.matmul(gr[:, 0:NN, 0:NN] @ sgd[:, 0:NN, 0:NN], grh[:, 0:NN, 0:NN], out=gg[:, 0:NN, 0:NN])

#     # # Rest iterations
#     # for IB in range(NB - 2, -1, -1):

#     #     nIB = IB - 1
#     #     idx = IB % 2
#     #     nidx = nIB % 2
#     #     pidx = (IB + 1) % 2
#     #     NI = Bmax[IB] - Bmin[IB] + 1
#     #     NP = Bmax[IB + 1] - Bmin[IB + 1] + 1
#     #     # print(f"IB: {IB}, idx: {idx}, pidx: {pidx}, NI: {NI}, NP: {NP}", flush=True)

#     #     if nIB >= 0:

#     #         with streams[nidx] as current_stream:

#     #             hd = ham_diag_gpu[nidx]
#     #             hu = ham_upper_gpu[nidx]
#     #             hl = ham_lower_gpu[nidx]
#     #             sld = sl_diag_gpu[nidx]
#     #             sll = sl_lower_gpu[nidx]
#     #             sgd = sg_diag_gpu[nidx]
#     #             sgl = sg_lower_gpu[nidx]

#     #             hd.set(ham_diag[nIB])
#     #             hu.set(ham_upper[nIB])
#     #             hl.set(ham_lower[nIB])
#     #             sld.set(sl_diag[nIB])
#     #             sll.set(sl_lower[nIB])
#     #             sgd.set(sg_diag[nIB])
#     #             sgl.set(sg_lower[nIB])

#     #             events[nidx].record(stream=current_stream)

                
#     #     hd = ham_diag_gpu[idx]
#     #     hu = ham_upper_gpu[idx]
#     #     hl = ham_lower_gpu[idx]
#     #     sld = sl_diag_gpu[idx]
#     #     sll = sl_lower_gpu[idx]
#     #     sgd = sg_diag_gpu[idx]
#     #     sgl = sg_lower_gpu[idx]

#     #     huh = ham_upper_H_gpu[idx]
#     #     gr = gR_gpu[IB]
#     #     pgr = gR_gpu[IB + 1]
#     #     grh = gR_H_gpu[idx]
#     #     gl = gL_gpu[IB]
#     #     pgl = gL_gpu[IB + 1]
#     #     gg = gG_gpu[IB]
#     #     pgg = gG_gpu[IB + 1]
#     #     slb = SigLB_gpu[IB]
#     #     sgb = SigGB_gpu[IB]

#     #     default_stream.wait_event(event=events[pidx])

#     #     cp.conjugate(hu[:, 0:NI, 0:NP].transpose((0,2,1)), out=huh[:, 0:NP, 0:NI])
#     #     gr[:, 0:NI, 0:NI] = cp.linalg.inv(hd[:, 0:NI, 0:NI] -
#     #                                       hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ hl[:, :NP, 0:NI])
#     #     cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)), out=grh[:, 0:NI, 0:NI])
#     #     al = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ sll[:, 0:NP, 0:NI]
#     #     slb[:, 0:NI, 0:NI] = hu[:, 0:NI, 0:NP] @ pgl[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI] - (al - cp.conjugate(al.transpose((0,2,1))))
#     #     gl[:, 0:NI, 0:NI] = gr[:, 0:NI, 0:NI] @ (sld[:, 0:NI, 0:NI] + slb[:, 0:NI, 0:NI]) @ grh[:, 0:NI, 0:NI]
#     #     ag = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ sgl[:, 0:NP, 0:NI]
#     #     sgb[:, 0:NI, 0:NI] = hu[:, 0:NI, 0:NP] @ pgg[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI] - (ag - cp.conjugate(ag.transpose((0,2,1))))
#     #     gg[:, 0:NI, 0:NI] = gr[:, 0:NI, 0:NI] @ (sgd[:, 0:NI, 0:NI] + sgb[:, 0:NI, 0:NI]) @ grh[:, 0:NI, 0:NI]


#     # Backward pass

#     # First iteration
#     IB = NB - 1
#     nIB = IB - 1
#     idx = IB % 2
#     nidx = nIB % 2
#     NN = Bmax[-1] - Bmin[-1] + 1
#     # print(f"IB: {IB}, idx: {idx}, NN: {NN}", flush=True)


#     with streams[idx] as current_stream:

#         # cpb.cuda.libs.cublas.setStream(cublas_handle, current_stream.ptr)
#         # cpb.cuda.libs.cusolver.setStream(cusolver_handle, current_stream.ptr)

#         hd = ham_diag_gpu[idx]
#         sld = sl_diag_gpu[idx]
#         sgd = sg_diag_gpu[idx]

#         gr = gR_gpu[IB]
#         grh = gR_H_gpu[idx]
#         gl = gL_gpu[IB]
#         gg = gG_gpu[IB]

#         hd.set(ham_diag[IB])
#         sld.set(sl_diag[IB])
#         sgd.set(sg_diag[IB])

#         with default_stream:
#             gr[:, 0:NN, 0:NN] = cp.linalg.inv(hd[:, 0:NN, 0:NN])
#             events[idx].record(stream=default_stream)
#             inv_event.record(stream=default_stream)
#         # events[idx].record(stream=current_stream)
#         current_stream.wait_event(event=inv_event)
#         current_stream.synchronize()
#         print(f"OK {IB}-0", flush=True)
#         # Here, potentially write gR back to host to save memory
#         cp.conjugate(gr[:, 0:NN, 0:NN].transpose((0,2,1)), out=grh[:, 0:NN, 0:NN])
#         current_stream.synchronize()
#         print(f"OK {IB}-1", flush=True)
#         cp.matmul(gr[:, 0:NN, 0:NN] @ sld[:, 0:NN, 0:NN], grh[:, 0:NN, 0:NN], out=gl[:, 0:NN, 0:NN])
#         current_stream.synchronize()
#         print(f"OK {IB}-2", flush=True)
#         cp.matmul(gr[:, 0:NN, 0:NN] @ sgd[:, 0:NN, 0:NN], grh[:, 0:NN, 0:NN], out=gg[:, 0:NN, 0:NN])
#         current_stream.synchronize()
#         print(f"OK {IB}-3", flush=True)
    
#     # Rest iterations
#     for IB in range(NB - 2, -1, -1):

#         idx = IB % 2
#         pidx = (IB + 1) % 2
#         NI = Bmax[IB] - Bmin[IB] + 1
#         NP = Bmax[IB + 1] - Bmin[IB + 1] + 1
#         print(f"IB: {IB}, idx: {idx}, pidx: {pidx}, NI: {NI}, NP: {NP}", flush=True)

#         with streams[idx] as current_stream:

#             # cpb.cuda.libs.cublas.setStream(cublas_handle, current_stream.ptr)
#             # cpb.cuda.libs.cusolver.setStream(cusolver_handle, current_stream.ptr)
                
#             hd = ham_diag_gpu[idx]
#             hu = ham_upper_gpu[idx]
#             hl = ham_lower_gpu[idx]
#             sld = sl_diag_gpu[idx]
#             sll = sl_lower_gpu[idx]
#             sgd = sg_diag_gpu[idx]
#             sgl = sg_lower_gpu[idx]

#             huh = ham_upper_H_gpu[idx]
#             gr = gR_gpu[IB]
#             pgr = gR_gpu[IB + 1]
#             grh = gR_H_gpu[idx]
#             gl = gL_gpu[IB]
#             pgl = gL_gpu[IB + 1]
#             gg = gG_gpu[IB]
#             pgg = gG_gpu[IB + 1]
#             slb = SigLB_gpu[IB]
#             sgb = SigGB_gpu[IB]


#             hd.set(ham_diag[IB])
#             hu.set(ham_upper[IB])
#             hl.set(ham_lower[IB])
#             sld.set(sl_diag[IB])
#             sll.set(sl_lower[IB])
#             sgd.set(sg_diag[IB])
#             sgl.set(sg_lower[IB])

#             cp.conjugate(hu[:, 0:NI, 0:NP].transpose((0,2,1)), out=huh[:, 0:NP, 0:NI])
#             current_stream.wait_event(event=events[pidx])
#             print(f"Event {pidx} completed", flush=True)
#             tmp = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ hl[:, :NP, 0:NI]
#             current_stream.synchronize()
#             print(f"OK {IB}-M2", flush=True)
#             print(type(tmp), tmp.shape)
#             print(type(hd), hd.shape)
#             print(type(hd[:, 0:NI, 0:NI]), hd[:, 0:NI, 0:NI].shape)
#             tmp2 = hd[:, 0:NI, 0:NI] - tmp
#             current_stream.synchronize()
#             print(f"OK {IB}-M1", flush=True)
#             print(type(tmp2), tmp2.shape, flush=True)
#             print(type(gr), gr.shape, flush=True)
#             print(type(gr[:, 0:NI, 0:NI]), gr[:, 0:NI, 0:NI].shape, flush=True)
#             # gr[:, 0:NI, 0:NI] = cp.linalg.inv(hd[:, 0:NI, 0:NI] -
#             #                                   hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ hl[:, :NP, 0:NI])
#             with default_stream:
#                 gr[:, 0:NI, 0:NI] = cp.linalg.inv(tmp2)
#                 events[idx].record(stream=default_stream)
#                 inv_event.record(stream=default_stream)
#             current_stream.wait_event(event=inv_event)
#             # gr[:, 0:NI, 0:NI] = tmp2
#             # events[idx].record(stream=current_stream)
#             current_stream.synchronize()
#             print(f"OK {IB}-0", flush=True)
#             cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)), out=grh[:, 0:NI, 0:NI])
#             current_stream.synchronize()
#             print(f"OK {IB}-1", flush=True)
#             al = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ sll[:, 0:NP, 0:NI]
#             current_stream.synchronize()
#             print(f"OK {IB}-2", flush=True)
#             slb[:, 0:NI, 0:NI] = hu[:, 0:NI, 0:NP] @ pgl[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI] - (al - al.transpose((0,2,1)))
#             current_stream.synchronize()
#             print(f"OK {IB}-3", flush=True)
#             gl[:, 0:NI, 0:NI] = gr[:, 0:NI, 0:NI] @ (sld[:, 0:NI, 0:NI] + slb[:, 0:NI, 0:NI]) @ grh[:, 0:NI, 0:NI]
#             current_stream.synchronize()
#             print(f"OK {IB}-4", flush=True)
#             ag = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP] @ sgl[:, 0:NP, 0:NI]
#             current_stream.synchronize()
#             print(f"OK {IB}-5", flush=True)
#             sgb[:, 0:NI, 0:NI] = hu[:, 0:NI, 0:NP] @ pgg[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI] - (ag - ag.transpose((0,2,1)))
#             current_stream.synchronize()
#             print(f"OK {IB}-6", flush=True)
#             gg[:, 0:NI, 0:NI] = gr[:, 0:NI, 0:NI] @ (sgd[:, 0:NI, 0:NI] + sgb[:, 0:NI, 0:NI]) @ grh[:, 0:NI, 0:NI]
#             current_stream.synchronize()
#             print(f"OK {IB}-7", flush=True)
        
#     for current_stream in streams:
#         current_stream.synchronize()
    
#     gR_gpu.get(out=GR)
#     gL_gpu.get(out=GL)
#     gG_gpu.get(out=GG)
#     cp.cuda.Stream.null.synchronize()


def rgf_standaloneGF_GPU(ham_diag, ham_upper, ham_lower,  # Input Hamiltonian + Boundary Conditions
                         sg_diag, sg_upper, sg_lower,  # Input Greater Self-Energies
                         sl_diag, sl_upper, sl_lower,  # Input Lesser Self-Energies 
                         SigGBR, SigLBR,  # Input ???
                         GR, GRnn1, GL, GLnn1, GG, GGnn1,  # Output Green's Functions
                         DOS, nE, nP, idE,  # Output Observables
                         Bmin_fi, Bmax_fi,  # Indices,
                         ham_diag_gpu, ham_upper_gpu, ham_lower_gpu,
                         sl_diag_gpu, sl_upper_gpu, sl_lower_gpu,
                         sg_diag_gpu, sg_upper_gpu, sg_lower_gpu,
                         SigLBR_gpu, SigGBR_gpu,
                         gR_gpu, gL_gpu, gG_gpu,
                         gR_H_gpu, gL_tmp, gG_tmp,
                         SigLB_gpu, SigGB_gpu,
                         AL, AG,
                         GR_gpu, GL_gpu, GG_gpu,
                         GRnn1_gpu, GLnn1_gpu, GGnn1_gpu):
    
    if ham_diag_gpu is None:
        Bmax = Bmax_fi - 1
        Bmin = Bmin_fi - 1
        Bsize = max(Bmax - Bmin + 1)  # Used for declaration of variable
        ham_diag_gpu = cp.empty_like(ham_diag)
        ham_upper_gpu = cp.empty_like(ham_upper)
        ham_lower_gpu = cp.empty_like(ham_lower)
        gR_gpu = cp.empty_like(ham_diag)
        gL_gpu = cp.empty_like(ham_diag)
        gG_gpu = cp.empty_like(ham_diag)
        gR_H_gpu = cp.empty((Bsize, Bsize), dtype=ham_diag.dtype)
        gL_tmp = cp.empty((Bsize, Bsize), dtype=ham_diag.dtype)
        gG_tmp = cp.empty((Bsize, Bsize), dtype=ham_diag.dtype)

    ham_diag_gpu.set(ham_diag)
    ham_upper_gpu.set(ham_upper)
    ham_lower_gpu.set(ham_lower)

    sl_diag_gpu.set(sl_diag)
    sl_upper_gpu.set(sl_upper)
    sl_lower_gpu.set(sl_lower)

    sg_diag_gpu.set(sg_diag)
    sg_upper_gpu.set(sg_upper)
    sg_lower_gpu.set(sg_lower)

    SigLBR_gpu.set(SigLBR)
    SigGBR_gpu.set(SigGBR)

    _rgf_gpu(ham_diag_gpu, ham_upper_gpu, ham_lower_gpu,
             sl_diag_gpu, sl_upper_gpu, sl_lower_gpu,
             sg_diag_gpu, sg_upper_gpu, sg_lower_gpu,
             SigLBR_gpu, SigGBR_gpu,
             gR_gpu, gL_gpu, gG_gpu,
             gR_H_gpu,
             gL_tmp, gG_tmp,
             SigLB_gpu, SigGB_gpu,
             None, None,
             GR_gpu, GL_gpu, GG_gpu,
             GRnn1_gpu, GLnn1_gpu, GGnn1_gpu,
             DOS, nE, nP, idE,
             Bmin_fi, Bmax_fi)
    
    # GL_gpu.get(out=GL)
    # GLnn1_gpu.get(out=GLnn1)
    # GG_gpu.get(out=GG)
    # GGnn1_gpu.get(out=GGnn1)


def _rgf_gpu(ham_diag_gpu, ham_upper_gpu, ham_lower_gpu,  # Input Hamiltonian + Boundary Conditions
             sl_diag_gpu, sl_upper_gpu, sl_lower_gpu,  # Input Lesser Self-Energies
             sg_diag_gpu, sg_upper_gpu, sg_lower_gpu,  # Input Greater Self-Energies
             SigLBR_gpu, SigGBR_gpu,  # Input ???
             gR_gpu, gL_gpu, gG_gpu,  # Temporary buffers for the Green's Functions (NB blocks)
             gR_H_gpu,  # (1 block)
             gL_tmp, gG_tmp,  # (1 block)
             SigLB_gpu, SigGB_gpu,  # Temporary buffers for ??? (NB - 1 blocks)
             AL, AG,  # Temporary buffers for the off-diagonal elements (???) (1 block)
             GR_gpu, GL_gpu, GG_gpu,  # Output Green's Functions - diagonal blocks (NB blocks)
             GRnn1_gpu, GLnn1_gpu, GGnn1_gpu,  # Output Green's Functions - off-diagonal blocks (NB - 1 blocks)
             DOS, nE, nP, idE,  # Output Observables
             Bmin_fi, Bmax_fi):

    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    Bsize = max(Bmax - Bmin + 1)  # Used for declaration of variables
    NB = len(Bmin)
    NT = Bmax[NB - 1] + 1  # Not used in this fcn
    LBsize = Bmax[0] - Bmin[0] + 1
    RBsize = Bmax[NB - 1] - Bmin[NB - 1] + 1
    energy_batchsize = ham_diag_gpu.shape[1]

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    gR_gpu[-1, 0:NN, 0:NN] = cp.linalg.inv(ham_diag_gpu[-1, 0:NN, 0:NN])
    # gL_gpu[-1, 0:NN, 0:NN] = gR_gpu[-1, 0:NN, 0:NN] @ (sl_diag_gpu[-1, 0:NN, 0:NN]) @ gR_gpu[-1, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    # gG_gpu[-1, 0:NN, 0:NN] = gR_gpu[-1, 0:NN, 0:NN] @ (sg_diag_gpu[-1, 0:NN, 0:NN]) @ gR_gpu[-1, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    cp.conjugate(gR_gpu[-1, 0:NN, 0:NN].transpose(), out=gR_H_gpu)
    cp.matmul(gR_gpu[-1, 0:NN, 0:NN], sl_diag_gpu[-1, 0:NN, 0:NN], out=gL_tmp)
    cp.matmul(gL_tmp, gR_H_gpu, out=gL_gpu[-1, 0:NN, 0:NN])
    cp.matmul(gR_gpu[-1, 0:NN, 0:NN], sg_diag_gpu[-1, 0:NN, 0:NN], out=gG_tmp)
    cp.matmul(gG_tmp, gR_H_gpu, out=gG_gpu[-1, 0:NN, 0:NN])

    return


def rgf_standaloneGF_batched_GPU(ham_diag, ham_upper, ham_lower,  # Input Hamiltonian + Boundary Conditions
                                 sg_diag, sg_upper, sg_lower,  # Input Greater Self-Energies
                                 sl_diag, sl_upper, sl_lower,  # Input Lesser Self-Energies 
                                 SigGBR, SigLBR,  # Input ???
                                 GR, GRnn1, GL, GLnn1, GG, GGnn1,  # Output Green's Functions
                                 DOS, nE, nP, idE,  # Output Observables
                                 Bmin_fi, Bmax_fi,  # Indices,
                                 ham_diag_gpu, ham_upper_gpu, ham_lower_gpu,
                                 sl_diag_gpu, sl_upper_gpu, sl_lower_gpu,
                                 sg_diag_gpu, sg_upper_gpu, sg_lower_gpu,
                                 SigLBR_gpu, SigGBR_gpu,
                                 gR_gpu, gL_gpu, gG_gpu,
                                 gR_H_gpu, gL_tmp, gG_tmp,
                                 SigLB_gpu, SigGB_gpu,
                                 AL, AG,
                                 GR_gpu, GL_gpu, GG_gpu,
                                 GRnn1_gpu, GLnn1_gpu, GGnn1_gpu):
    
    if ham_diag_gpu is None:
        Bmax = Bmax_fi - 1
        Bmin = Bmin_fi - 1
        Bsize = max(Bmax - Bmin + 1)  # Used for declaration of variable
        energy_batchsize = ham_diag.shape[1]
        ham_diag_gpu = cp.empty_like(ham_diag)
        ham_upper_gpu = cp.empty_like(ham_upper)
        ham_lower_gpu = cp.empty_like(ham_lower)
        gR_gpu = cp.empty_like(ham_diag)
        gL_gpu = cp.empty_like(ham_diag)
        gG_gpu = cp.empty_like(ham_diag)
        gR_H_gpu = cp.empty((energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
        gL_tmp = cp.empty((energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
        gG_tmp = cp.empty((energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)
    

    NB = len(ham_diag)

    for i in range(NB):

        ham_diag_gpu[i].set(ham_diag[i])
        ham_upper_gpu[i].set(ham_upper[i])
        ham_lower_gpu[i].set(ham_lower[i])

        sl_diag_gpu[i].set(sl_diag[i])
        sl_upper_gpu[i].set(sl_upper[i])
        sl_lower_gpu[i].set(sl_lower[i])

        sg_diag_gpu[i].set(sg_diag[i])
        sg_upper_gpu[i].set(sg_upper[i])
        sg_lower_gpu[i].set(sg_lower[i])

    SigLBR_gpu.set(SigLBR)
    SigGBR_gpu.set(SigGBR)

    # ham_diag_gpu.set(ham_diag)
    # ham_upper_gpu.set(ham_upper)
    # ham_lower_gpu.set(ham_lower)

    # sl_diag_gpu.set(sl_diag)
    # sl_upper_gpu.set(sl_upper)
    # sl_lower_gpu.set(sl_lower)

    # sg_diag_gpu.set(sg_diag)
    # sg_upper_gpu.set(sg_upper)
    # sg_lower_gpu.set(sg_lower)

    # SigLBR_gpu.set(SigLBR)
    # SigGBR_gpu.set(SigGBR)

    _rgf_batched_gpu(ham_diag_gpu, ham_upper_gpu, ham_lower_gpu,
                     sl_diag_gpu, sl_upper_gpu, sl_lower_gpu,
                     sg_diag_gpu, sg_upper_gpu, sg_lower_gpu,
                     SigLBR_gpu, SigGBR_gpu,
                     gR_gpu, gL_gpu, gG_gpu,
                     gR_H_gpu,
                     gL_tmp, gG_tmp,
                     SigLB_gpu, SigGB_gpu,
                     None, None,
                     GR_gpu, GL_gpu, GG_gpu,
                     GRnn1_gpu, GLnn1_gpu, GGnn1_gpu,
                     DOS, nE, nP, idE,
                     Bmin_fi, Bmax_fi)
    
    # GL_gpu.get(out=GL)
    # GLnn1_gpu.get(out=GLnn1)
    # GG_gpu.get(out=GG)
    # GGnn1_gpu.get(out=GGnn1)


def _rgf_batched_gpu(ham_diag_gpu, ham_upper_gpu, ham_lower_gpu,  # Input Hamiltonian + Boundary Conditions
                     sl_diag_gpu, sl_upper_gpu, sl_lower_gpu,  # Input Lesser Self-Energies
                     sg_diag_gpu, sg_upper_gpu, sg_lower_gpu,  # Input Greater Self-Energies
                     SigLBR_gpu, SigGBR_gpu,  # Input ???
                     gR_gpu, gL_gpu, gG_gpu,  # Temporary buffers for the Green's Functions (NB blocks)
                     gR_H_gpu,  # (1 block)
                     gL_tmp, gG_tmp,  # (1 block)
                     SigLB_gpu, SigGB_gpu,  # Temporary buffers for ??? (NB - 1 blocks)
                     AL, AG,  # Temporary buffers for the off-diagonal elements (???) (1 block)
                     GR_gpu, GL_gpu, GG_gpu,  # Output Green's Functions - diagonal blocks (NB blocks)
                     GRnn1_gpu, GLnn1_gpu, GGnn1_gpu,  # Output Green's Functions - off-diagonal blocks (NB - 1 blocks)
                     DOS, nE, nP, idE,  # Output Observables
                     Bmin_fi, Bmax_fi):

    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    Bsize = max(Bmax - Bmin + 1)  # Used for declaration of variables
    NB = len(Bmin)
    NT = Bmax[NB - 1] + 1  # Not used in this fcn
    LBsize = Bmax[0] - Bmin[0] + 1
    RBsize = Bmax[NB - 1] - Bmin[NB - 1] + 1
    energy_batchsize = ham_diag_gpu.shape[1]

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    gR_gpu[-1, :, 0:NN, 0:NN] = cp.linalg.inv(ham_diag_gpu[-1, :, 0:NN, 0:NN])
    # gL_gpu[-1, :, 0:NN, 0:NN] = gR_gpu[-1, :, 0:NN, 0:NN] @ (sl_diag_gpu[-1, :, 0:NN, 0:NN]) @ gR_gpu[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    # gG_gpu[-1, :, 0:NN, 0:NN] = gR_gpu[-1, :, 0:NN, 0:NN] @ (sg_diag_gpu[-1, :, 0:NN, 0:NN]) @ gR_gpu[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    cp.conjugate(gR_gpu[-1, :, 0:NN, 0:NN].transpose((0,2,1)), out=gR_H_gpu)
    cp.matmul(gR_gpu[-1, :, 0:NN, 0:NN], sl_diag_gpu[-1, :, 0:NN, 0:NN], out=gL_tmp)
    cp.matmul(gL_tmp, gR_H_gpu, out=gL_gpu[-1, :, 0:NN, 0:NN])
    cp.matmul(gR_gpu[-1, :, 0:NN, 0:NN], sg_diag_gpu[-1, :, 0:NN, 0:NN], out=gG_tmp)
    cp.matmul(gG_tmp, gR_H_gpu, out=gG_gpu[-1, :, 0:NN, 0:NN])

    for IB in range(NB - 2, -1, -1):
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.inv(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
                                                  - ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                                                  @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
                                                  @ ham_lower_gpu[IB, :, :NP, 0:NI])
        # gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.solve(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
        #                     - ham_upper_gpu[IB, :, 0:NI, 0:NP] \
        #                     @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
        #                     @ ham_lower_gpu[IB, :, :NP, 0:NI], gpu_identity_batch)#######
        # AL, What is this? Handling off-diagonal sigma elements?
        AL[:] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
            @ sl_lower_gpu[IB, :, 0:NP, 0:NI]
        
        SigLB_gpu[IB, :, 0:NI, 0:NI] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gL_gpu[IB+1, :, 0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj())

        # gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigL_c \
        #                     + M_r \
        #                     @ gL[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AL - AL.T.conj()))  \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL

        gL_gpu[IB, :, 0:NI, 0:NI] = gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ (sl_diag_gpu[IB, :, 0:NI, 0:NI] \
                            + SigLB_gpu[IB, :, 0:NI, 0:NI])  \
                            @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0, 2, 1)).conj() # Confused about the AL

        ### What is this?
        AG[:] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
            @ sg_lower_gpu[IB, :, 0:NP, 0:NI]     # Handling off-diagonal sigma elements? Prob. need to check

        # gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigG_c \
        #                     + M_r \
        #                     @ gG[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AG - AG.T.conj())) \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
        SigGB_gpu[IB, :, 0:NI, 0:NI] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gG_gpu[IB+1, :,  0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj())

        gG_gpu[IB, :, 0:NI, 0:NI] = gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ (sg_diag_gpu[IB, :, 0:NI, 0:NI] \
                                + SigGB_gpu[IB, :, 0:NI, 0:NI]) \
                                @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() # Confused about the AG. 




    GR_gpu[0, :,  :NI, :NI] = gR_gpu[0, :, :NI, :NI]
    GRnn1_gpu[0, :,  :NI, :NP] = -GR_gpu[0, :, :NI, :NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :, :NP, :NP]

    GL_gpu[0, :, :NI, :NI] = gL_gpu[0, :, :NI, :NI]
    GLnn1_gpu[0, :, :NI, :NP] = GR_gpu[0, :, :NI, :NI] @ sl_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :, :NP, :NP].transpose((0,2,1)).conj() \
                - GR_gpu[0,:, :NI,:NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gL_gpu[1,:, :NP, :NP] \
                - GL_gpu[0,:, :NI,:NI] @ ham_lower_gpu[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR_gpu[1,:, :NP, :NP].transpose((0,2,1)).conj()

    GG_gpu[0, :, :NI, :NI] = gG_gpu[0, :, :NI, :NI]
    GGnn1_gpu[0, :, :NI, :NP] = GR_gpu[0, :,:NI, :NI] @ sg_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :,:NP, :NP].transpose((0,2,1)).conj() \
                - GR_gpu[0,:, :NI,:NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gG_gpu[1, :, :NP, :NP] \
                - GG_gpu[0,:,:NI,:NI] @ ham_lower_gpu[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR_gpu[1, :, :NP, :NP].transpose((0,2,1)).conj() 
    
    idE[:, 0] = cp.real(cp.trace(SigGB_gpu[0, :, :NI, :NI] @ GL_gpu[0, :, :NI, :NI] - GG_gpu[0, :, :NI, :NI] @ SigLB_gpu[0, :, :NI, :NI], axis1 = 1, axis2 = 2)).get()

    for IB in range(1, NB):

        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        # # Extracting off-diagonal Hamiltonian block (upper)
        # M_u = M[Bmin[IB - 1]:Bmax[IB - 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # # Extracting off-diagonal Hamiltonian block (left)
        # M_l = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (left)
        # SigL_l = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (left)
        # SigG_l = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        GR_gpu[IB, :, :NI, :NI] = gR_gpu[IB, :, :NI, :NI] + gR_gpu[IB, :,  :NI, :NI] \
                        @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                        @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
                        @ ham_upper_gpu[IB-1, :, :NM, :NI] \
                        @ gR_gpu[IB, :,  :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = gR_gpu[IB, :, :NI, :NI] \
            @ sl_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR_gpu[IB,:, 0:NI, 0:NI].transpose((0,2,1)).conj()
        # What is this?
        BL = gR_gpu[IB, :, :NI, :NI] \
            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :, :NM, :NI] \
            @ gL_gpu[IB, :, :NI, :NI]

        GL_gpu[IB, :, 0:NI, 0:NI] = gL_gpu[IB, :, :NI, :NI] \
                            + gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                            @ GL_gpu[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR_gpu[IB, :, :NI, :NI].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj()) + (BL - BL.transpose((0,2,1)).conj())


        AG = gR_gpu[IB, :, 0:NI, 0:NI] \
            @ sg_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR_gpu[IB, :, :NI, :NI].transpose((0,2,1)).conj()

        BG = gR_gpu[IB, :, 0:NI, 0:NI] \
            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :, :NM, :NI] \
            @ gG_gpu[IB, :, 0:NI, 0:NI]

        GG_gpu[IB, :, 0:NI, 0:NI] = gG_gpu[IB, :, 0:NI, 0:NI] \
                            + gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                            @ GG_gpu[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj()) + (BG - BG.transpose((0,2,1)).conj()) #

        if IB < NB - 1:  #Off-diagonal are only interesting for IdE!

            # # Extracting off-diagonal Hamiltonian block (right)
            # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # # Extracting off-diagonal Hamiltonian block (lower)
            # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # # Extracting off-diagonal lesser Self-energy block (right)
            # SigL_r = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # Extracting off-diagonal greater Self-energy block (right)
            # SigG_r = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1_gpu[IB, :, 0:NI, 0:NP] = - GR_gpu[IB, :,  0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP]

            GLnn1_gpu[IB, :, 0:NI, 0:NP] = GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ sl_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI] \
                                    @ gL_gpu[IB+1, :, 0:NP, 0:NP] \
                                    - GL_gpu[IB, :, :NI, :NI] \
                                    @ ham_lower_gpu[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            GGnn1_gpu[IB, :, 0:NI, 0:NP] = GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ sg_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI]  \
                                    @ gG_gpu[IB+1, :, 0:NP, 0:NP] \
                                    - GG_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_lower_gpu[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            idE[:, IB] = cp.real(cp.trace(SigGB_gpu[IB, :NI, :NI] @ GL_gpu[IB, :NI, :NI] - GG_gpu[IB, :NI, :NI] @ SigLB_gpu[IB, :NI, :NI], axis1 = 1, axis2 = 2)).get() 
    for IB in range(NB):
        
        NI = Bmax[IB] - Bmin[IB] + 1
        # GR[IB, :, :, :] *= factor
        # GL[IB, :, :, :] *= factor
        # GG[IB, :, :, :] *= factor
        DOS[:, IB] = 1j * cp.trace(GR_gpu[IB, :, :, :] - GR_gpu[IB, :, :, :].transpose((0,2,1)).conj(), axis1= 1, axis2 = 2).get()
        nE[:, IB] = -1j * cp.trace(GL_gpu[IB, :, :, :], axis1= 1, axis2 = 2).get()
        nP[:, IB] = 1j * cp.trace(GG_gpu[IB, :, :, :], axis1= 1, axis2 = 2).get()

        # if IB < NB-1:
        #     NP = Bmax[IB+1] - Bmin[IB+1] + 1
        #     #idE[IB] = -2 * np.trace(np.real(H[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].toarray() @ GLnn1[IB, 0:NI, 0:NP]))
        #     # GRnn1[IB, :, :, :] *= factor
        #     # GLnn1[IB, :, :, :] *= factor
        #     # GGnn1[IB, :, :, :] *= factor

    
    #idE[NB - 1] = idE[NB - 2]
    idE[:, NB-1] = cp.real(cp.trace(SigGBR_gpu[:, :NI, :NI] @ GL_gpu[NB-1, :, :NI, :NI] - GG_gpu[NB-1, :, :NI, :NI] @ SigLBR_gpu[:, :NI, :NI], axis1 = 1, axis2 = 2)).get()

    #Final Data Transfer
    #GR[:, :, :, :] = GR_gpu.get()
    GL[:, :, :, :] = GL_gpu.get()
    GG[:, :, :, :] = GG_gpu.get()
    #GRnn1[:, :, :, :] = GRnn1_gpu.get()
    GLnn1[:, :, :, :] = GLnn1_gpu.get()
    GGnn1[:, :, :, :] = GGnn1_gpu.get()
