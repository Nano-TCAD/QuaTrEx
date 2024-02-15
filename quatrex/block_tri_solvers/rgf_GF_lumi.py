# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
import cupy as cp


def rgf_standaloneGF_batched_GPU(ham_diag, ham_upper, ham_lower,  # Input Hamiltonian + Boundary Conditions
                                 sg_diag, sg_upper, sg_lower,  # Input Greater Self-Energies
                                 sl_diag, sl_upper, sl_lower,  # Input Lesser Self-Energies
                                 SigGBR, SigLBR,  # Input ???
                                 GR_host, GRnn1_host,  # Output Retarded Green's Functions (unused)
                                 GL_host, GLnn1_host,  # Output Lesser Green's Functions
                                 GG_host, GGnn1_host,  # Output Greater Green's Functions
                                 DOS, nE, nP, idE,  # Output Observables
                                 Bmin_fi, Bmax_fi,  # Indices
                                 solve: bool = True
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

    computation_stream = cp.cuda.Stream.null
    input_stream = cp.cuda.Stream(non_blocking=True)
    output_stream = cp.cuda.Stream(non_blocking=True)
    input_events = [cp.cuda.Event() for _ in range(2)]
    computation_event = cp.cuda.Event()

    gR_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Retarded (right)
    gL_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser (right)
    gG_gpu = cp.empty((NB, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater (right)
    SigLB_gpu = cp.empty((NB-1, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Lesser boundary self-energy
    SigGB_gpu = cp.empty((NB-1, energy_batchsize, Bsize, Bsize), dtype=ham_diag.dtype)  # Greater boundary self-energy
    DOS_gpu = cp.empty((energy_batchsize, NB), dtype=ham_diag.dtype)
    nE_gpu = cp.empty((energy_batchsize, NB), dtype=ham_diag.dtype)
    nP_gpu = cp.empty((energy_batchsize, NB), dtype=ham_diag.dtype)
    idE_gpu = cp.empty((energy_batchsize, NB), dtype=idE.dtype)

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
        with input_stream:

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
            input_events[idx].record(stream=input_stream)

    if solve:
        gpu_identity = cp.identity(NN, dtype=ham_diag.dtype)
        gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis=0)
        gr[:, 0:NN, 0:NN] = cp.linalg.solve(hd[:, 0:NN, 0:NN], gpu_identity_batch)
        computation_stream.synchronize()
    else:
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

        with input_stream:
                
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

            input_events[idx].record(stream=input_stream)

        computation_stream.wait_event(event=input_events[pidx])
        hupgr = hu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP]
        al = hupgr @ sll[:, 0:NP, 0:NI]
        ag = hupgr @ sgl[:, 0:NP, 0:NI]
        inv_arg = hd[:, 0:NI, 0:NI] - hupgr @ hl[:, :NP, 0:NI]
        if solve:
            gpu_identity = cp.identity(NI, dtype=ham_diag.dtype)
            gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis=0)
            gr[:, 0:NI, 0:NI] = cp.linalg.solve(inv_arg, gpu_identity_batch)
            computation_stream.synchronize()
        else:
            gr[:, 0:NI, 0:NI] = cp.linalg.inv(inv_arg)
        cp.conjugate(hu[:, 0:NI, 0:NP].transpose((0,2,1)), out=huh[:, 0:NP, 0:NI])
        cp.subtract(hu[:, 0:NI, 0:NP] @ pgl[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI],
                    al - cp.conjugate(al.transpose((0,2,1))), out=slb[:, 0:NI, 0:NI])
        cp.subtract(hu[:, 0:NI, 0:NP] @ pgg[:, 0:NP, 0:NP] @ huh[:, 0:NP, 0:NI],
                    ag - cp.conjugate(ag.transpose((0,2,1))), out=sgb[:, 0:NI, 0:NI])
        cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)), out=grh[:, 0:NI, 0:NI])
        cp.matmul(gr[:, 0:NI, 0:NI] @ (sld[:, 0:NI, 0:NI] + slb[:, 0:NI, 0:NI]), grh[:, 0:NI, 0:NI], out=gl[:, 0:NI, 0:NI])
        cp.matmul(gr[:, 0:NI, 0:NI] @ (sgd[:, 0:NI, 0:NI] + sgb[:, 0:NI, 0:NI]), grh[:, 0:NI, 0:NI], out=gg[:, 0:NI, 0:NI])

        if IB == 0:

            computation_stream.synchronize()
            computation_event.record(stream=computation_stream)

            with output_stream:
                output_stream.wait_event(event=computation_event)
                # GR_gpu[0].get(out=GR_host[0])
                GL_gpu[0].get(out=GL_host[0])
                GG_gpu[0].get(out=GG_host[0])

            DOS_gpu[:, 0] = 1j * cp.trace(gr[:, 0:NI, 0:NI] - grh[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nE_gpu[:, 0] = -1j * cp.trace(gl[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nP_gpu[:, 0] = 1j * cp.trace(gg[:, 0:NI, 0:NI], axis1=1, axis2=2)
            idE_gpu[:, 0] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ gl[:, 0:NI, 0:NI] -
                                             gg[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))

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

    with input_stream:
            
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

            input_events[nidx].record(stream=input_stream)

    computation_stream.wait_event(event=input_events[idx])
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

    computation_stream.synchronize()
    computation_event.record(stream=computation_stream)
    with output_stream:
        output_stream.wait_event(event=computation_event)
        # GRnn1.get(out=GRnn1_host[0])
        GLnn1.get(out=GLnn1_host[0])
        GGnn1.get(out=GGnn1_host[0])

    # Rest iterations
    for IB in range(1, NB):

        pIB = IB - 1
        nIB = IB + 1
        idx = IB % 2
        pidx = pIB % 2
        nidx = nIB % 2
        NI = Bmax[IB] - Bmin[IB] + 1
        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1

        phu = ham_upper_gpu[pidx]
        phl = ham_lower_gpu[pidx]
        psll = sl_lower_gpu[pidx]
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

        with input_stream:
                
            if nIB < NB:

                nphu = ham_upper_gpu[idx]
                nphl = ham_lower_gpu[idx]
                npsll = sl_lower_gpu[idx]
                npsgl = sg_lower_gpu[idx]

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

            input_events[nidx].record(stream=input_stream)

        computation_stream.wait_event(event=input_events[idx])
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
            cp.negative(GR[:, 0:NI, 0:NI] @ hu[:, 0:NI, 0:NP] @ ngr[:, 0:NP, 0:NP], out=GRnn1[:, 0:NI, 0:NP])
            cp.subtract(GR[:, 0:NI, 0:NI] @ slu[:, 0:NI, 0:NP] @ ngrh,
                        GRhu @ ngl[:, 0:NP, 0:NP] + GL[:, 0:NI, 0:NI] @ hlhngrh, out=GLnn1[:, 0:NI, 0:NP])
            cp.subtract(GR[:, 0:NI, 0:NI] @ sgu[:, 0:NI, 0:NP] @ ngrh,
                        GRhu @ ngg[:, 0:NP, 0:NP] + GG[:, 0:NI, 0:NI] @ hlhngrh, out=GGnn1[:, 0:NI, 0:NP])
        
            computation_stream.synchronize()
            computation_event.record(stream=computation_stream)
            with output_stream:
                output_stream.wait_event(event=computation_event)
                # GR.get(out=GR_host[IB])
                GL.get(out=GL_host[IB])
                GG.get(out=GG_host[IB])
                # GRnn1.get(out=GRnn1_host[IB])
                GLnn1.get(out=GLnn1_host[IB])
                GGnn1.get(out=GGnn1_host[IB])

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
    computation_stream.wait_event(event=input_events[nidx])
    idE_gpu[:, NB - 1] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ GL[:, 0:NI, 0:NI] -
                                          GG[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))

    GL.get(out=GL_host[-1])
    GG.get(out=GG_host[-1])
    DOS_gpu.get(out=DOS)
    nE_gpu.get(out=nE)
    nP_gpu.get(out=nP)
    idE_gpu.get(out=idE)
    output_stream.synchronize()
    computation_stream.synchronize()
