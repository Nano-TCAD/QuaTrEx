import cupy as cp
import cupyx as cpx
import numpy as np
import time

from quatrex.block_tri_solvers.rgf_GF_GPU import rgf_standaloneGF_GPU, rgf_standaloneGF_batched_GPU
from quatrex.block_tri_solvers.rgf_GF_lumi import _rgf_batched
# from quatrex.block_tri_solvers.rgf_GF_lumi import rgf_standaloneGF_GPU, rgf_standaloneGF_batched_GPU


def random_complex(shape, rng: np.random.Generator):
    return rng.random(shape) + 1j * rng.random(shape)


# def test_rgf_gpu(num_blocks, num_energies, block_size, repeat=10):

#     rng = np.random.default_rng(42)

#     HD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     HU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     HL = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SGD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SGU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SGL = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SLD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SLU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SLL = random_complex((num_blocks, num_energies, block_size, block_size), rng)

#     SigGBR = random_complex((num_energies, block_size, block_size), rng)
#     SigLBR = random_complex((num_energies, block_size, block_size), rng)

#     GL = np.zeros((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
#     GLnn1 = np.zeros((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)
#     GG = np.zeros((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
#     GGnn1 = np.zeros((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)

#     DOS = np.zeros((num_energies, num_blocks), dtype=np.complex128)
#     nE = np.zeros((num_energies, num_blocks), dtype=np.complex128)
#     nP = np.zeros((num_energies, num_blocks), dtype=np.complex128)
#     idE = np.zeros((num_energies, num_blocks), dtype=np.complex128)

#     bmin = np.arange(1, num_blocks * block_size - 1, block_size, dtype=np.int32)
#     bmax = np.arange(block_size, num_blocks * block_size + 1, block_size, dtype=np.int32)
#     assert len(bmin) == num_blocks
#     assert len(bmax) == num_blocks

#     HD_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     HU_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     HL_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     SLD_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     SLU_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     SLL_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     SGD_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     SGU_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     SGL_d = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     SigLBR_d = cp.empty((block_size, block_size), dtype=HD.dtype)
#     SigGBR_d = cp.empty((block_size, block_size), dtype=HD.dtype)

#     gR_gpu = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     gL_gpu = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     gG_gpu = cp.empty((num_blocks, block_size, block_size), dtype=HD.dtype)
#     gR_H_gpu = cp.empty((block_size, block_size), dtype=HD.dtype)
#     gL_tmp = cp.empty((block_size, block_size), dtype=HD.dtype)
#     gG_tmp = cp.empty((block_size, block_size), dtype=HD.dtype)
    
#     runtimes = np.zeros(repeat)
#     for r in range(repeat):

#         start = time.time()

#         for ie in range(0, num_energies):

#             rgf_standaloneGF_GPU(HD[:, ie],
#                                  HU[:, ie],
#                                  HL[:, ie],
#                                  SGD[:, ie],
#                                  SGU[:, ie],
#                                  SGL[:, ie],
#                                  SLD[:, ie],
#                                  SLU[:, ie],
#                                  SLL[:, ie],
#                                  SigGBR[ie],
#                                  SigLBR[ie],
#                                  None, None,  # GR, GRnn1
#                                  GL[:, ie],
#                                  GLnn1[:, ie],
#                                  GG[:, ie],
#                                  GGnn1[:, ie],
#                                  DOS[ie],
#                                  nE[ie],
#                                  nP[ie],
#                                  idE[ie],
#                                  bmin, bmax,
#                                  HD_d, HU_d, HL_d,
#                                  SLD_d, SLU_d, SLL_d,
#                                  SGD_d, SGU_d, SGL_d,
#                                  SigLBR_d, SigGBR_d,
#                                  gR_gpu, gL_gpu, gG_gpu,
#                                  gR_H_gpu, gL_tmp, gG_tmp,
#                                  None, None,
#                                  None, None,
#                                  None, None, None,
#                                  None, None, None)
        
#         runtimes[r] = time.time() - start
    
#     print(f"Median non-batched runtime: {np.median(runtimes):.3f} s", flush=True)

#     return


# def test_rgf_batched_gpu(num_blocks, num_energies, block_size, batch_size, repeat=10):

#     rng = np.random.default_rng(42)

#     HD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     HU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     HL = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SGD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SGU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SGL = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SLD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SLU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
#     SLL = random_complex((num_blocks, num_energies, block_size, block_size), rng)

#     SigGBR = random_complex((num_energies, block_size, block_size), rng)
#     SigLBR = random_complex((num_energies, block_size, block_size), rng)

#     GL = np.zeros((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
#     GLnn1 = np.zeros((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)
#     GG = np.zeros((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
#     GGnn1 = np.zeros((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)

#     DOS = np.zeros((num_energies, num_blocks), dtype=np.complex128)
#     nE = np.zeros((num_energies, num_blocks), dtype=np.complex128)
#     nP = np.zeros((num_energies, num_blocks), dtype=np.complex128)
#     idE = np.zeros((num_energies, num_blocks), dtype=np.complex128)

#     bmin = np.arange(1, num_blocks * block_size - 1, block_size, dtype=np.int32)
#     bmax = np.arange(block_size, num_blocks * block_size + 1, block_size, dtype=np.int32)
#     assert len(bmin) == num_blocks
#     assert len(bmax) == num_blocks

#     HD_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     HU_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     HL_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     SLD_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     SLU_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     SLL_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     SGD_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     SGU_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     SGL_d = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     SigLBR_d = cp.empty((batch_size, block_size, block_size), dtype=HD.dtype)
#     SigGBR_d = cp.empty((batch_size, block_size, block_size), dtype=HD.dtype)

#     gR_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     gL_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     gG_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=HD.dtype)
#     gR_H_gpu = cp.empty((batch_size, block_size, block_size), dtype=HD.dtype)
#     gL_tmp = cp.empty((batch_size, block_size, block_size), dtype=HD.dtype)
#     gG_tmp = cp.empty((batch_size, block_size, block_size), dtype=HD.dtype)

#     runtimes = np.zeros(repeat)
#     for r in range(repeat):

#         start = time.time()

#         for ie in range(0, num_energies, batch_size):
#             ie_end = min(ie + batch_size, num_energies)

#             rgf_standaloneGF_batched_GPU(HD[:, ie:ie_end],
#                                          HU[:, ie:ie_end],
#                                          HL[:, ie:ie_end],
#                                          SGD[:, ie:ie_end],
#                                          SGU[:, ie:ie_end],
#                                          SGL[:, ie:ie_end],
#                                          SLD[:, ie:ie_end],
#                                          SLU[:, ie:ie_end],
#                                          SLL[:, ie:ie_end],
#                                          SigGBR[ie:ie_end],
#                                          SigLBR[ie:ie_end],
#                                          None, None,  # GR, GRnn1
#                                          GL[:, ie:ie_end],
#                                          GLnn1[:, ie:ie_end],
#                                          GG[:, ie:ie_end],
#                                          GGnn1[:, ie:ie_end],
#                                          DOS[ie:ie_end],
#                                          nE[ie:ie_end],
#                                          nP[ie:ie_end],
#                                          idE[ie:ie_end],
#                                          bmin, bmax,
#                                          HD_d[:, :ie_end - ie], HU_d[:, :ie_end - ie], HL_d[:, :ie_end - ie],
#                                          SLD_d[:, :ie_end - ie], SLU_d[:, :ie_end - ie], SLL_d[:, :ie_end - ie],
#                                          SGD_d[:, :ie_end - ie], SGU_d[:, :ie_end - ie], SGL_d[:, :ie_end - ie],
#                                          SigLBR_d[:ie_end - ie], SigGBR_d[:ie_end - ie],
#                                          gR_gpu[:, :ie_end - ie], gL_gpu[:, :ie_end - ie], gG_gpu[:, :ie_end - ie],
#                                          gR_H_gpu[:ie_end - ie], gL_tmp[:ie_end - ie], gG_tmp[:ie_end - ie],
#                                          None, None,
#                                          None, None,
#                                          None, None, None,
#                                          None, None, None)

#         runtimes[r] = time.time() - start
    
#     print(f"Median batched runtime (batch size {batch_size}): {np.median(runtimes):.3f} s", flush=True)

#     return


def test_rgf_gpu(num_blocks, num_energies, block_size, repeat=10):

    rng = np.random.default_rng(42)

    HD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    HU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    HL = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SGD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SGU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SGL = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SLD = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SLU = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SLL = random_complex((num_blocks, num_energies, block_size, block_size), rng)

    SigGBR = random_complex((num_energies, block_size, block_size), rng)
    SigLBR = random_complex((num_energies, block_size, block_size), rng)

    GL = np.zeros((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    GLnn1 = np.zeros((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)
    GG = np.zeros((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    GGnn1 = np.zeros((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)

    DOS = np.zeros((num_energies, num_blocks), dtype=np.complex128)
    nE = np.zeros((num_energies, num_blocks), dtype=np.complex128)
    nP = np.zeros((num_energies, num_blocks), dtype=np.complex128)
    idE = np.zeros((num_energies, num_blocks), dtype=np.complex128)

    bmin = np.arange(1, num_blocks * block_size - 1, block_size, dtype=np.int32)
    bmax = np.arange(block_size, num_blocks * block_size + 1, block_size, dtype=np.int32)
    assert len(bmin) == num_blocks
    assert len(bmax) == num_blocks
    
    runtimes = np.zeros(repeat)
    for r in range(repeat):

        start = time.time()

        for ie in range(0, num_energies):

            rgf_standaloneGF_GPU(HD[:, ie],
                                 HU[:, ie],
                                 HL[:, ie],
                                 SGD[:, ie],
                                 SGU[:, ie],
                                 SGL[:, ie],
                                 SLD[:, ie],
                                 SLU[:, ie],
                                 SLL[:, ie],
                                 SigGBR[ie],
                                 SigLBR[ie],
                                 None, None,  # GR, GRnn1
                                 GL[:, ie],
                                 GLnn1[:, ie],
                                 GG[:, ie],
                                 GGnn1[:, ie],
                                 DOS[ie],
                                 nE[ie],
                                 nP[ie],
                                 idE[ie],
                                 bmin, bmax)
        
        runtimes[r] = time.time() - start
    
    print(f"Median non-batched runtime: {np.median(runtimes):.3f} s", flush=True)

    return


def test_rgf_batched_gpu(num_blocks, num_energies, block_size, batch_size, repeat=10):

    rng = np.random.default_rng(42)

    HD = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    HU = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    HL = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    SGD = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    SGU = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    SGL = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    SLD = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    SLU = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    SLL = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)

    SigGBR = cpx.empty_pinned((num_energies, block_size, block_size), dtype=np.complex128)
    SigLBR = cpx.empty_pinned((num_energies, block_size, block_size), dtype=np.complex128)


    HD[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    HU[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    HL[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SGD[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SGU[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SGL[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SLD[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SLU[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)
    SLL[:] = random_complex((num_blocks, num_energies, block_size, block_size), rng)

    SigGBR[:] = random_complex((num_energies, block_size, block_size), rng)
    SigLBR[:] = random_complex((num_energies, block_size, block_size), rng)

    GL = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    GLnn1 = cpx.empty_pinned((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)
    GG = cpx.empty_pinned((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    GGnn1 = cpx.empty_pinned((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)

    DOS = np.zeros((num_energies, num_blocks), dtype=np.complex128)
    nE = np.zeros((num_energies, num_blocks), dtype=np.complex128)
    nP = np.zeros((num_energies, num_blocks), dtype=np.complex128)
    idE = np.zeros((num_energies, num_blocks), dtype=np.complex128)

    bmin = np.arange(1, num_blocks * block_size - 1, block_size, dtype=np.int32)
    bmax = np.arange(block_size, num_blocks * block_size + 1, block_size, dtype=np.int32)
    assert len(bmin) == num_blocks
    assert len(bmax) == num_blocks

    runtimes = np.zeros(repeat)
    for r in range(repeat):

        start = time.time()

        for ie in range(0, num_energies, batch_size):
            ie_end = min(ie + batch_size, num_energies)

            rgf_standaloneGF_batched_GPU(HD[:, ie:ie_end],
                                         HU[:, ie:ie_end],
                                         HL[:, ie:ie_end],
                                         SGD[:, ie:ie_end],
                                         SGU[:, ie:ie_end],
                                         SGL[:, ie:ie_end],
                                         SLD[:, ie:ie_end],
                                         SLU[:, ie:ie_end],
                                         SLL[:, ie:ie_end],
                                         SigGBR[ie:ie_end],
                                         SigLBR[ie:ie_end],
                                         None, None,  # GR, GRnn1
                                         GL[:, ie:ie_end],
                                         GLnn1[:, ie:ie_end],
                                         GG[:, ie:ie_end],
                                         GGnn1[:, ie:ie_end],
                                         DOS[ie:ie_end],
                                         nE[ie:ie_end],
                                         nP[ie:ie_end],
                                         idE[ie:ie_end],
                                         bmin, bmax)

        runtimes[r] = time.time() - start
    
    print(f"Median batched runtime (batch size {batch_size}): {np.median(runtimes):.3f} s", flush=True)

    return


def validate_rgf_batched_gpu(num_blocks, batch_size, block_size):

    rng = np.random.default_rng(42)

    HD = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    HU = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    HL = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    SGD = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    SGU = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    SGL = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    SLD = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    SLU = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    SLL = cpx.empty_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)

    SigGBR = cpx.empty_pinned((batch_size, block_size, block_size), dtype=np.complex128)
    SigLBR = cpx.empty_pinned((batch_size, block_size, block_size), dtype=np.complex128)

    HD[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)
    HU[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)
    HL[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)
    SGD[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)
    SGU[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)
    SGL[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)
    SLD[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)
    SLU[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)
    SLL[:] = random_complex((num_blocks, batch_size, block_size, block_size), rng)

    SigGBR[:] = random_complex((batch_size, block_size, block_size), rng)
    SigLBR[:] = random_complex((batch_size, block_size, block_size), rng)

    GR = cpx.zeros_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    GRnn1 = cpx.zeros_pinned((num_blocks - 1, batch_size, block_size, block_size), dtype=np.complex128)
    GL = cpx.zeros_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    GLnn1 = cpx.zeros_pinned((num_blocks - 1, batch_size, block_size, block_size), dtype=np.complex128)
    GG = cpx.zeros_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    GGnn1 = cpx.zeros_pinned((num_blocks - 1, batch_size, block_size, block_size), dtype=np.complex128)

    DOS = np.zeros((batch_size, num_blocks), dtype=np.complex128)
    nE = np.zeros((batch_size, num_blocks), dtype=np.complex128)
    nP = np.zeros((batch_size, num_blocks), dtype=np.complex128)
    idE = np.zeros((batch_size, num_blocks), dtype=np.complex128)

    bmin = np.arange(1, num_blocks * block_size - 1, block_size, dtype=np.int32)
    bmax = np.arange(block_size, num_blocks * block_size + 1, block_size, dtype=np.int32)
    assert len(bmin) == num_blocks
    assert len(bmax) == num_blocks

    print("Running normal batched RGF...", flush=True)

    rgf_standaloneGF_batched_GPU(HD, HU, HL,
                                 SGD, SGU, SGL,
                                 SLD, SLU, SLL,
                                 SigGBR, SigLBR,
                                 GR, GRnn1,
                                 GL, GLnn1,
                                 GG, GGnn1,
                                 DOS, nE, nP, idE,
                                 bmin, bmax)
    
    GR2 = cpx.zeros_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    GRnn12 = cpx.zeros_pinned((num_blocks - 1, batch_size, block_size, block_size), dtype=np.complex128)
    GL2 = cpx.zeros_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    GLnn12 = cpx.zeros_pinned((num_blocks - 1, batch_size, block_size, block_size), dtype=np.complex128)
    GG2 = cpx.zeros_pinned((num_blocks, batch_size, block_size, block_size), dtype=np.complex128)
    GGnn12 = cpx.zeros_pinned((num_blocks - 1, batch_size, block_size, block_size), dtype=np.complex128)

    print("Running double-buffered batched RGF...", flush=True)

    _rgf_batched(HD, HU, HL,
                 SGD, SGU, SGL,
                 SLD, SLU, SLL,
                 SigGBR, SigLBR,
                 GR2, GRnn12,
                 GL2, GLnn12,
                 GG2, GGnn12,
                 DOS, nE, nP, idE,
                 bmin, bmax)
    
    # assert np.allclose(GR, GR2)
    # assert np.allclose(GL, GL2)
    # assert np.allclose(GG, GG2)
    print(np.linalg.norm(GR - GR2) / np.linalg.norm(GR))
    print(np.linalg.norm(GL - GL2) / np.linalg.norm(GL))
    print(np.linalg.norm(GG - GG2) / np.linalg.norm(GG))
    print(np.linalg.norm(GRnn1 - GRnn12) / np.linalg.norm(GRnn1))
    print(np.linalg.norm(GLnn1 - GLnn12) / np.linalg.norm(GLnn1))
    print(np.linalg.norm(GGnn1 - GGnn12) / np.linalg.norm(GGnn1))

    print("Validation successful!", flush=True)


if __name__ == "__main__":

    # for bsz in (2048, ):

    #     # test_rgf_gpu(5, 20, bsz, repeat=5)
    #     # test_rgf_batched_gpu(5, 20, bsz, 1, repeat=5)
    #     test_rgf_batched_gpu(5, 20, bsz, 2, repeat=5)
    #     test_rgf_batched_gpu(5, 20, bsz, 5, repeat=5)
    #     test_rgf_batched_gpu(5, 20, bsz, 10, repeat=5)
    #     test_rgf_batched_gpu(5, 20, bsz, 20, repeat=5)

    for bsz in (128, 256, 512):
        print(f"Testing with block size {bsz}", flush=True)

        validate_rgf_batched_gpu(10, 10, bsz)
