import numpy as np
import time

from quatrex.block_tri_solvers.rgf_GF_GPU import rgf_standaloneGF_GPU, rgf_standaloneGF_batched_GPU


def random_complex(shape, rng: np.random.Generator):
    return rng.random(shape) + 1j * rng.random(shape)


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

            rgf_standaloneGF_GPU(
                HD[:, ie],
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
                bmin,
                bmax
            )
        
        runtimes[r] = time.time() - start
    
    print(f"Median non-batched runtime: {np.median(runtimes):.3f} s")

    return


def test_rgf_batched_gpu(num_blocks, num_energies, block_size, batch_size, repeat=10):

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

        for ie in range(0, num_energies, batch_size):
            ie_end = min(ie + batch_size, num_energies)

            rgf_standaloneGF_batched_GPU(
                HD[:, ie:ie_end],
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
                bmin,
                bmax
            )

        runtimes[r] = time.time() - start
    
    print(f"Median batched runtime (batch size {batch_size}): {np.median(runtimes):.3f} s")

    return


if __name__ == "__main__":

    test_rgf_gpu(5, 20, 416)
    test_rgf_batched_gpu(5, 20, 416, 1)
    test_rgf_batched_gpu(5, 20, 416, 10)
