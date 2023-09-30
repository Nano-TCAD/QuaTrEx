import cupy as cp
import cupyx as cpx
import numpy as np


def _pin_memory(array):
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


def sinv_rgf_cpu(A: np.ndarray, block_size: int) -> np.ndarray:

    # Assumption: block_size divides A.shape[0] evenly
    num_blocks = A.shape[0] // block_size

    # Storage for the full backward substitution 
    B = np.empty_like(A)
    
    # 0. Inverse of the first block
    B[:block_size, :block_size] = np.linalg.inv(A[:block_size, :block_size])

    # 1. Forward substitution (performed left to right)
    for i in range(1, num_blocks, 1):

        l_slice = slice((i-1)*block_size, i*block_size)
        d_slice = slice(i*block_size, (i+1)*block_size)
        u_slice = slice((i+1)*block_size, (i+2)*block_size)

        B[d_slice, d_slice] = np.linalg.inv(A[d_slice, d_slice] -
                                            A[d_slice, l_slice] @
                                            B[l_slice, l_slice] @
                                            A[l_slice, d_slice])

    # 2. Backward substitution (performed right to left)
    for i in range(num_blocks - 2, -1, -1):

        l_slice = slice((i-1)*block_size, i*block_size)
        d_slice = slice(i*block_size, (i+1)*block_size)
        u_slice = slice((i+1)*block_size, (i+2)*block_size)

        lower_factor = B[u_slice, u_slice] @ A[u_slice, d_slice] @ B[d_slice, d_slice]
        B[u_slice, d_slice] = -lower_factor
        # Assumption: A is not symmetric
        tmp = B[d_slice, d_slice] @ A[d_slice, u_slice]
        B[d_slice, u_slice] = -tmp @ B[u_slice, u_slice]
        B[d_slice, d_slice] += tmp @ lower_factor
    
    return B


def sinv_rgf_gpu(A: np.ndarray, block_size: int) -> cp.ndarray:

    # Assumption: block_size divides A.shape[0] evenly
    num_blocks = A.shape[0] // block_size

    # Storage for the full backward substitution 
    # B = cpx.empty_like_pinned(A)
    B = [[cpx.empty_pinned((block_size, block_size), dtype=A.dtype) for _ in range(num_blocks)] for _ in range(num_blocks)]

    # Buffers
    A_dd = [cp.empty((block_size, block_size), dtype=A.dtype) for _ in range(2)]
    A_dl = [cp.empty((block_size, block_size), dtype=A.dtype) for _ in range(2)]
    A_ld = [cp.empty((block_size, block_size), dtype=A.dtype) for _ in range(2)]
    B_dd = [cp.empty((block_size, block_size), dtype=A.dtype) for _ in range(2)]
    # Streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    # Events
    events = [cp.cuda.Event() for _ in range(2)]
    
    # 0. Inverse of the first block
    with streams[0] as stream:
        A_dd[0].set(A[:block_size, :block_size])
        B_dd[0] = cp.linalg.inv(A_dd[0])
        events[0].record(stream=stream)
        B_dd[0].get(out=B[0][0])


    # 1. Forward substitution (performed left to right)
    for i in range(1, num_blocks, 1):

        l_slice = slice((i-1)*block_size, i*block_size)
        d_slice = slice(i*block_size, (i+1)*block_size)
        u_slice = slice((i+1)*block_size, (i+2)*block_size)

        with streams[i % 2] as stream:
            A_dd[i % 2].set(A[d_slice, d_slice])
            A_dl[i % 2].set(A[d_slice, l_slice])
            A_ld[i % 2].set(A[l_slice, d_slice])
            stream.wait_event(event=events[(i-1) % 2])
            B_dd[i % 2][:] = cp.linalg.inv(A_dd[i % 2] - A_dl[i % 2] @ B_dd[(i-1) % 2] @ A_ld[i % 2])
            events[i % 2].record(stream=stream)
            B_dd[i % 2].get(out=B[i][i])

    # # 2. Backward substitution (performed right to left)
    # for i in range(num_blocks - 2, -1, -1):

    #     l_slice = slice((i-1)*block_size, i*block_size)
    #     d_slice = slice(i*block_size, (i+1)*block_size)
    #     u_slice = slice((i+1)*block_size, (i+2)*block_size)

    #     lower_factor = B[u_slice, u_slice] @ A[u_slice, d_slice] @ B[d_slice, d_slice]
    #     B[u_slice, d_slice] = -lower_factor
    #     # Assumption: A is not symmetric
    #     tmp = B[d_slice, d_slice] @ A[d_slice, u_slice]
    #     B[d_slice, u_slice] = -tmp @ B[u_slice, u_slice]
    #     B[d_slice, d_slice] += tmp @ lower_factor

    streams[0].synchronize()
    streams[1].synchronize()
    
    return B


if __name__ == "__main__":

    pinned_memory_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

    rng = np.random.default_rng(0)

    block_size = 1024
    num_blocks = 10
    N = block_size * num_blocks
    A = cpx.empty_pinned((N, N), dtype=np.complex128)
    A[:] = rng.random((N, N)) + 1j * rng.random((N, N))

    B = sinv_rgf_gpu(A, block_size)
