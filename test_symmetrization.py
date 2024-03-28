import cupy as cp
import cupyx as cpx
import numpy as np
import scipy.sparse as sp
import time

from quatrex.block_tri_solvers.rgf_GF_GPU_combo import map_to_mapping, self_energy_preprocess_2d


def random_complex(shape, rng: np.random.Generator):
    result = cpx.empty_pinned(shape, dtype=np.complex128)
    result[:] = rng.random(shape) + 1j * rng.random(shape)
    return result


if __name__ == "__main__":

    small_folder = '/users/ziogasal/QuaTrEx/test_data/small'
    large_folder = '/users/ziogasal/QuaTrEx/test_data/large'

    simulations = {
        "small": {
            "folder": small_folder,
            "num_energies": 64,
            "num_blocks": 13,
            "block_size": 416,
            "nnz": 491040
        },
        "large": {
            "folder": large_folder,
            "num_energies": 32,
            "num_blocks": 18,
            "block_size": 416,
            "nnz": 3149584
        },
    }

    simulation = simulations["large"]

    map_diag = np.load(f"{simulation['folder']}/map_diag.npy")
    map_upper = np.load(f"{simulation['folder']}/map_upper.npy")
    map_lower = np.load(f"{simulation['folder']}/map_lower.npy")
    rows = np.load(f"{simulation['folder']}/rows.npy")
    columns = np.load(f"{simulation['folder']}/columns.npy")
    ij2ji = np.load(f"{simulation['folder']}/ij2ji.npy")

    num_energies = simulation["num_energies"]
    num_blocks = simulation["num_blocks"]
    block_size = simulation["block_size"]
    nnz = simulation["nnz"]
    matrix_size = num_blocks * block_size
    assert simulation["nnz"] == len(rows)

    mapping_diag = map_to_mapping(map_diag, num_blocks)
    mapping_upper = map_to_mapping(map_upper, num_blocks - 1)
    mapping_lower = map_to_mapping(map_lower, num_blocks - 1)

    mapping_diag_dev = cp.asarray(mapping_diag)
    mapping_upper_dev = cp.asarray(mapping_upper)
    mapping_lower_dev = cp.asarray(mapping_lower)
    rows_dev = cp.asarray(rows)
    columns_dev = cp.asarray(columns)
    ij2ji_dev = cp.asarray(ij2ji)

    rng = np.random.default_rng(0)

    sr_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    sl_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    sg_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    sr_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))
    sl_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))
    sg_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))

    self_energy_preprocess_2d(sl_dev, sg_dev, sr_dev, sl_phn_dev, sg_phn_dev, sr_phn_dev,
                              rows_dev, columns_dev, ij2ji_dev)
