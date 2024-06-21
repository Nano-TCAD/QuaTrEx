#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
            name = "QuaTrEx",
            packages = setuptools.find_packages(),
            install_requires=["scipy==1.12.0",
                        "toml",
                            "cupy==12.3.0",
                                "mpi4py==3.1.5",
                                    "dace",
                                        "argparse",
                                                "mkl-service",
                                                    "numba",
                                                        "sympy",
                                                        "matplotlib",
                                                        "h5py"


            ])
