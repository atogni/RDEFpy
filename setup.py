"""Extension build: the Cython I/Q unpacker (rdefpy.prd.unpack_cyth).

Everything else is declared in pyproject.toml; setuptools needs this file only
because ext_modules cannot be expressed there. The extension is optional: with
no compiler the build continues and the reader uses its NumPy path.

Set RDEFPY_AVX2=1 for an AVX2/FMA binary (local builds only -- a wheel built
that way crashes with an illegal instruction on pre-2013 CPUs).
"""
import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

avx2 = bool(os.environ.get("RDEFPY_AVX2"))
if sys.platform == "win32":
    args = ["/O2"] + (["/arch:AVX2"] if avx2 else [])
else:
    args = ["-O3"] + (["-mavx2", "-mfma"] if avx2 else [])

setup(
    ext_modules=cythonize(
        [
            Extension(
                "rdefpy.prd.unpack_cyth",
                ["src/rdefpy/prd/unpack_cyth.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=args,
                optional=True,
            ),
        ],
        compiler_directives={"language_level": "3"},
        # Always regenerate the C: it is specific to the NumPy headers of the
        # build environment, and a stale unpack_cyth.c from another NumPy
        # fails to compile (seen: 2.5-generated C under a 3.11/NumPy 2.4 build).
        force=True,
    )
)
