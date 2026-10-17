# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""CCSDS Delta-DOR Raw Data Exchange Format (RDEF) reader.

Reads the ASCII observation files (.obs) of CCSDS 506.1-B-1 (June 2013), §4.

    from rdefpy import read_observation

    obs = read_observation("session.obs")
    print(obs)

Product-file (.prd, §5) support is not yet published.
"""

from rdefpy.obs import ObservationFile, ProductEntry, Scan, read_observation

__all__ = [
    # Observation files (.obs, §4)
    "ObservationFile",
    "ProductEntry",
    "Scan",
    "read_observation",
]
