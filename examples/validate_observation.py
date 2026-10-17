"""Check RDEF observation files against CCSDS 506.1-B-1 §4.

    python examples/validate_observation.py FILE.obs [FILE.obs ...]

Prints OK or the first rule broken for each file; exits 1 if any fails.
"""

import sys

from rdefpy import read_observation

failed = False
for path in sys.argv[1:]:
    try:
        obs = read_observation(path)
    except (ValueError, OSError) as e:
        print(f"FAIL  {e}")
        failed = True
    else:
        n = sum(len(s.products) for s in obs)
        print(f"OK    {path}: {len(obs)} scans, {n} product files")

sys.exit(failed)
