"""Select a time window of a product file, write it, and read it back.

    python examples/select_and_write.py
"""

import tempfile
from pathlib import Path

import numpy as np

from rdefpy import read_product, write_product

data = Path(__file__).parents[1] / "tests" / "data"
pf = read_product(data / "EXMPn001tSsDS24r01c01-26039120000.prd")
fs = pf[0].header.sample_rate

part = pf.select(start_epoch="2026-02-08T12:00:01", stop_epoch="2026-02-08T12:00:02")
print(f"{len(pf)} records -> {len(part)} selected, first at {part[0].header.timetag}")

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "part.prd"
    write_product(out, part)
    back = read_product(out)
    print(f"written {out.stat().st_size} bytes, read back {len(back)} record")
    print("samples equal:", np.array_equal(back.samples[:], pf.samples[fs:]))  # the second record

    # A selection of every record is byte-identical to the original file.
    write_product(out, pf.select())
    print("bytes equal:", out.read_bytes() == pf.path.read_bytes())
