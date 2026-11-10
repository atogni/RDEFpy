"""Summarise an RDEF product file: file info, first record header, power per record.

    python examples/read_product_summary.py
"""

from pathlib import Path

from rdefpy import read_product

data = Path(__file__).parents[1] / "tests" / "data"
pf = read_product(data / "EXMPn001tSsDS24r01c01-26039120000.prd")

pf.info()
print()
print(pf[0].header)

for i, rec in enumerate(pf):
    d = rec.data  # take it once: each access of an unloaded record may re-read the disk
    print(f"record {i}: {rec.header.timetag}  {len(d)} samples  {d.power_db:.2f} dB  rms {d.rms:.1f}")
