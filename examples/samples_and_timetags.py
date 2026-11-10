"""Read samples across records: slices of pf.samples and pf.timetags, and a record loop.

    python examples/samples_and_timetags.py
"""

from pathlib import Path

import numpy as np

from rdefpy import read_product

data = Path(__file__).parents[1] / "tests" / "data"
pf = read_product(data / "EXMPn001tSsDS24r01c01-26039120000.prd")
fs = pf[0].header.sample_rate

# A slice that crosses the boundary between record 0 and record 1.
x = pf.samples[fs - 3 : fs + 3]
t = pf.timetags[fs - 3 : fs + 3]
for ti, xi in zip(t, x):
    print(ti, xi)

# Record by record, decoding straight into one reused complex buffer.
buf = None
total = 0.0
for rec in pf:
    buf = rec.read_samples(out=buf)
    total += np.sum(np.abs(buf) ** 2)
print(f"mean power {10 * np.log10(total / (len(pf) * fs)):.2f} dB")

whole = pf.samples[:]
print("loop matches pf.samples:", np.isclose(total, np.sum(np.abs(whole) ** 2)))
