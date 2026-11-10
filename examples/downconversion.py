"""Downconversion frequency and phase of a product file (§5.2.6).

    python examples/downconversion.py
"""

from pathlib import Path

import numpy as np

from rdefpy import read_product

np.set_printoptions(formatter={"float": "{:.3f}".format})
data = Path(__file__).parents[1] / "tests" / "data"
pf = read_product(data / "EXMPn001tSsDS24r01c01-26039120000.prd")
fs = pf[0].header.sample_rate

# The file has a fixed downconversion: all phase coefficients are zero.
print("per-record [Hz]:", pf.record_downconversion_frequency)

# Give the headers one frequency ramp, f_var = C1 + 2*C2*t from the first record's
# start (headers only, the samples are untouched). Each record continues it from its
# own start t = n s: c1 is the frequency there, and the phase reached so far is split
# into whole turns (channel_accumulated_phase) and the fraction c0, so phase and
# frequency are continuous from record to record (Table 5-1).
C1, C2 = 100.0, -3.0  # turns/s, turns/s^2
steered = pf.copy()
for n, rec in enumerate(steered):
    turns = C1 * n + C2 * n**2  # phase at the start of record n
    rec.correct_header(
        channel_accumulated_phase=float(np.floor(turns)),
        channel_phase_polynomial_coefficient_0=float(turns - np.floor(turns)),
        channel_phase_polynomial_coefficient_1=C1 + 2 * C2 * n,
        channel_phase_polynomial_coefficient_2=C2,
    )

print("first sample [Hz]:", steered.record_downconversion_frequency)
print("record mean  [Hz]:", steered.mean_downconversion_frequency)
print("phase at first sample [turns]:", steered.record_downconversion_phase)

# Per sample, across the record boundary: continuous.
print("per-sample [Hz]:   ", steered.downconversion_frequency[fs - 1 : fs + 1])
print("per-sample [turns]:", steered.downconversion_phase[fs - 1 : fs + 1])

# Tag each mean at its record midpoint, t0 + T/2 (not the mean of the sample times).
half = np.timedelta64(500_000_000, "ns")  # one-second records
for rec, f in zip(steered, steered.mean_downconversion_frequency):
    print(f"{rec.header.timetag + half}  {f:.3f} Hz")
