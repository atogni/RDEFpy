"""The scripts in examples/ run cleanly."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
OBS = Path(__file__).parent / "data" / "EXMPn000tIsDS24r01c00-26039120000.obs"


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=tmp_path, capture_output=True, text=True, timeout=120)


# What each example must print: its result, not only "something". An example
# that says "samples equal: False" exits 0 too.
EXPECTED = {
    "carrier_synthetic.py": ["truth frequency before downconversion 8405000123.456700 Hz",
                             "2026-02-08T12:00:00.500000000  before DC 8405000123.4567"],  # block midpoint
    "downconversion.py": ["per-record [Hz]: [8405000000.000 8405000000.000]",
                          "first sample [Hz]: [8405000100.000 8405000094.000]",
                          "record mean  [Hz]: [8405000097.000 8405000091.000]",
                          "2026-02-08T12:00:01.500000000  8405000091.000 Hz"],
    "read_observation.py": ["ObservationFile EXMPn000tIsDS24r01c00-26039120000", "scans: 1", "products: 5"],
    "read_product_summary.py": ["records: 2", "record 1: 2026-02-08T12:00:01.000000000  2000 samples"],
    "samples_and_timetags.py": ["2026-02-08T12:00:01.000000000 (65535+1j)", "loop matches pf.samples: True"],
    "select_and_write.py": ["2 records -> 1 selected, first at 2026-02-08T12:00:01.000000000",
                            "samples equal: True", "bytes equal: True"],
    "stream_large_file.py": ["scan_product: 10 records, consistent length: True",
                             "read_headers: 2026-02-08T12:00:00.000000000 to 2026-02-08T12:00:09.000000000"],
}


@pytest.mark.parametrize(
    "script",
    sorted(p for p in EXAMPLES.glob("*.py") if p.name != "validate_observation.py"),
    ids=lambda p: p.name,
)
def test_example_runs(script, tmp_path):
    r = _run(tmp_path, script)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()
    assert script.name in EXPECTED, "new example: add the lines it must print to EXPECTED"
    for text in EXPECTED[script.name]:
        assert text in r.stdout
    if script.name == "carrier_synthetic.py":  # every block within 0.01 mHz of the truth it prints
        errors = [float(ln.split("error")[1].split()[0]) for ln in r.stdout.splitlines() if "error" in ln]
        assert len(errors) == 3 and max(map(abs, errors)) <= 0.01


def test_validate_observation(tmp_path):
    r = _run(tmp_path, EXAMPLES / "validate_observation.py", OBS)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"OK    {OBS}: 1 scans, 5 product files"


def test_validate_observation_reports_the_rule_broken(tmp_path):
    bad = tmp_path / "bad.obs"
    bad.write_bytes(OBS.read_bytes().replace(b"E *=END=*", b"E END"))
    r = _run(tmp_path, EXAMPLES / "validate_observation.py", OBS, bad)
    assert r.returncode == 1
    ok, fail = r.stdout.splitlines()
    assert ok.startswith("OK") and fail.startswith("FAIL  bad.obs:") and "end line is not 'E *=END=*'" in fail


def test_validate_observation_needs_a_file(tmp_path):
    r = _run(tmp_path, EXAMPLES / "validate_observation.py")
    assert r.returncode != 0
