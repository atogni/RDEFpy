"""rdefpy against the published sample files in tests/reference/.

``make_reference.py`` copies each test product file there as
``sample_<bits>bit.prd`` and writes its .hdr/.tab/.hex from ``rdef_spec``
alone. These tests check (1) the committed files are still what the spec
transcription produces, and (2) rdefpy reads the same headers and samples
from the sample files, on both decoders.
"""

from pathlib import Path

import numpy as np
import pytest

import make_reference as ref
import rdef_spec as spec
from rdefpy import read_product

PRDS = sorted(ref.DATA.glob("*.prd"))
IDS = [ref.name(p) for p in PRDS]


def test_one_sample_set_per_bit_depth():
    assert sorted(IDS) == sorted(f"sample_{b}bit" for b in spec.VALID_SAMPLE_SIZES)


@pytest.mark.parametrize("prd", PRDS, ids=IDS)
def test_sample_prd_is_the_test_file(prd):
    assert (ref.OUT / (ref.name(prd) + ".prd")).read_bytes() == prd.read_bytes()


@pytest.mark.parametrize("prd", PRDS, ids=IDS)
@pytest.mark.parametrize("ext", ref.RENDER)
def test_committed_file_matches_spec(prd, ext):
    """Regenerating gives the committed text: nobody edited it by hand."""
    committed = (ref.OUT / (ref.name(prd) + ext)).read_text(encoding="ascii")
    assert committed == ref.RENDER[ext](prd)


def _read_tab(path: Path):
    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
    recs = {}
    for row in rows:
        recs.setdefault(int(row[0]), []).append(row[2:].reshape(-1, 2))
    return {r: np.concatenate(p) for r, p in recs.items()}


@pytest.mark.parametrize("prd", PRDS, ids=IDS)
def test_rdefpy_samples_match_tab(prd, decode_path):
    stem = ref.OUT / ref.name(prd)
    tab = _read_tab(stem.with_suffix(".tab"))
    pf = read_product(stem.with_suffix(".prd"))
    assert sorted(tab) == list(range(1, len(pf) + 1))
    for n, rec in enumerate(pf, 1):
        np.testing.assert_array_equal(rec.data.i_samples, tab[n][:, 0])
        np.testing.assert_array_equal(rec.data.q_samples, tab[n][:, 1])


@pytest.mark.parametrize("prd", PRDS, ids=IDS)
def test_rdefpy_headers_match_hdr(prd):
    sample = (ref.OUT / ref.name(prd)).with_suffix(".prd")
    expected = []  # one {field: text} per "Record n of N" block of the .hdr
    for line in sample.with_suffix(".hdr").read_text(encoding="ascii").splitlines():
        if line.startswith("Record "):
            expected.append({})
        elif " = " in line:
            field, value = line.split(" = ")
            expected[-1][field.strip()] = value
    pf = read_product(sample)
    assert len(pf) == len(expected)
    for rec, want in zip(pf, expected):
        assert set(spec.MODEL_FIELDS) <= set(want)
        for field in spec.MODEL_FIELDS:
            got = getattr(rec.header, field)
            if spec.LAYOUT[field][2] == "x":  # reserved bytes, written as make_reference._field does
                got = bytes(got)
                assert want[field] == (got.hex(" ") if any(got) else f"{len(got)} bytes, all zero"), field
            else:
                assert float(got) == float(want[field]), field


# =============================================================================
# Downconversion / agency-block cases (make_reference.CASES)
# =============================================================================

CASES = list(ref.CASES)
# f_DC near 8.4 GHz: float64 spacing there is 2**-20 Hz = 9.5e-7 Hz; rdefpy
# and rdef_spec sum the same terms in a different order, so allow about 4 spacings.
F_TOL_HZ = 4e-6
# Phase up to ~2.6e3 turns: spacing 4.5e-13 turns; c1*tau etc. add a few roundings.
PHASE_TOL_TURNS = 1e-9


def _read_dc(path: Path) -> np.ndarray:
    """Rows (f_first, f_mean, phase_first), one per record."""
    lines = path.read_text(encoding="ascii").splitlines()
    return np.array([[float(v) for v in ln.split()[1:]] for ln in lines if ln and not ln.startswith("#")])


def test_case_files_are_listed():
    assert CASES == ["steered_ps0", "dsn_agency_block"]


@pytest.mark.parametrize("case", CASES)
def test_case_prd_is_what_rdef_spec_builds(case):
    assert (ref.OUT / case).with_suffix(".prd").read_bytes() == ref.case_bytes(case)


@pytest.mark.parametrize("case", CASES)
def test_committed_case_files_match_spec(case):
    prd = (ref.OUT / case).with_suffix(".prd")
    for ext, render in ref.case_render(case).items():
        assert prd.with_suffix(ext).read_text(encoding="ascii") == render(prd), ext


@pytest.mark.parametrize("case", CASES)
def test_rdefpy_downconversion_matches_dc(case):
    prd = (ref.OUT / case).with_suffix(".prd")
    want = _read_dc(prd.with_suffix(".dc"))
    pf = read_product(prd)
    first = np.array([rec.downconversion_frequency[0] for rec in pf])
    np.testing.assert_allclose(pf.record_downconversion_frequency, want[:, 0], rtol=0, atol=F_TOL_HZ)
    np.testing.assert_allclose(first, want[:, 0], rtol=0, atol=F_TOL_HZ)
    np.testing.assert_allclose(pf.mean_downconversion_frequency, want[:, 1], rtol=0, atol=F_TOL_HZ)
    np.testing.assert_allclose(pf.record_downconversion_phase, want[:, 2], rtol=0, atol=PHASE_TOL_TURNS)


def _picosecond_warnings(path, **kw) -> int:
    """How many times read_product warns about picoseconds beyond DSN 0222's range."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_product(path, **kw)
    return sum("DSN 0222" in str(w.message) and issubclass(w.category, RuntimeWarning) for w in caught)


@pytest.mark.parametrize("case, n_warn", [("steered_ps0", 0), ("dsn_agency_block", 0)])
def test_auto_picosecond_warning_count(case, n_warn):
    assert _picosecond_warnings((ref.OUT / case).with_suffix(".prd")) == n_warn


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("ps, n_warn", [(100_000.0, 0), (100_001.0, 1), (999_999_999_999.0, 1)])
def test_auto_picosecond_warning_fires_above_the_0222_range(case, ps, n_warn, write_prd):
    """The positive control of the count above: the same file with every
    record's picoseconds moved to DSN 0222's limit (Table 3-1: 0 to 100,000 ps)
    and past it. One warning per read, not one per record; none once
    t0_offset is passed."""
    raw = bytearray(ref.case_bytes(case))
    off, size, _ = spec.LAYOUT["time_tag_picoseconds_of_the_second"]
    pos, n_records = 0, 0
    while pos < len(raw):
        raw[pos + off : pos + off + size] = np.float64(ps).astype("<f8").tobytes()
        pos += spec.parse_header(raw[pos : pos + spec.HEADER_SIZE])["record_length"]
        n_records += 1
    assert n_records > 1
    path = write_prd(bytes(raw))
    assert _picosecond_warnings(path) == n_warn
    assert _picosecond_warnings(path, t0_offset=0.0) == 0


def test_rdefpy_agency_fields_match_agency():
    prd = (ref.OUT / "dsn_agency_block").with_suffix(".prd")
    want = {}
    for line in prd.with_suffix(".agency").read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            left, value = line.split(" = ")
            rec, field = left.split()
            want.setdefault(int(rec), {})[field] = value
    rows = read_product(prd).agency_fields()  # agency_flag 3, layout "auto"
    assert sorted(want) == list(range(1, len(rows) + 1))
    for n, row in enumerate(rows, 1):
        for field, value in want[n].items():
            if field == "spare":
                assert value == "19 bytes, all zero" and not any(bytes(row[field]))
            else:
                assert float(row[field]) == float(value), (n, field)


def test_changed_sample_size_on_disk_record_raises(tmp_path):
    from rdefpy import write_product
    pf = read_product(ref.OUT / "sample_16bit.prd")
    rec = pf[0]
    assert rec._data is None  # samples on disk: the case that misread
    rec.correct_header(sample_size=8, sample_rate=2 * int(rec.header.sample_rate))
    for touch in (lambda: write_product(tmp_path / "x.prd", pf), lambda: rec.data, rec.read_samples, lambda: len(rec)):
        with pytest.raises(ValueError, match="on disk"):
            touch()
