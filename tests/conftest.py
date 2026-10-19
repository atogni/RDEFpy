"""Shared fixtures for the CCSDS 506.1-B-1 spec tests (test_spec_*.py).

``decode_path`` runs a test once per I/Q decoder: the Cython extension (when
built) and the NumPy reference. Both must meet the spec on their own, so a
green run on one does not excuse the other. A skipped ``[cython]`` case means
the extension is not built, so parity is NOT proven. Build it with
``uv sync`` (or ``pip install -e .``) and rebuild after editing the .pyx.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _require_fresh_build() -> None:
    """Fail if unpack_cyth.pyx is newer than the binary actually imported.

    An edited .pyx with a stale build would let every [cython] case test the
    old code and pass. Failing, not skipping: skips are what hid that before.
    """
    from rdefpy.prd import unpack_cyth

    built = Path(unpack_cyth.__file__)
    pyx = Path(__file__).parents[1] / "src" / "rdefpy" / "prd" / "unpack_cyth.pyx"
    if pyx.exists() and pyx.stat().st_mtime > built.stat().st_mtime:
        pytest.fail(
            f"{built.name} is older than unpack_cyth.pyx: rebuild with "
            "`uv sync --reinstall-package rdefpy` before trusting [cython] results"
        )


@pytest.fixture(params=["cython", "numpy"])
def decode_path(request, monkeypatch):
    """Force the reader onto one I/Q decoder for the duration of a test."""
    from rdefpy.prd import reader

    if request.param == "cython":
        if reader._unpack_fast is None:
            pytest.skip("unpack_cyth extension not built")
        _require_fresh_build()
    else:
        monkeypatch.setattr(reader, "_unpack_fast", None)
    return request.param


@pytest.fixture
def write_prd(tmp_path):
    """Write raw record bytes to a .prd file and return its path."""
    counter = iter(range(10**6))

    def _write(*records: bytes, name: str | None = None):
        path = tmp_path / (name or f"spec{next(counter)}.prd")
        path.write_bytes(b"".join(records))
        return path

    return _write
