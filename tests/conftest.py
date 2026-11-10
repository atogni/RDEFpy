"""Shared fixtures for the test modules (spec tests, reference files, readers).

Every session first checks that a built unpack_cyth is not older than its
.pyx (``_fresh_build``), whichever tests run.

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
    stale binary and pass. It fails rather than skips, so it cannot go unnoticed.
    """
    from rdefpy.prd import unpack_cyth

    built = Path(unpack_cyth.__file__)
    pyx = Path(__file__).parents[1] / "src" / "rdefpy" / "prd" / "unpack_cyth.pyx"
    if pyx.exists() and pyx.stat().st_mtime > built.stat().st_mtime:
        pytest.fail(
            f"{built.name} is older than unpack_cyth.pyx: rebuild with "
            "`uv sync --reinstall-package rdefpy` before trusting [cython] results"
        )


def _clobber(path):
    """Replace the file with different bytes of a different size."""
    path.write_bytes(b"\0" * (path.stat().st_size + 4))


@pytest.fixture(scope="session", autouse=True)
def _fresh_build():
    """Any run, even of one module, fails on a stale built extension."""
    try:
        from rdefpy.prd import unpack_cyth  # noqa: F401
    except ImportError:  # not built: the NumPy paths only, nothing stale
        return
    _require_fresh_build()


@pytest.fixture(params=["cython", "numpy"])
def decode_path(request, monkeypatch):
    """Force the reader onto one I/Q decoder for the duration of a test."""
    from rdefpy.prd import classes, reader

    if request.param == "cython":
        if reader._unpack_fast is None:
            pytest.skip("unpack_cyth extension not built")
        _require_fresh_build()
    else:
        monkeypatch.setattr(reader, "_unpack_fast", None)
        monkeypatch.setattr(reader, "_complex_fast", None)
        monkeypatch.setattr(classes, "_iq_to_complex", None)
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
