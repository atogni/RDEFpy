"""File names against CCSDS 506.1-B-1 §6.2.

    MMMMnNNNtTsSSSSrRRcCC-YYDDDHHMMSS.XXX

checked on the bundled product-file fixtures.
The pattern is 37 characters long. Table 4-2's "39 ASCII characters" does not
add up, so nothing here asserts 39.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import rdef_spec as spec

DATA = Path(__file__).parent / "data"


def test_pattern_length():
    assert len("MMMMnNNNtTsSSSSrRRcCC-YYDDDHHMMSS.XXX") == spec.FILENAME_LENGTH == 37


@pytest.mark.parametrize("path", sorted(DATA.iterdir()), ids=lambda p: p.name)
def test_fixture_names_follow_section_6(path):
    m = spec.FILENAME_RE.match(path.name)
    assert m, path.name
    if m["ext"] == "obs":  # §6.2.2 e, §6.2.5: type I, scan 000, channel 00
        assert (m["type"], m["scan"], m["channel"]) == ("I", "000", "00")
    else:
        assert m["type"] in "SQ"  # §6.2.2 e: spacecraft or quasar
