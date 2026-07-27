from __future__ import annotations

from scripts.control_compare import _get, _verdict


def test_control_comparison_paths_and_direction_verdicts():
    payload = {"counts": {"hit": 2, "total": 10}}
    assert _get(payload, "counts.hit") == 2
    assert _get(payload, "counts.hit//counts.total") == 0.2
    assert _verdict(0.1, 0.3, "lower") == "PASS"
    assert _verdict(0.3, 0.1, "lower") == "FAIL"
    assert _verdict(1.0, 1.01, "lower") == "TIE"
