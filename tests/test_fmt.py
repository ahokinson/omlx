from __future__ import annotations

from omlx._fmt import human_params


def test_human_params_billions_round():
    assert human_params(20_900_000_000) == "21b"
    assert human_params(8_000_000_000) == "8b"
    assert human_params(1_000_000_000) == "1b"


def test_human_params_millions():
    assert human_params(700_000_000) == "700m"
    assert human_params(1_000_000) == "1m"


def test_human_params_small_count_verbatim():
    assert human_params(512) == "512"
