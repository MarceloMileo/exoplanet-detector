import numpy as np
import pytest

from exoplanet_detector import LightCurve


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        LightCurve(np.arange(10), np.ones(9))


def test_select_keeps_flux_err():
    lc = LightCurve(np.arange(5), np.ones(5), np.full(5, 0.1))
    sub = lc.select(np.array([True, False, True, False, True]))
    assert len(sub) == 3
    assert sub.flux_err is not None and sub.flux_err.size == 3
