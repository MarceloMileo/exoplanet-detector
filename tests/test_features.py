import numpy as np
import pytest

from exoplanet_detector.features import FEATURE_NAMES, extract_features, features_to_array
from exoplanet_detector.preprocess import preprocess
from exoplanet_detector.search import TransitCandidate
from exoplanet_detector.synthetic import FFI_CADENCE, make_lightcurve

PERIOD, T0, DURATION, DEPTH = 3.5, 1.0, 0.16, 0.01


def features_for(**kwargs):
    lc = make_lightcurve(
        period=PERIOD, t0=T0, duration=DURATION, depth=DEPTH, cadence=FFI_CADENCE, **kwargs
    )
    candidate = TransitCandidate(PERIOD, T0, DURATION, DEPTH, depth_snr=0.0, power=0.0)
    return extract_features(preprocess(lc), candidate)


def test_planet_looks_clean():
    f = features_for()
    assert f["depth"] == pytest.approx(DEPTH, rel=0.15)
    assert f["duration_ratio"] == pytest.approx(DURATION / PERIOD)
    assert f["odd_even_sigma"] < 3
    assert abs(f["secondary_sigma"]) < 3
    assert f["shape_ratio"] == pytest.approx(1.0, abs=0.1)


def test_detects_odd_even_difference():
    assert features_for(odd_even_ratio=0.5)["odd_even_sigma"] > 5


def test_detects_secondary_eclipse():
    assert features_for(secondary_depth=0.004)["secondary_sigma"] > 5


def test_detects_v_shape():
    assert features_for(shape="v")["shape_ratio"] == pytest.approx(2 / 3, abs=0.1)


def test_features_to_array_follows_feature_names():
    f = features_for()
    array = features_to_array([f, f])
    assert array.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_array_equal(array[0], [f[name] for name in FEATURE_NAMES])
