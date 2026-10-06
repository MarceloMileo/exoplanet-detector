import numpy as np
import pytest

from exoplanet_detector.features import (
    FEATURE_NAMES,
    extract_features,
    features_to_array,
    fit_trapezoid,
    trapezoid_model,
)
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
    assert f["ingress_ratio"] < 0.2


def test_detects_odd_even_difference():
    assert features_for(odd_even_ratio=0.5)["odd_even_sigma"] > 5


def test_detects_secondary_eclipse():
    assert features_for(secondary_depth=0.004)["secondary_sigma"] > 5


def test_detects_v_shape():
    assert features_for(shape="v")["ingress_ratio"] > 0.8


def test_trapezoid_model_conserves_transit_area():
    # A exposição espalha o trânsito no tempo, mas não muda a luz "perdida".
    phase = np.linspace(-0.5, 0.5, 20001)
    sharp = trapezoid_model(phase, depth=0.01, duration=0.015, ingress_ratio=0.1)
    smeared = trapezoid_model(phase, 0.01, 0.015, 0.1, exposure=FFI_CADENCE)
    assert np.sum(1 - smeared) == pytest.approx(np.sum(1 - sharp), rel=1e-3)
    # Trânsito (22 min) mais curto que a exposição (30 min): o fundo fica raso.
    assert smeared.min() > sharp.min()


@pytest.mark.parametrize("shape, expected", [("box", 0.0), ("v", 1.0)])
def test_fit_trapezoid_short_transit_long_exposure(shape, expected):
    # Trânsito de 1.9 h com exposições de 30 min: o caso da Kepler-10b.
    duration = 1.9 / 24
    lc = make_lightcurve(
        period=0.8375,
        t0=0.3,
        duration=duration,
        depth=0.002,
        noise=2e-4,
        shape=shape,
        cadence=FFI_CADENCE,
        baseline=90.0,
        variability=0.0,
    )
    candidate = TransitCandidate(0.8375, 0.3, 0.08, 0.002, depth_snr=0.0, power=0.0)
    fit = fit_trapezoid(preprocess(lc), candidate)
    assert fit.ingress_ratio == pytest.approx(expected, abs=0.15)
    assert fit.duration == pytest.approx(duration, rel=0.1)


def test_features_to_array_follows_feature_names():
    f = features_for()
    array = features_to_array([f, f])
    assert array.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_array_equal(array[0], [f[name] for name in FEATURE_NAMES])
