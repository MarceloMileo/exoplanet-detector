import pytest

from exoplanet_detector import preprocess, search_transit
from exoplanet_detector.synthetic import FFI_CADENCE, make_lightcurve


@pytest.mark.parametrize("period", [1.7, 3.5, 8.2])
def test_recovers_injected_period(period):
    lc = preprocess(make_lightcurve(period=period, t0=0.6, depth=0.004))
    candidate = search_transit(lc)
    assert candidate.period == pytest.approx(period, rel=0.01)
    assert candidate.depth == pytest.approx(0.004, rel=0.25)
    assert candidate.depth_snr > 10


def test_star_without_planet_has_weak_signal():
    with_planet = search_transit(preprocess(make_lightcurve(period=3.5, depth=0.004)))
    without = search_transit(preprocess(make_lightcurve(period=None)))
    assert without.depth_snr < with_planet.depth_snr / 3


def test_rejects_too_short_series():
    lc = preprocess(make_lightcurve(baseline=0.8, period=None))
    with pytest.raises(ValueError):
        search_transit(lc)


def test_two_stage_search_matches_exhaustive_search():
    lc = preprocess(
        make_lightcurve(period=5.3, t0=2.0, depth=0.003, baseline=60.0, cadence=FFI_CADENCE)
    )
    fast = search_transit(lc, coarse_factor=20)
    exhaustive = search_transit(lc, coarse_factor=None)
    assert fast.period == pytest.approx(exhaustive.period, rel=1e-3)
    assert fast.power == pytest.approx(exhaustive.power, rel=0.05)
