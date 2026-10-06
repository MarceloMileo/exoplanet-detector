import numpy as np

from exoplanet_detector import LightCurve, preprocess
from exoplanet_detector.preprocess import clip_upper_outliers, remove_nans
from exoplanet_detector.synthetic import make_lightcurve


def test_remove_nans_and_sort():
    lc = LightCurve(np.array([3.0, 1.0, np.nan, 2.0]), np.array([1.0, np.nan, 1.0, 1.0]))
    clean = remove_nans(lc)
    np.testing.assert_array_equal(clean.time, [2.0, 3.0])


def test_clip_keeps_dips_and_removes_spikes():
    flux = np.ones(100) + np.random.default_rng(1).normal(0, 1e-3, 100)
    flux[10] = 1.1  # raio cósmico
    flux[50] = 0.95  # queda (pode ser trânsito)
    clipped = clip_upper_outliers(LightCurve(np.arange(100.0), flux))
    assert 1.1 not in clipped.flux
    assert 0.95 in clipped.flux


def test_preprocess_removes_stellar_variability():
    lc = make_lightcurve(period=None, variability=0.02, noise=1e-4)
    flat = preprocess(lc)
    assert abs(np.median(flat.flux) - 1) < 1e-4
    assert np.std(flat.flux) < 5e-4


def test_flatten_keeps_noise_on_smooth_variability():
    # Variabilidade forte e lenta com pouco ruído: trechos monótonos longos.
    # Uma mediana móvel pura devolveria fluxo == 1.0 exato e "apagaria" o ruído.
    lc = make_lightcurve(period=None, variability=0.05, noise=1e-5, cadence=30 / 60 / 24)
    flat = preprocess(lc)
    assert len(flat) > 0.95 * len(lc)
    assert np.mean(flat.flux == 1.0) < 0.01
