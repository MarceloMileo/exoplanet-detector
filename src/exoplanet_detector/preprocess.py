"""Limpeza e detrending de curvas de luz antes da busca por trânsitos."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from exoplanet_detector.lightcurve import LightCurve


def remove_nans(lc: LightCurve) -> LightCurve:
    """Remove pontos com tempo, fluxo ou erro inválidos (NaN/inf)."""
    mask = np.isfinite(lc.time) & np.isfinite(lc.flux)
    if lc.flux_err is not None:
        mask &= np.isfinite(lc.flux_err)
    order = np.argsort(lc.time[mask])
    return lc.select(np.flatnonzero(mask)[order])


def normalize(lc: LightCurve) -> LightCurve:
    """Divide o fluxo pela mediana, deixando a linha de base em 1.0."""
    median = np.median(lc.flux)
    flux_err = None if lc.flux_err is None else lc.flux_err / median
    return LightCurve(lc.time, lc.flux / median, flux_err)


def flatten(lc: LightCurve, window_length: int = 401) -> LightCurve:
    """Remove a variabilidade lenta da estrela dividindo por uma mediana móvel.

    A janela deve ser bem maior que a duração do trânsito; caso contrário, o
    filtro "come" o próprio sinal que queremos detectar.
    """
    window_length = min(window_length, len(lc))
    trend = median_filter(lc.flux, size=window_length, mode="nearest")
    flux_err = None if lc.flux_err is None else lc.flux_err / trend
    return LightCurve(lc.time, lc.flux / trend, flux_err)


def clip_upper_outliers(lc: LightCurve, sigma: float = 4.0) -> LightCurve:
    """Remove apenas outliers para CIMA (raios cósmicos, flares).

    Outliers para baixo são preservados, pois os trânsitos são justamente quedas
    de fluxo.
    """
    median = np.median(lc.flux)
    mad_std = 1.4826 * np.median(np.abs(lc.flux - median))
    return lc.select(lc.flux < median + sigma * mad_std)


def preprocess(lc: LightCurve, window_length: int = 401, sigma: float = 4.0) -> LightCurve:
    """Pipeline padrão: limpa NaNs, normaliza, remove tendência e outliers."""
    lc = remove_nans(lc)
    lc = normalize(lc)
    lc = flatten(lc, window_length=window_length)
    return clip_upper_outliers(lc, sigma=sigma)
