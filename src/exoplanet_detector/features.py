"""Features que separam planetas de falsos positivos (diagnósticos de "vetting").

Todas são calculadas a partir da curva pré-processada e do candidato do BLS.
"""

from __future__ import annotations

import numpy as np

from exoplanet_detector.lightcurve import LightCurve
from exoplanet_detector.search import TransitCandidate

FEATURE_NAMES = (
    "depth",
    "depth_snr",
    "duration_ratio",
    "odd_even_sigma",
    "secondary_sigma",
    "shape_ratio",
)


def _phase(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Fase em dias, centrada no trânsito: valores em [-period/2, period/2)."""
    return (time - t0 + 0.5 * period) % period - 0.5 * period


def _depth(flux: np.ndarray, mask: np.ndarray, noise: float) -> tuple[float, float]:
    """Profundidade média nos pontos de `mask` e sua incerteza."""
    n = int(mask.sum())
    if n == 0:
        return np.nan, np.nan
    return float(1.0 - flux[mask].mean()), noise / np.sqrt(n)


def extract_features(lc: LightCurve, candidate: TransitCandidate) -> dict[str, float]:
    """Calcula as features de vetting de um candidato.

    - depth, depth_snr: força do sinal.
    - duration_ratio: duração / período. Eclipses longos demais para a órbita
      sugerem estrelas grandes, não planetas.
    - odd_even_sigma: diferença entre trânsitos ímpares e pares, em sigmas. Uma
      binária detectada com metade do período alterna primário e secundário.
    - secondary_sigma: significância de uma queda em fase 0.5 (eclipse
      secundário), que planetas pequenos praticamente não produzem.
    - shape_ratio: profundidade média no trânsito todo / no miolo. Fundo chato
      (planeta) dá ~1; formato em V (binária rasante) dá ~0.67.
    """
    period, t0, half = candidate.period, candidate.t0, candidate.duration / 2
    phase = _phase(lc.time, period, t0)
    in_transit = np.abs(phase) < half
    out_of_transit = lc.flux[~in_transit]
    noise = 1.4826 * float(np.median(np.abs(out_of_transit - 1.0)))
    if noise == 0:  # curva com resíduo degenerado: o MAD não serve
        noise = float(np.std(out_of_transit))

    depth, depth_err = _depth(lc.flux, in_transit, noise)

    epoch = np.floor((lc.time - t0 + 0.5 * period) / period)
    odd = epoch % 2 == 1
    d_odd, e_odd = _depth(lc.flux, in_transit & odd, noise)
    d_even, e_even = _depth(lc.flux, in_transit & ~odd, noise)
    odd_even_sigma = abs(d_odd - d_even) / np.hypot(e_odd, e_even)

    secondary = np.abs(_phase(lc.time, period, t0 + 0.5 * period)) < half
    d_sec, e_sec = _depth(lc.flux, secondary, noise)

    d_core, _ = _depth(lc.flux, np.abs(phase) < half / 2, noise)

    return {
        "depth": depth,
        "depth_snr": depth / depth_err,
        "duration_ratio": candidate.duration / period,
        "odd_even_sigma": float(odd_even_sigma),
        "secondary_sigma": d_sec / e_sec,
        "shape_ratio": depth / d_core if d_core > 0 else np.nan,
    }


def features_to_array(features: list[dict[str, float]]) -> np.ndarray:
    """Converte uma lista de dicionários de features numa matriz (n, n_features).

    Valores infinitos viram NaN, que o classificador trata como ausentes.
    """
    array = np.array([[f[name] for name in FEATURE_NAMES] for f in features], dtype=float)
    return np.where(np.isfinite(array), array, np.nan)
