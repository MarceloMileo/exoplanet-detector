"""Features que separam planetas de falsos positivos (diagnósticos de "vetting").

Todas são calculadas a partir da curva pré-processada e do candidato do BLS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from exoplanet_detector.lightcurve import LightCurve
from exoplanet_detector.search import TransitCandidate

FEATURE_NAMES = (
    "depth",
    "depth_snr",
    "duration_ratio",
    "odd_even_sigma",
    "secondary_sigma",
    "ingress_ratio",
)
EXPOSURE_SUBSAMPLES = 11


@dataclass(frozen=True)
class TrapezoidFit:
    """Trapézio ajustado ao trânsito dobrado em fase."""

    depth: float
    duration: float  # dias, do primeiro ao último contato
    ingress_ratio: float  # duração das rampas / metade da duração: 0 = caixa, 1 = V
    t0_offset: float  # correção do centro do trânsito em relação ao BLS (dias)


def _phase(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Fase em dias, centrada no trânsito: valores em [-period/2, period/2)."""
    return (time - t0 + 0.5 * period) % period - 0.5 * period


def _depth(flux: np.ndarray, mask: np.ndarray, noise: float) -> tuple[float, float]:
    """Profundidade média nos pontos de `mask` e sua incerteza."""
    n = int(mask.sum())
    if n == 0:
        return np.nan, np.nan
    return float(1.0 - flux[mask].mean()), noise / np.sqrt(n)


def trapezoid_model(
    phase: np.ndarray,
    depth: float,
    duration: float,
    ingress_ratio: float,
    t0_offset: float = 0.0,
    exposure: float = 0.0,
) -> np.ndarray:
    """Fluxo relativo de um trânsito trapezoidal, integrado sobre a exposição.

    Cada ponto observado é a média do fluxo durante `exposure` dias; sem essa
    integração, trânsitos curtos medidos com exposições longas pareceriam ter
    rampas (formato em V) mesmo quando o fundo é chato.
    """
    half = duration / 2
    ingress = max(ingress_ratio * half, 1e-9)
    offsets = ((np.arange(EXPOSURE_SUBSAMPLES) + 0.5) / EXPOSURE_SUBSAMPLES - 0.5) * exposure
    x = np.abs(phase[:, None] + offsets[None, :] - t0_offset)
    dip = depth * np.clip((half - x) / ingress, 0.0, 1.0)
    return 1.0 - dip.mean(axis=1)


def fit_trapezoid(lc: LightCurve, candidate: TransitCandidate) -> TrapezoidFit:
    """Ajusta um trapézio ao trânsito do candidato, levando em conta a cadência.

    O tempo de exposição é estimado pela cadência mediana (o Kepler e o TESS
    integram continuamente entre um ponto e o próximo).
    """
    period, duration = candidate.period, candidate.duration
    exposure = float(np.median(np.diff(lc.time)))
    phase = _phase(lc.time, period, candidate.t0)
    window = np.abs(phase) < min(2 * duration + exposure, period / 2)
    phase, flux = phase[window], lc.flux[window]

    def residuals(params: np.ndarray) -> np.ndarray:
        return trapezoid_model(phase, *params, exposure=exposure) - flux

    initial = [max(candidate.depth, 1e-6), duration, 0.3, 0.0]
    lower = [0.0, exposure / 4, 0.0, -duration / 2]
    upper = [1.0, min(3 * duration, period / 2), 1.0, duration / 2]
    initial = np.clip(initial, lower, upper)
    result = least_squares(residuals, initial, bounds=(lower, upper), loss="soft_l1", f_scale=0.01)
    return TrapezoidFit(*(float(v) for v in result.x))


def extract_features(lc: LightCurve, candidate: TransitCandidate) -> dict[str, float]:
    """Calcula as features de vetting de um candidato.

    - depth, depth_snr: força do sinal.
    - duration_ratio: duração / período. Eclipses longos demais para a órbita
      sugerem estrelas grandes, não planetas.
    - odd_even_sigma: diferença entre trânsitos ímpares e pares, em sigmas. Uma
      binária detectada com metade do período alterna primário e secundário.
    - secondary_sigma: significância de uma queda em fase 0.5 (eclipse
      secundário), que planetas pequenos praticamente não produzem.
    - ingress_ratio: fração do trânsito ocupada pelas rampas de entrada e saída,
      de um trapézio ajustado já descontando a exposição. Planetas pequenos têm
      fundo chato (~0); binárias rasantes têm formato em V (~1).
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

    return {
        "depth": depth,
        "depth_snr": depth / depth_err,
        "duration_ratio": candidate.duration / period,
        "odd_even_sigma": float(odd_even_sigma),
        "secondary_sigma": d_sec / e_sec,
        "ingress_ratio": fit_trapezoid(lc, candidate).ingress_ratio,
    }


def features_to_array(features: list[dict[str, float]]) -> np.ndarray:
    """Converte uma lista de dicionários de features numa matriz (n, n_features).

    Valores infinitos viram NaN, que o classificador trata como ausentes.
    """
    array = np.array([[f[name] for name in FEATURE_NAMES] for f in features], dtype=float)
    return np.where(np.isfinite(array), array, np.nan)
