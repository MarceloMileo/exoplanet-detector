"""Busca de trânsitos periódicos com Box Least Squares (BLS)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.signal import find_peaks

from exoplanet_detector.lightcurve import LightCurve

DEFAULT_DURATIONS = (0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.33)  # dias


@dataclass(frozen=True)
class TransitCandidate:
    """Melhor sinal de trânsito encontrado pelo BLS."""

    period: float  # dias
    t0: float  # tempo do centro do primeiro trânsito (dias)
    duration: float  # dias
    depth: float  # queda relativa de fluxo
    depth_snr: float  # profundidade / incerteza da profundidade
    power: float  # valor do periodograma BLS no pico


def _best(result) -> TransitCandidate:
    best = int(np.argmax(result.power))
    return TransitCandidate(
        period=float(result.period[best]),
        t0=float(result.transit_time[best]),
        duration=float(result.duration[best]),
        depth=float(result.depth[best]),
        depth_snr=float(result.depth_snr[best]),
        power=float(result.power[best]),
    )


def _top_peaks(power: np.ndarray, n_peaks: int) -> np.ndarray:
    """Índices dos `n_peaks` máximos locais mais altos do periodograma."""
    peaks, _ = find_peaks(power)
    peaks = np.union1d(peaks, [int(np.argmax(power))])
    return peaks[np.argsort(power[peaks])[::-1][:n_peaks]]


def search_transit(
    lc: LightCurve,
    min_period: float = 0.5,
    max_period: float | None = None,
    durations: tuple[float, ...] = DEFAULT_DURATIONS,
    frequency_factor: float = 1.0,
    coarse_factor: float | None = 20.0,
    n_peaks: int = 5,
) -> TransitCandidate:
    """Roda o BLS na curva (já pré-processada) e retorna o pico mais forte.

    Por padrão, `max_period` é metade da duração da série, garantindo ao menos
    dois trânsitos observados.

    O número de períodos testados cresce com o quadrado da duração da série
    (milhões para um ano de Kepler), então a busca é feita em duas etapas: uma
    grade grossa (`coarse_factor` vezes mais esparsa) encontra os `n_peaks`
    melhores picos, que são refinados com a grade fina (`frequency_factor`).
    Use `coarse_factor=None` para a busca exaustiva em uma etapa.
    """
    baseline = lc.time.max() - lc.time.min()
    if max_period is None:
        max_period = baseline / 2
    if max_period <= min_period:
        raise ValueError("série curta demais para o intervalo de períodos pedido")

    model = BoxLeastSquares(lc.time, lc.flux, dy=lc.flux_err)

    def bls(low: float, high: float, factor: float):
        periods = model.autoperiod(
            durations, minimum_period=low, maximum_period=high, frequency_factor=factor
        )
        return model.power(periods, durations, objective="snr")

    if coarse_factor is None or coarse_factor <= frequency_factor:
        return _best(bls(min_period, max_period, frequency_factor))

    coarse = bls(min_period, max_period, coarse_factor)
    last = coarse.period.size - 1
    candidates = []
    for i in _top_peaks(np.asarray(coarse.power), n_peaks):
        neighbors = coarse.period[max(i - 1, 0)], coarse.period[min(i + 1, last)]
        candidates.append(_best(bls(min(neighbors), max(neighbors), frequency_factor)))
    return max(candidates, key=lambda c: c.power)
