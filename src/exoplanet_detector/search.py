"""Busca de trânsitos periódicos com Box Least Squares (BLS)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares

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


def search_transit(
    lc: LightCurve,
    min_period: float = 0.5,
    max_period: float | None = None,
    durations: tuple[float, ...] = DEFAULT_DURATIONS,
    frequency_factor: float = 1.0,
) -> TransitCandidate:
    """Roda o BLS na curva (já pré-processada) e retorna o pico mais forte.

    Por padrão, `max_period` é metade da duração da série, garantindo ao menos
    dois trânsitos observados. `frequency_factor` > 1 deixa a grade de períodos
    mais grossa: a busca fica mais rápida, com risco de errar picos estreitos.
    """
    baseline = lc.time.max() - lc.time.min()
    if max_period is None:
        max_period = baseline / 2
    if max_period <= min_period:
        raise ValueError("série curta demais para o intervalo de períodos pedido")

    model = BoxLeastSquares(lc.time, lc.flux, dy=lc.flux_err)
    periods = model.autoperiod(
        durations,
        minimum_period=min_period,
        maximum_period=max_period,
        frequency_factor=frequency_factor,
    )
    result = model.power(periods, durations, objective="snr")

    best = int(np.argmax(result.power))
    return TransitCandidate(
        period=float(result.period[best]),
        t0=float(result.transit_time[best]),
        duration=float(result.duration[best]),
        depth=float(result.depth[best]),
        depth_snr=float(result.depth_snr[best]),
        power=float(result.power[best]),
    )
