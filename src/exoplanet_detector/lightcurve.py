"""Estrutura básica de uma curva de luz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LightCurve:
    """Série temporal de fluxo de uma estrela.

    `time` em dias (ex.: BJD - 2454833 no Kepler), `flux` em unidades arbitrárias
    e `flux_err` opcional com a incerteza de cada ponto.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        if time.shape != flux.shape or time.ndim != 1:
            raise ValueError("time e flux devem ser vetores 1D do mesmo tamanho")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "flux", flux)
        if self.flux_err is not None:
            flux_err = np.asarray(self.flux_err, dtype=float)
            if flux_err.shape != flux.shape:
                raise ValueError("flux_err deve ter o mesmo tamanho de flux")
            object.__setattr__(self, "flux_err", flux_err)

    def __len__(self) -> int:
        return self.time.size

    def select(self, mask: np.ndarray) -> LightCurve:
        """Retorna uma nova curva apenas com os pontos onde `mask` é verdadeiro."""
        flux_err = None if self.flux_err is None else self.flux_err[mask]
        return LightCurve(self.time[mask], self.flux[mask], flux_err)
