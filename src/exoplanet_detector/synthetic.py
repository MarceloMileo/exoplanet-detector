"""Geração de curvas de luz sintéticas, úteis para testes e demonstrações."""

from __future__ import annotations

import numpy as np

from exoplanet_detector.lightcurve import LightCurve


def make_lightcurve(
    period: float | None = 3.5,
    t0: float = 1.0,
    duration: float = 0.12,
    depth: float = 0.005,
    baseline: float = 27.0,
    cadence: float = 2.0 / 60 / 24,
    noise: float = 0.001,
    variability: float = 0.01,
    seed: int | None = 0,
) -> LightCurve:
    """Simula uma estrela com variabilidade lenta, ruído e (opcionalmente) um planeta.

    Os padrões imitam um setor do TESS: 27 dias com cadência de 2 minutos.
    Use `period=None` para gerar uma estrela sem trânsito.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, baseline, cadence)
    flux = 1.0 + variability * np.sin(2 * np.pi * time / 6.3)

    if period is not None:
        phase = (time - t0 + 0.5 * period) % period - 0.5 * period
        flux = np.where(np.abs(phase) < duration / 2, flux * (1 - depth), flux)

    flux = flux + rng.normal(0.0, noise, time.size)
    return LightCurve(time, flux, np.full(time.size, noise))
