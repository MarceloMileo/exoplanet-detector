"""Geração de curvas de luz sintéticas, úteis para testes e demonstrações."""

from __future__ import annotations

from typing import Literal

import numpy as np

from exoplanet_detector.lightcurve import LightCurve

TESS_CADENCE = 2.0 / 60 / 24  # 2 minutos, em dias
FFI_CADENCE = 30.0 / 60 / 24  # 30 minutos (imagens completas do TESS)
EXPOSURE_SUBSAMPLES = 9


def _eclipse_profile(
    phase: np.ndarray, duration: float, depth: float, shape: Literal["box", "v"]
) -> np.ndarray:
    """Queda relativa de fluxo para cada fase (0 fora do eclipse)."""
    x = np.abs(phase) / (duration / 2)
    profile = np.where(x < 1, 1.0, 0.0) if shape == "box" else np.clip(1 - x, 0, None)
    return depth * profile


def make_lightcurve(
    period: float | None = 3.5,
    t0: float = 1.0,
    duration: float = 0.12,
    depth: float = 0.005,
    baseline: float = 27.0,
    cadence: float = TESS_CADENCE,
    noise: float = 0.001,
    variability: float = 0.01,
    shape: Literal["box", "v"] = "box",
    odd_even_ratio: float = 1.0,
    secondary_depth: float = 0.0,
    seed: int | np.random.Generator | None = 0,
) -> LightCurve:
    """Simula uma estrela com variabilidade lenta, ruído e (opcionalmente) eclipses.

    Os padrões imitam um setor do TESS (27 dias, cadência de 2 minutos) com um
    planeta: trânsito de fundo chato e todos com a mesma profundidade.
    Use `period=None` para gerar uma estrela sem trânsito.

    Para simular uma binária eclipsante, use `shape="v"`, `odd_even_ratio`
    (profundidade dos eclipses ímpares relativa aos pares) e/ou
    `secondary_depth` (eclipse secundário em fase 0.5).

    Como num detector real, cada ponto é a média do fluxo durante a exposição
    (igual à cadência), o que suaviza as bordas de trânsitos curtos.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, baseline, cadence)
    flux = 1.0 + variability * np.sin(2 * np.pi * time / 6.3)

    if period is not None:

        def dip_at(t: np.ndarray) -> np.ndarray:
            shifted = t - t0 + 0.5 * period
            phase = shifted % period - 0.5 * period
            odd = np.floor(shifted / period) % 2 == 1
            primary_depth = np.where(odd, depth * odd_even_ratio, depth)
            dip = _eclipse_profile(phase, duration, 1.0, shape) * primary_depth
            if secondary_depth > 0:
                phase2 = (t - t0) % period - 0.5 * period
                dip += _eclipse_profile(phase2, duration, secondary_depth, shape)
            return dip

        offsets = ((np.arange(EXPOSURE_SUBSAMPLES) + 0.5) / EXPOSURE_SUBSAMPLES - 0.5) * cadence
        flux = flux * (1 - np.mean([dip_at(time + u) for u in offsets], axis=0))

    flux = flux + rng.normal(0.0, noise, time.size)
    return LightCurve(time, flux, np.full(time.size, noise))


def _common_params(rng: np.random.Generator) -> dict:
    period = rng.uniform(1.0, 9.0)
    return {
        "period": period,
        "t0": rng.uniform(0.0, period),
        "duration": rng.uniform(0.08, 0.25),
        "noise": rng.uniform(5e-4, 1.5e-3),
        "variability": rng.uniform(0.0, 0.02),
        "baseline": 27.0,
        "cadence": FFI_CADENCE,
        "seed": rng,
    }


def random_planet(rng: np.random.Generator) -> LightCurve:
    """Estrela com um planeta de parâmetros aleatórios (cadência de 30 min)."""
    return make_lightcurve(depth=rng.uniform(0.003, 0.015), shape="box", **_common_params(rng))


def random_eclipsing_binary(rng: np.random.Generator) -> LightCurve:
    """Binária eclipsante aleatória: o falso positivo clássico.

    Sempre tem eclipse em V e ao menos uma das assinaturas: eclipses
    alternados com profundidades diferentes ou eclipse secundário.
    """
    depth = rng.uniform(0.005, 0.03)
    kind = rng.integers(3)  # 0: odd/even, 1: secundário, 2: ambos
    odd_even_ratio = rng.uniform(0.3, 0.7) if kind != 1 else 1.0
    secondary_depth = depth * rng.uniform(0.2, 0.6) if kind != 0 else 0.0
    return make_lightcurve(
        depth=depth,
        shape="v",
        odd_even_ratio=odd_even_ratio,
        secondary_depth=secondary_depth,
        **_common_params(rng),
    )
