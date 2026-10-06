"""Acesso a dados reais: tabela KOI (NASA Exoplanet Archive) e curvas do Kepler (MAST)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from exoplanet_detector.lightcurve import LightCurve

KOI_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
KOI_COLUMNS = (
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
)
DEFAULT_CACHE = Path("data")


def fetch_koi_table(cache_path: Path | None = DEFAULT_CACHE / "koi_cumulative.csv") -> pd.DataFrame:
    """Baixa a tabela KOI "cumulative" (uma linha por objeto de interesse do Kepler).

    O resultado é salvo em `cache_path` e reutilizado nas próximas chamadas.
    """
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path)
    query = f"select {','.join(KOI_COLUMNS)} from cumulative"
    table = pd.read_csv(f"{KOI_TAP_URL}?{urlencode({'query': query, 'format': 'csv'})}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_path, index=False)
    return table


def labeled_kois(
    table: pd.DataFrame, min_period: float = 0.5, max_period: float = 30.0
) -> pd.DataFrame:
    """Seleciona KOIs com rótulo confiável para treinar o classificador.

    - Apenas estrelas com um único KOI: em sistemas múltiplos o BLS acha só o
      sinal mais forte, e o rótulo pode não corresponder a ele.
    - Planeta (label=1): disposição CONFIRMED.
    - Falso positivo (label=0): FALSE POSITIVE com a flag de eclipse estelar
      (`koi_fpflag_ss`), ou seja, binárias eclipsantes. Outros falsos positivos
      (ex.: deslocamento de centroide) não são visíveis só na curva de luz.
    """
    single = table[table.groupby("kepid")["kepid"].transform("size") == 1]
    in_range = single["koi_period"].between(min_period, max_period)
    planet = single["koi_disposition"] == "CONFIRMED"
    binary = (single["koi_disposition"] == "FALSE POSITIVE") & (single["koi_fpflag_ss"] == 1)
    selected = single[in_range & (planet | binary)].copy()
    selected["label"] = planet[selected.index].astype(int)
    return selected.reset_index(drop=True)


def download_kepler(
    kepid: int,
    quarters: tuple[int, ...] = (1, 2, 3, 4),
    cache_dir: Path | None = DEFAULT_CACHE / "lightkurve",
) -> LightCurve:
    """Baixa as curvas de cadência longa (30 min) do Kepler e junta os quarters.

    Usa o fluxo PDCSAP, já corrigido de efeitos instrumentais pela missão. A
    busca é pelo ID KIC; buscar por nome (ex.: "Kepler-10") exige o serviço de
    resolução de nomes do MAST.
    """
    import lightkurve as lk  # import pesado: só quando necessário

    search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler", exptime=1800)
    wanted = [int(mission.split()[-1]) in quarters for mission in search.mission]
    if not any(wanted):
        raise LookupError(f"KIC {kepid}: nenhuma curva nos quarters {quarters}")
    download_dir = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        download_dir = str(cache_dir)
    collection = search[np.array(wanted)].download_all(
        flux_column="pdcsap_flux", download_dir=download_dir
    )
    klc = collection.stitch().remove_nans()
    return LightCurve(
        np.asarray(klc.time.value),
        np.asarray(klc.flux.value),
        np.asarray(klc.flux_err.value),
    )
