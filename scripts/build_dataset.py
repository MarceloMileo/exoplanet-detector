"""Monta o dataset de features a partir de curvas reais do Kepler.

Uso:
    uv run python scripts/build_dataset.py --n-per-class 150 --workers 4

Baixa as curvas (com cache em data/), roda o pipeline completo em cada estrela
e salva features + rótulos em src/exoplanet_detector/resources/kepler_features.csv.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_detector.data import download_kepler, fetch_koi_table, labeled_kois
from exoplanet_detector.features import extract_features
from exoplanet_detector.preprocess import preprocess
from exoplanet_detector.search import search_transit

DEFAULT_OUT = Path("src/exoplanet_detector/resources/kepler_features.csv")


def period_matches(found: float, expected: float, tolerance: float = 0.01) -> bool:
    """O BLS achou o período do catálogo (ou metade/dobro, aliases comuns)?"""
    return any(abs(found / (expected * k) - 1) < tolerance for k in (0.5, 1.0, 2.0))


def process(row: dict, coarse_factor: float) -> dict:
    warnings.filterwarnings("ignore")
    logging.getLogger("lightkurve").setLevel(logging.ERROR)
    lc = preprocess(download_kepler(int(row["kepid"])))
    candidate = search_transit(lc, max_period=30.0, coarse_factor=coarse_factor)
    return {
        "kepid": int(row["kepid"]),
        "kepoi_name": row["kepoi_name"],
        "label": int(row["label"]),
        "koi_period": row["koi_period"],
        "bls_period": candidate.period,
        "period_match": period_matches(candidate.period, row["koi_period"]),
        **extract_features(lc, candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--coarse-factor", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    kois = labeled_kois(fetch_koi_table())
    sample = kois.groupby("label").sample(n=args.n_per_class, random_state=args.seed)
    logging.info("%d estrelas selecionadas (%d por classe)", len(sample), args.n_per_class)

    rows, failures = [], 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(process, row, args.coarse_factor): row["kepoi_name"]
            for row in sample.to_dict("records")
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - uma estrela ruim não para o lote
                failures += 1
                logging.warning("%s falhou: %s", futures[future], exc)
            if i % 20 == 0:
                logging.info("%d/%d processadas", i, len(futures))

    dataset = pd.DataFrame(rows).sort_values("kepoi_name")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False, float_format="%.6g")
    logging.info(
        "%d linhas salvas em %s (%d falhas); BLS achou o período do catálogo em %.0f%%",
        len(dataset),
        args.out,
        failures,
        100 * np.mean(dataset["period_match"]),
    )


if __name__ == "__main__":
    main()
