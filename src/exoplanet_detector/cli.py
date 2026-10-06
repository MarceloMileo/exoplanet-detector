"""Linha de comando: analisa uma estrela do Kepler pelo ID KIC.

exoplanet-detector 11904151
"""

from __future__ import annotations

import argparse
import logging
import warnings

from exoplanet_detector.classify import TransitClassifier, kepler_training_set
from exoplanet_detector.data import download_kepler
from exoplanet_detector.features import extract_features
from exoplanet_detector.preprocess import preprocess
from exoplanet_detector.search import search_transit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exoplanet-detector",
        description="Procura trânsitos numa estrela do Kepler e estima se é um planeta.",
    )
    parser.add_argument("kepid", type=int, help="ID KIC da estrela (ex.: 11904151 = Kepler-10)")
    parser.add_argument(
        "--quarters", type=int, nargs="+", default=[1, 2, 3, 4], help="quarters do Kepler"
    )
    parser.add_argument("--max-period", type=float, default=30.0, help="maior período (dias)")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.getLogger("lightkurve").setLevel(logging.ERROR)

    print(f"Baixando KIC {args.kepid} (quarters {args.quarters})...")
    lc = preprocess(download_kepler(args.kepid, quarters=tuple(args.quarters)))
    candidate = search_transit(lc, max_period=args.max_period)
    features = extract_features(lc, candidate)
    classifier = TransitClassifier().fit(*kepler_training_set())
    probability = classifier.predict_proba([features])[0]

    print(f"\n{len(lc)} pontos, {lc.time.max() - lc.time.min():.0f} dias de observação")
    print(f"  período       {candidate.period:.5f} d")
    print(f"  duração       {candidate.duration * 24:.2f} h")
    print(f"  profundidade  {candidate.depth * 1e6:.0f} ppm (SNR {candidate.depth_snr:.1f})")
    print(f"  odd/even      {features['odd_even_sigma']:.1f} sigma")
    print(f"  secundário    {features['secondary_sigma']:.1f} sigma")
    print(f"  ingresso      {features['ingress_ratio']:.2f} (0 = fundo chato, 1 = V)")
    verdict = "PLANETA" if probability >= 0.5 else "FALSO POSITIVO (binária eclipsante?)"
    print(f"\nP(planeta) = {probability:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
