"""Classificador planeta vs. falso positivo a partir das features de vetting."""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from exoplanet_detector.features import FEATURE_NAMES, extract_features, features_to_array
from exoplanet_detector.lightcurve import LightCurve
from exoplanet_detector.preprocess import preprocess
from exoplanet_detector.search import search_transit
from exoplanet_detector.synthetic import random_eclipsing_binary, random_planet

PLANET, FALSE_POSITIVE = 1, 0


def analyze(lc: LightCurve, frequency_factor: float = 1.0) -> dict[str, float]:
    """Pipeline completo até as features: pré-processa, busca e extrai."""
    clean = preprocess(lc)
    candidate = search_transit(clean, frequency_factor=frequency_factor)
    return extract_features(clean, candidate)


def synthetic_training_set(
    n_per_class: int, seed: int = 0, frequency_factor: float = 3.0
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Gera features e rótulos de planetas e binárias eclipsantes simulados.

    As features passam pelo mesmo BLS usado na inferência, para que o modelo
    aprenda com os mesmos erros (ex.: binárias detectadas com metade do período).
    """
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for _ in range(n_per_class):
        features.append(analyze(random_planet(rng), frequency_factor))
        labels.append(PLANET)
        features.append(analyze(random_eclipsing_binary(rng), frequency_factor))
        labels.append(FALSE_POSITIVE)
    return features, np.array(labels)


def kepler_training_set() -> tuple[list[dict[str, float]], np.ndarray]:
    """Features e rótulos de estrelas reais do Kepler (tabela KOI).

    Gerado por `scripts/build_dataset.py` e distribuído com o pacote, para
    treinar o modelo sem baixar nenhuma curva de luz.
    """
    path = files("exoplanet_detector") / "resources" / "kepler_features.csv"
    with path.open() as f:
        dataset = pd.read_csv(f)
    features = dataset[list(FEATURE_NAMES)].to_dict("records")
    return features, dataset["label"].to_numpy()


class TransitClassifier:
    """Random Forest sobre as features de `extract_features`."""

    def __init__(self, n_estimators: int = 300, random_state: int | None = 0) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2, random_state=random_state
        )

    def fit(self, features: list[dict[str, float]], labels: np.ndarray) -> TransitClassifier:
        self.model.fit(features_to_array(features), labels)
        return self

    def predict_proba(self, features: list[dict[str, float]]) -> np.ndarray:
        """Probabilidade de cada candidato ser um planeta."""
        proba = self.model.predict_proba(features_to_array(features))
        return proba[:, list(self.model.classes_).index(PLANET)]

    def predict(self, features: list[dict[str, float]], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(int)

    def feature_importances(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.model.feature_importances_, strict=True))
