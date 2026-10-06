"""Avalia o classificador em estrelas reais do Kepler.

Uso:
    uv run python scripts/evaluate.py

Compara dois modelos no dataset real (src/exoplanet_detector/resources/):
1. treinado só com dados sintéticos e testado nas estrelas reais;
2. treinado com dados reais, via validação cruzada estratificada (5 folds).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from exoplanet_detector.classify import (
    TransitClassifier,
    kepler_training_set,
    synthetic_training_set,
)


def report(name: str, labels: np.ndarray, proba: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    print(f"ROC AUC: {roc_auc_score(labels, proba):.3f}")
    print(
        classification_report(
            labels, proba >= 0.5, target_names=["falso positivo", "planeta"], digits=3
        )
    )


def main() -> None:
    features, labels = kepler_training_set()
    print(f"Dataset real: {len(labels)} estrelas ({labels.sum()} planetas)")

    synthetic = TransitClassifier().fit(*synthetic_training_set(n_per_class=150))
    report("Treinado em sintéticos -> testado no Kepler", labels, synthetic.predict_proba(features))

    proba = np.zeros(len(labels))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train, test in folds.split(np.zeros(len(labels)), labels):
        model = TransitClassifier().fit([features[i] for i in train], labels[train])
        proba[test] = model.predict_proba([features[i] for i in test])
    report("Treinado no Kepler (validação cruzada, 5 folds)", labels, proba)

    importances = TransitClassifier().fit(features, labels).feature_importances()
    print("Importância das features (modelo real):")
    for name, value in sorted(importances.items(), key=lambda item: -item[1]):
        print(f"  {name:16s} {value:.3f}")


if __name__ == "__main__":
    main()
