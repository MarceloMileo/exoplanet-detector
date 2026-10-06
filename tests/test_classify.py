import numpy as np
import pytest

from exoplanet_detector.classify import (
    FALSE_POSITIVE,
    PLANET,
    TransitClassifier,
    analyze,
    synthetic_training_set,
)
from exoplanet_detector.features import FEATURE_NAMES
from exoplanet_detector.synthetic import random_eclipsing_binary, random_planet


@pytest.fixture(scope="module")
def classifier():
    features, labels = synthetic_training_set(n_per_class=25, seed=0, frequency_factor=5)
    return TransitClassifier(n_estimators=100).fit(features, labels)


def test_training_set_is_balanced():
    _, labels = synthetic_training_set(n_per_class=2, seed=0, frequency_factor=5)
    assert sorted(labels) == [FALSE_POSITIVE, FALSE_POSITIVE, PLANET, PLANET]


def test_classifies_unseen_examples(classifier):
    rng = np.random.default_rng(42)
    planets = [analyze(random_planet(rng), frequency_factor=5) for _ in range(10)]
    binaries = [analyze(random_eclipsing_binary(rng), frequency_factor=5) for _ in range(10)]
    accuracy = (
        np.sum(classifier.predict(planets) == PLANET)
        + np.sum(classifier.predict(binaries) == FALSE_POSITIVE)
    ) / 20
    assert accuracy >= 0.85


def test_predict_proba_is_probability(classifier):
    rng = np.random.default_rng(7)
    proba = classifier.predict_proba([analyze(random_planet(rng), frequency_factor=5)])
    assert proba.shape == (1,)
    assert 0.0 <= proba[0] <= 1.0


def test_feature_importances_cover_all_features(classifier):
    importances = classifier.feature_importances()
    assert list(importances) == list(FEATURE_NAMES)
    assert sum(importances.values()) == pytest.approx(1.0)
