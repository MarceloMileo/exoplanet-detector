"""Testes da camada de dados. A rede é sempre simulada (mocks)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from exoplanet_detector import data


def koi(kepid, disposition, ss=0, period=5.0):
    return {
        "kepid": kepid,
        "kepoi_name": f"K{kepid:05d}.01",
        "koi_disposition": disposition,
        "koi_fpflag_ss": ss,
        "koi_period": period,
    }


def test_labeled_kois_selects_reliable_labels():
    table = pd.DataFrame(
        [
            koi(1, "CONFIRMED"),  # planeta
            koi(2, "FALSE POSITIVE", ss=1),  # binária eclipsante
            koi(3, "FALSE POSITIVE", ss=0),  # outro tipo de falso positivo
            koi(4, "CANDIDATE"),  # sem rótulo definitivo
            koi(5, "CONFIRMED", period=200.0),  # período fora do intervalo
            koi(6, "CONFIRMED"),  # sistema múltiplo...
            koi(6, "CONFIRMED"),  # ...descartado
        ]
    )
    selected = data.labeled_kois(table)
    assert selected[["kepid", "label"]].values.tolist() == [[1, 1], [2, 0]]


def test_fetch_koi_table_uses_cache(tmp_path, monkeypatch):
    calls = []
    fake = pd.DataFrame([koi(1, "CONFIRMED")])
    real_read_csv = pd.read_csv

    def fake_read_csv(source, *args, **kwargs):
        if str(source).startswith("http"):
            calls.append(source)
            return fake
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(data.pd, "read_csv", fake_read_csv)
    cache = tmp_path / "koi.csv"

    first = data.fetch_koi_table(cache)
    second = data.fetch_koi_table(cache)

    assert len(calls) == 1
    assert "query=select" in calls[0] and "cumulative" in calls[0]
    pd.testing.assert_frame_equal(first, second)


class FakeSearch:
    """Imita o SearchResult do lightkurve com um produto por quarter."""

    def __init__(self, quarters):
        self.mission = [f"Kepler Quarter {q:02d}" for q in quarters]
        self.selected = None

    def __getitem__(self, mask):
        self.selected = [m for m, keep in zip(self.mission, mask, strict=True) if keep]
        return self

    def download_all(self, flux_column, download_dir):
        assert flux_column == "pdcsap_flux"
        time = np.arange(100.0)
        klc = SimpleNamespace(
            time=SimpleNamespace(value=time),
            flux=SimpleNamespace(value=np.ones(100)),
            flux_err=SimpleNamespace(value=np.full(100, 1e-3)),
        )
        klc.remove_nans = lambda: klc
        return SimpleNamespace(stitch=lambda: klc)


@pytest.fixture
def fake_lightkurve(monkeypatch):
    import lightkurve

    search = FakeSearch(quarters=[0, 1, 2, 3, 4, 5])
    queries = []

    def search_lightcurve(target, **kwargs):
        queries.append(target)
        return search

    monkeypatch.setattr(lightkurve, "search_lightcurve", search_lightcurve)
    return search, queries


def test_download_kepler_filters_quarters(fake_lightkurve, tmp_path):
    search, queries = fake_lightkurve
    lc = data.download_kepler(11904151, quarters=(1, 2), cache_dir=tmp_path)
    assert queries == ["KIC 11904151"]
    assert search.selected == ["Kepler Quarter 01", "Kepler Quarter 02"]
    assert len(lc) == 100


def test_download_kepler_without_data_raises(fake_lightkurve, tmp_path):
    with pytest.raises(LookupError):
        data.download_kepler(1, quarters=(17,), cache_dir=tmp_path)
