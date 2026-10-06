import re

import pytest

from exoplanet_detector import cli
from exoplanet_detector.synthetic import FFI_CADENCE, make_lightcurve


def test_cli_reports_candidate(monkeypatch, capsys):
    def fake_download(kepid, quarters):
        assert (kepid, quarters) == (123, (1, 2))
        return make_lightcurve(period=2.5, depth=0.004, cadence=FFI_CADENCE, baseline=60.0)

    monkeypatch.setattr(cli, "download_kepler", fake_download)
    cli.main(["123", "--quarters", "1", "2"])

    out = capsys.readouterr().out
    period = float(re.search(r"período\s+([\d.]+) d", out).group(1))
    assert period == pytest.approx(2.5, rel=1e-3)
    assert "P(planeta) = " in out
