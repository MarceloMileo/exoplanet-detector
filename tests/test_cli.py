from exoplanet_detector import cli
from exoplanet_detector.synthetic import FFI_CADENCE, make_lightcurve


def test_cli_reports_candidate(monkeypatch, capsys):
    def fake_download(kepid, quarters):
        assert (kepid, quarters) == (123, (1, 2))
        return make_lightcurve(period=2.5, depth=0.004, cadence=FFI_CADENCE, baseline=60.0)

    monkeypatch.setattr(cli, "download_kepler", fake_download)
    cli.main(["123", "--quarters", "1", "2"])

    out = capsys.readouterr().out
    assert "período       2.5" in out
    assert "P(planeta) = " in out
