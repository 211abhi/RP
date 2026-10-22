import numpy as np, torch
from mimic_forecaster.cli.predict import forecast, resolve_keep
from mimic_forecaster.data.mimic import DEFAULT_FEATURES, HourlyStay
from mimic_forecaster.data.preprocessing import RobustStats
from datetime import datetime, timedelta


class Persistence(torch.nn.Module):
    """Stands in for a trained model whose residual heads output zero."""
    def forward(self, values, observed, present, age):
        return {"forecast": values[:, -1:, :], "impute": values[:, -1, :],
                "drug_logits": torch.zeros(len(values), 3)}


def _stay(hours, features):
    values = [[float(h + i) for i in range(features)] for h in range(hours)]
    return HourlyStay("1", "1", [datetime(2020, 1, 1) + timedelta(hours=h) for h in range(hours)],
                      tuple(f.name for f in DEFAULT_FEATURES[:features]), values,
                      [[1] * features for _ in range(hours)], [1] * features)


def test_forecast_returns_original_units():
    features = 3
    stats = RobustStats(np.array([2.0, 3.0, 4.0], np.float32), np.array([2.0, 4.0, 5.0], np.float32))
    out = forecast(Persistence(), _stay(30, features), stats, DEFAULT_FEATURES[:features], context=24, max_gap=6)
    # A zero residual over the last observed bin must round-trip the scaling exactly.
    assert out["forecast"].shape == (1, features)
    assert np.allclose(out["forecast"][0], [29.0, 30.0, 31.0], atol=1e-4)
    assert out["drug_probability"].shape == (3,)


def test_withheld_column_is_zeroed_before_it_reaches_the_model():
    """The imputed column must be reconstructed, not read back out of the input."""
    features = 3
    stats = RobustStats(np.zeros(features, np.float32), np.ones(features, np.float32))
    keep = np.array([1.0, 0.0, 1.0], np.float32)
    out = forecast(Persistence(), _stay(30, features), stats, DEFAULT_FEATURES[:features],
                   context=24, max_gap=6, keep=keep)
    # Persistence echoes its input, so a hidden column must come back at the median (0), and
    # a supplied column at its true last value.
    assert out["impute"][1] == 0.0
    assert np.allclose([out["impute"][0], out["impute"][2]], [29.0, 31.0], atol=1e-4)


def test_resolve_keep_rejects_an_unknown_column():
    names = [f.name for f in DEFAULT_FEATURES]
    assert resolve_keep(names, None).tolist() == [1.0] * len(names)
    assert resolve_keep(names, ["heart_rate", "spo2"]).tolist()[:3] == [1.0, 0.0, 1.0]
    try:
        resolve_keep(names, ["blood_unicorns"])
    except SystemExit as error:
        assert "blood_unicorns" in str(error); return
    raise AssertionError("an unknown column must fail loudly")


def test_forecast_rejects_short_history():
    try:
        forecast(Persistence(), _stay(10, 3), RobustStats(np.zeros(3, np.float32), np.ones(3, np.float32)),
                 DEFAULT_FEATURES[:3], context=24, max_gap=6)
    except ValueError as error:
        assert "needs 24" in str(error); return
    raise AssertionError("short history must fail loudly")


def test_load_patient_csv_bins_hours_and_marks_missing_columns(tmp_path):
    from mimic_forecaster.cli.predict import load_patient_csv
    rows = ["time,heart_rate,spo2"] + [f"2024-01-01 {h % 24:02d}:10,{80 + h},97" for h in range(24)]
    rows += ["2024-01-01 23:40,100,", "2024-01-01 05:30,999,96"]  # duplicate hour, out-of-range value
    path = tmp_path / "patient.csv"; path.write_text("\n".join(rows))
    stay, supplied = load_patient_csv(path, DEFAULT_FEATURES, min_hours=24)
    assert supplied == ["heart_rate", "spo2"] and len(stay.timestamps) == 24
    assert stay.feature_present_mask == [1, 0, 1, 0, 0, 0, 0]
    assert stay.values[23][0] == 101.5 and stay.values[5][0] == 85.0  # median of 103,100; 999 dropped
    try:
        load_patient_csv(path, DEFAULT_FEATURES, min_hours=48)
    except SystemExit as error:
        assert "48" in str(error); return
    raise AssertionError("a too-short history must fail loudly")
