import csv
import math

from mimic_forecaster.data.mimic import FeatureSpec, prepare_stays


def _write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_hourly_median_masks_missing_cells_and_absent_feature(tmp_path):
    stays = tmp_path / "icustays.csv"
    charts = tmp_path / "chartevents.csv"
    _write(stays, [{
        "subject_id": "1", "stay_id": "10",
        "intime": "2020-01-01 00:10:00", "outtime": "2020-01-01 02:10:00",
    }])
    _write(charts, [
        {"stay_id": "10", "charttime": "2020-01-01 00:20:00", "itemid": "1", "valuenum": "70"},
        {"stay_id": "10", "charttime": "2020-01-01 00:50:00", "itemid": "1", "valuenum": "90"},
        {"stay_id": "10", "charttime": "2020-01-01 01:20:00", "itemid": "1", "valuenum": "999"},
    ])
    features = (
        FeatureSpec("heart_rate", (1,), 20, 260),
        FeatureSpec("optional_absent", (2,), 0, 100),
    )
    result = prepare_stays(stays, charts, features)
    assert len(result) == 1
    stay = result[0]
    assert len(stay.timestamps) == 3
    assert stay.values[0][0] == 80.0
    assert stay.observed_mask[0] == [1, 0]
    assert math.isnan(stay.values[1][0])
    assert stay.observed_mask[1] == [0, 0]
    assert stay.feature_present_mask == [1, 0]


def test_missing_required_schema_column_fails(tmp_path):
    stays = tmp_path / "icustays.csv"
    charts = tmp_path / "chartevents.csv"
    _write(stays, [{"subject_id": "1", "stay_id": "10", "intime": "2020-01-01 00:00:00"}])
    _write(charts, [{"stay_id": "10", "charttime": "2020-01-01 00:00:00", "itemid": "1", "valuenum": "1"}])
    try:
        prepare_stays(stays, charts)
    except ValueError as error:
        assert "outtime" in str(error)
    else:
        raise AssertionError("expected schema validation failure")

