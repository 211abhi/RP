"""Schema differences between MIMIC releases must be absorbed, not adapted around."""
import pytest
from mimic_forecaster.data.mimic import FeatureSpec, config_sources, load_stay_bounds, collect_hourly_events

HR = (FeatureSpec("heart_rate", (220045, 211), 20.0, 260.0),)


def _write(tmp_path, name, rows):
    path = tmp_path / name; path.write_text("\n".join(rows) + "\n"); return path


def test_stay_id_column_alias(tmp_path):
    """MIMIC-III spells it icustay_id, MIMIC-IV spells it stay_id."""
    for column in ("stay_id", "icustay_id"):
        path = _write(tmp_path, f"{column}.csv", [
            f"subject_id,{column},intime,outtime",
            "7,100,2130-01-01 00:00:00,2130-01-01 05:00:00"])
        assert load_stay_bounds(path) == {"100": ("7", *load_stay_bounds(path)["100"][1:])}


def test_missing_stay_column_fails_loudly(tmp_path):
    path = _write(tmp_path, "bad.csv", ["subject_id,intime,outtime", "7,2130-01-01 00:00:00,2130-01-01 05:00:00"])
    with pytest.raises(ValueError, match="no stay id column"):
        load_stay_bounds(path)


def test_carevue_and_metavision_merge_into_one_feature(tmp_path):
    """Different charting systems, same canonical column -- the point of the registry."""
    path = _write(tmp_path, "chartevents.csv", [
        "icustay_id,charttime,itemid,valuenum",
        "100,2130-01-01 00:30:00,220045,80",   # metavision
        "100,2130-01-01 00:45:00,211,90",      # carevue, same hour and feature
    ])
    buckets, seen = collect_hourly_events(path, HR, {"100"})
    assert len(buckets) == 1 and sorted(next(iter(buckets.values()))) == [80.0, 90.0]
    assert seen == {220045, 211}


def test_error_flagged_rows_are_dropped(tmp_path):
    """MIMIC-III carries an error column; MIMIC-IV dropped it. Honour it where present."""
    path = _write(tmp_path, "chartevents.csv", [
        "icustay_id,charttime,itemid,valuenum,error",
        "100,2130-01-01 00:30:00,211,90,0",
        "100,2130-01-01 00:45:00,211,250,1",   # flagged bad by the source
    ])
    buckets, _ = collect_hourly_events(path, HR, {"100"})
    assert next(iter(buckets.values())) == [90.0]


def test_out_of_range_alarm_limits_are_rejected(tmp_path):
    """A value in the right unit and a plausible range is still not a measurement."""
    path = _write(tmp_path, "chartevents.csv", [
        "icustay_id,charttime,itemid,valuenum",
        "100,2130-01-01 00:30:00,211,15",      # below the 20 bpm floor
    ])
    buckets, _ = collect_hourly_events(path, HR, {"100"})
    assert not buckets


def test_subject_ids_are_namespaced_per_source(tmp_path):
    """Subject 10006 in MIMIC-III is not subject 10006 in MIMIC-IV."""
    path = _write(tmp_path, "icustays.csv", [
        "subject_id,stay_id,intime,outtime",
        "10006,100,2130-01-01 00:00:00,2130-01-01 05:00:00"])
    assert load_stay_bounds(path, "mimic-iii-demo")["100"][0] == "mimic-iii-demo:10006"
    assert load_stay_bounds(path)["100"][0] == "10006"


def test_config_sources_accepts_both_config_shapes():
    single = {"data": {"root": "r", "icustays": "i", "chartevents": "c"}}
    assert config_sources(single) == [{"name": "", "root": "r", "icustays": "i", "chartevents": "c"}]
    pooled = {"data": {"sources": [{"name": "a", "root": "r", "icustays": "i", "chartevents": "c"}]}}
    assert config_sources(pooled)[0]["name"] == "a"
    with pytest.raises(ValueError, match="unique"):
        config_sources({"data": {"sources": [{"name": "a"}, {"name": "a"}]}})
