"""Prescription ingestion: free-text drug names, release aliases, interval expansion."""
from datetime import datetime, timedelta

import pytest

from mimic_forecaster.data.prescriptions import (
    DrugSpec, classify_drug, collect_drug_intervals, drug_hour_matrix, load_drug_registry)

SPECS = (DrugSpec("vancomycin", ("vancomycin",)),
         DrugSpec("opioid", ("morphine", "fentanyl")))


def _write(tmp_path, name, rows):
    path = tmp_path / name; path.write_text("\n".join(rows) + "\n"); return path


def test_free_text_variants_collapse_to_one_class():
    for text in ("Vancomycin", "vancomycin hcl", "VANCOMYCIN ORAL LIQUID", " Vancomycin  "):
        assert classify_drug(text, SPECS) == 0
    assert classify_drug("Morphine Sulfate", SPECS) == 1
    assert classify_drug("Fentanyl Citrate", SPECS) == 1
    assert classify_drug("acetaminophen", SPECS) is None
    assert classify_drug("", SPECS) is None


def test_first_match_wins_so_specific_classes_must_come_first():
    ordered = (DrugSpec("heparin", ("heparin",)), DrugSpec("saline", ("sodium chloride 0.9",)))
    assert classify_drug("Heparin Flush (10 units/ml)", ordered) == 0
    assert classify_drug("Sodium Chloride 0.9%  Flush", ordered) == 1


def test_start_stop_column_aliases(tmp_path):
    """MIMIC-IV spells them starttime/stoptime, MIMIC-III startdate/enddate."""
    for start, stop in (("starttime", "stoptime"), ("startdate", "enddate")):
        path = _write(tmp_path, f"{start}.csv", [
            f"hadm_id,drug,drug_type,{start},{stop}",
            "900,Vancomycin,MAIN,2130-01-01 00:00:00,2130-01-01 03:00:00"])
        intervals, seen = collect_drug_intervals(path, SPECS)
        assert seen == {0}
        assert intervals["900"] == [(0, datetime(2130, 1, 1), datetime(2130, 1, 1, 3))]


def test_missing_start_column_fails_loudly(tmp_path):
    path = _write(tmp_path, "bad.csv", ["hadm_id,drug,stoptime", "900,Vancomycin,2130-01-01 00:00:00"])
    with pytest.raises(ValueError, match="no prescription start"):
        collect_drug_intervals(path, SPECS)


def test_additive_rows_are_not_a_second_decision(tmp_path):
    path = _write(tmp_path, "p.csv", [
        "hadm_id,drug,drug_type,starttime,stoptime",
        "900,Vancomycin,MAIN,2130-01-01 00:00:00,2130-01-01 01:00:00",
        "900,Vancomycin,BASE,2130-01-01 00:00:00,2130-01-01 01:00:00"])
    assert len(collect_drug_intervals(path, SPECS)[0]["900"]) == 1


def test_open_ended_order_is_bounded_not_infinite(tmp_path):
    """No stop time must not mark a drug active for the rest of the admission."""
    path = _write(tmp_path, "p.csv", [
        "hadm_id,drug,drug_type,starttime,stoptime",
        "900,Vancomycin,MAIN,2130-01-01 00:00:00,"])
    intervals, _ = collect_drug_intervals(path, SPECS, default_duration_hours=6)
    assert intervals["900"][0][2] == datetime(2130, 1, 1) + timedelta(hours=6)


def test_other_admissions_are_not_joined(tmp_path):
    path = _write(tmp_path, "p.csv", [
        "hadm_id,drug,drug_type,starttime,stoptime",
        "900,Vancomycin,MAIN,2130-01-01 00:00:00,2130-01-01 01:00:00",
        "901,Vancomycin,MAIN,2130-01-01 00:00:00,2130-01-01 01:00:00"])
    assert set(collect_drug_intervals(path, SPECS, {"900"})[0]) == {"900"}


def test_intervals_are_clipped_to_the_stay_grid():
    hours = [datetime(2130, 1, 1, h) for h in range(5)]
    # Starts before the stay, ends inside it.
    before = [(0, datetime(2129, 12, 31, 20), datetime(2130, 1, 1, 1))]
    assert [row[0] for row in drug_hour_matrix(before, hours, 2)] == [1, 1, 0, 0, 0]
    # Entirely after the stay.
    after = [(0, datetime(2130, 1, 2), datetime(2130, 1, 3))]
    assert [row[0] for row in drug_hour_matrix(after, hours, 2)] == [0] * 5
    # A single-hour exposure of the second class only.
    single = [(1, datetime(2130, 1, 1, 2), datetime(2130, 1, 1, 2))]
    assert drug_hour_matrix(single, hours, 2)[2] == [0, 1]


def test_registry_rejects_a_pattern_claimed_twice(tmp_path):
    path = _write(tmp_path, "r.yaml", [
        'version: "x"', "drugs:", "  - name: a", '    match: ["insulin"]',
        "  - name: b", '    match: ["insulin"]'])
    with pytest.raises(ValueError, match="two drug classes"):
        load_drug_registry(path)


def test_shipped_registry_has_no_cross_class_shadowing():
    """First-match-wins is only safe while no class's pattern contains another class's.

    If it did, the earlier class would silently swallow the later one's drugs and the
    swallowed class would look like it never happens.
    """
    version, specs, digest = load_drug_registry("configs/drug_registry.yaml")
    assert version and len(digest) == 64 and len(specs) > 20
    patterns = [(index, pattern) for index, spec in enumerate(specs) for pattern in spec.patterns]
    shadowed = [(a, b) for a in patterns for b in patterns if a[0] != b[0] and b[1] in a[1]]
    assert not shadowed, f"one class's pattern contains another's: {shadowed}"


def test_shipped_registry_routes_real_drug_strings():
    """The exact strings these cohorts contain, including the trimmed-pattern cases."""
    specs = load_drug_registry("configs/drug_registry.yaml")[1]
    names = [spec.name for spec in specs]
    for drug, expected in [("Norepinephrine", "vasopressor"),
                           ("MethylPREDNISolone Sodium Succ", "corticosteroid"),
                           ("Esomeprazole", "acid_suppression"),
                           ("Nitroglycerine SL", "nitroglycerin"),
                           ("Levalbuterol Neb", "bronchodilator"),
                           ("Heparin Flush (10 units/ml)", "heparin"),
                           ("Sodium Chloride 0.9%  Flush", "saline"),
                           ("Potassium Chloride Replacement (Critical Care and Oncology)",
                            "potassium_chloride"),
                           ("HYDROmorphone (Dilaudid)", "opioid"),
                           ("Vancomycin Oral Liquid", "vancomycin")]:
        index = classify_drug(drug, specs)
        assert index is not None and names[index] == expected, f"{drug} -> {names[index] if index is not None else None}"
