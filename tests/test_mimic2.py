import csv, io, tarfile
from mimic_forecaster.data.mimic2 import convert

FILES = {
    "ICUSTAY_DETAIL": "ICUSTAY_ID,SUBJECT_ID,HADM_ID,ICUSTAY_INTIME,ICUSTAY_OUTTIME,ICUSTAY_AGE_GROUP\n"
                      '1,7,70,3400-01-01 10:00:00 EST,3400-01-02 10:00:00 EST,"adult"\n'
                      '2,7,71,3400-02-01 10:00:00 EST,3400-02-02 10:00:00 EST,"neonate"\n',
    "CHARTEVENTS": "SUBJECT_ID,ICUSTAY_ID,ITEMID,CHARTTIME,VALUE1NUM,VALUE2NUM\n"
                   "7,1,455,3400-01-01 11:00:00 EST,120,70\n"   # cuff BP: one row, both pressures
                   "7,1,999,3400-01-01 11:00:00 EST,5,\n",      # not a registry vital
    "LABEVENTS": "SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME,VALUE,VALUENUM\n"
                 "7,70,,50112,3400-01-01 12:00:00 EST,140,140\n"  # MIMIC-II glucose
                 "7,70,,50157,3400-01-01 12:00:00 EST,HOLD,\n",   # unmapped
    "POE_ORDER": "HADM_ID,MEDICATION,START_DT,STOP_DT\n70,Heparin,3400-01-01 10:00:00 EST,\n",
}


def test_convert_splits_bp_maps_labs_drops_neonates(tmp_path):
    tarball = tmp_path / "m2.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        for table, text in FILES.items():
            data = text.encode(); info = tarfile.TarInfo(f"MIMIC-Importer-2.6/07/7/{table}-7.txt"); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    counts = convert(tarball, tmp_path / "out", chart_itemids={455, 8441})
    read = lambda name: list(csv.DictReader((tmp_path / "out" / f"{name}.csv").open()))
    assert [r["icustay_id"] for r in read("ICUSTAYS")] == ["1"]                 # neonate dropped
    assert read("ICUSTAYS")[0]["intime"] == "3400-01-01 10:00:00"                # timezone stripped
    assert [(r["itemid"], r["valuenum"]) for r in read("CHARTEVENTS")] == [("455", "120"), ("8441", "70")]
    assert [(r["itemid"], r["valuenum"]) for r in read("LABEVENTS")] == [("50931", "140")]
    assert read("PRESCRIPTIONS")[0]["drug"] == "Heparin" and read("PRESCRIPTIONS")[0]["enddate"] == ""
    assert counts["subjects"] == 1
