"""Convert the MIMIC-II demo flat files (v2.6, 4,000 subjects) into MIMIC-III-shaped tables.

MIMIC-II ships one folder per subject inside one tarball, with uppercase headers, a
timezone suffix on every timestamp (`3485-04-23 17:16:00 EST`), its own lab itemids, and
medication orders in POE_ORDER instead of PRESCRIPTIONS. Rather than teach the loader a
fourth dialect, this writes the four tables it already reads, so MIMIC-II becomes one more
`data.sources` entry:

  ICUSTAYS.csv       icustay_id, subject_id, hadm_id, intime, outtime
  CHARTEVENTS.csv    icustay_id, itemid, charttime, valuenum   (registry vitals only)
  LABEVENTS.csv      subject_id, hadm_id, itemid, charttime, valuenum  (MIMIC-III itemids)
  PRESCRIPTIONS.csv  hadm_id, drug, drug_type, startdate, enddate

Chart itemids are the CareVue ids MIMIC-III also uses, except that MIMIC-II charts cuff
pressure as one item (455) with systolic in VALUE1NUM and diastolic in VALUE2NUM; it is
split into 455 and 8441 here.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import tarfile
from pathlib import Path

# MIMIC-II lab itemid -> the MIMIC-III/IV itemid the feature registry uses (blood only).
LAB_MAP = {50112: 50931, 50149: 50971, 50159: 50983, 50172: 50882, 50177: 51006, 50090: 50912,
           50140: 50960, 50079: 50893, 50468: 51301, 50386: 51222, 50428: 51265, 50399: 51237,
           50170: 50885, 50010: 50813, 50018: 50820}
NBP, NBP_DIASTOLIC = 455, 8441
MEMBER = re.compile(r"/(\d+)/([A-Z_0-9]+)-\1\.txt$")
TABLES = {"ICUSTAY_DETAIL", "CHARTEVENTS", "LABEVENTS", "POE_ORDER"}


def clean_time(value: str) -> str:
    """Drop the timezone word MIMIC-II appends; blank stays blank."""
    value = (value or "").strip()
    return re.sub(r"\s+[A-Z]{3}$", "", value)


def convert(tarball: str | Path, out: str | Path, chart_itemids: set[int]) -> dict[str, int]:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    specs = {"ICUSTAYS": ["icustay_id", "subject_id", "hadm_id", "intime", "outtime"],
             "CHARTEVENTS": ["icustay_id", "itemid", "charttime", "valuenum"],
             "LABEVENTS": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
             "PRESCRIPTIONS": ["hadm_id", "drug", "drug_type", "startdate", "enddate"]}
    handles = {name: (out / f"{name}.csv").open("w", newline="", encoding="utf-8") for name in specs}
    writers = {name: csv.writer(handles[name]) for name in specs}
    for name, header in specs.items(): writers[name].writerow(header)
    counts = dict.fromkeys(specs, 0); subjects = set()
    try:
        with tarfile.open(tarball, "r|gz") as archive:  # streamed: one pass, nothing extracted
            for member in archive:
                match = MEMBER.search(member.name)
                if not member.isfile() or "/._" in member.name or not match or match.group(2) not in TABLES:
                    continue
                subjects.add(match.group(1))
                text = io.StringIO(archive.extractfile(member).read().decode("utf-8", errors="replace"))
                rows = csv.DictReader(text); table = match.group(2)
                if table == "ICUSTAY_DETAIL":
                    for r in rows:
                        # Adults only: the 28 neonatal stays have different vitals and drugs.
                        if r["ICUSTAY_INTIME"] and r["ICUSTAY_OUTTIME"] and r["ICUSTAY_AGE_GROUP"] == "adult":
                            writers["ICUSTAYS"].writerow([r["ICUSTAY_ID"], r["SUBJECT_ID"], r["HADM_ID"],
                                                          clean_time(r["ICUSTAY_INTIME"]), clean_time(r["ICUSTAY_OUTTIME"])])
                            counts["ICUSTAYS"] += 1
                elif table == "CHARTEVENTS":
                    for r in rows:
                        itemid = int(r["ITEMID"]) if r["ITEMID"].isdigit() else -1
                        values = [(itemid, r["VALUE1NUM"])]
                        if itemid == NBP: values.append((NBP_DIASTOLIC, r["VALUE2NUM"]))
                        for item, value in values:
                            if item in chart_itemids and value:
                                writers["CHARTEVENTS"].writerow([r["ICUSTAY_ID"], item, clean_time(r["CHARTTIME"]), value])
                                counts["CHARTEVENTS"] += 1
                elif table == "LABEVENTS":
                    for r in rows:
                        mapped = LAB_MAP.get(int(r["ITEMID"])) if r["ITEMID"].isdigit() else None
                        if mapped and r["VALUENUM"]:
                            writers["LABEVENTS"].writerow([r["SUBJECT_ID"], r["HADM_ID"], mapped, clean_time(r["CHARTTIME"]), r["VALUENUM"]])
                            counts["LABEVENTS"] += 1
                else:  # POE_ORDER: one row per medication order, the PRESCRIPTIONS equivalent
                    for r in rows:
                        if r["MEDICATION"] and r["HADM_ID"] and r["START_DT"]:
                            writers["PRESCRIPTIONS"].writerow([r["HADM_ID"], r["MEDICATION"], "MAIN",
                                                               clean_time(r["START_DT"]), clean_time(r["STOP_DT"])])
                            counts["PRESCRIPTIONS"] += 1
    finally:
        for handle in handles.values(): handle.close()
    return {"subjects": len(subjects), **counts}


def main():
    from .registry import load_registry
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tarball", default="data/raw/mimic-ii-demo/mimic2_flatfiles.tar.gz")
    parser.add_argument("--out", default="data/raw/mimic-ii-demo/mimic2")
    parser.add_argument("--feature-registry", default="configs/feature_registry.yaml")
    args = parser.parse_args()
    _, features, _ = load_registry(args.feature_registry)
    chart = {itemid for f in features if f.table == "chartevents" for itemid in f.itemids}
    print(convert(args.tarball, args.out, chart))


if __name__ == "__main__":
    main()
