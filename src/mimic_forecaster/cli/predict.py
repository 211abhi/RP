"""Predict from one ICU stay: next-hour vitals, medication classes, and withheld columns.

`--features` is the point of the missing-column path: name only the columns a site actually
charts and every other configured vital is marked absent, exactly as it would be at a
source that never records it. The response then carries the imputation head's estimate of
those absent columns, flagged as estimated rather than measured.
"""
import argparse,csv,json,math,numpy as np,torch
from collections import defaultdict
from datetime import timedelta
from statistics import median
from mimic_forecaster.config import config_digest,load_config,resolve_path
from mimic_forecaster.data.artifacts import load_prepared
from mimic_forecaster.data.mimic import HourlyStay,_floor_hour,_hours,_parse_time,load_sources
from mimic_forecaster.data.preprocessing import causal_impute,transform
from mimic_forecaster.data.prescriptions import load_drug_registry
from mimic_forecaster.data.registry import load_registry
from mimic_forecaster.pipeline import build_forecaster

def load_checkpoint(cfg,artifact,n_features,registry_digest,n_drugs=0):
    checkpoint=torch.load(artifact/"checkpoints"/"multitask.pt",map_location="cpu",weights_only=False)
    for key,value in {"config_digest":config_digest(cfg),"registry_digest":registry_digest}.items():
        if checkpoint["contract"].get(key)!=value: raise ValueError(f"checkpoint contract mismatch: {key}")
    model=build_forecaster(cfg,n_features,n_drugs); model.load_state_dict(checkpoint["model"]); model.eval()
    return model,checkpoint

def load_patient_csv(path,features,min_hours):
    """One patient from a wide CSV: a `time` column plus any subset of the registry vitals.

    Rows are binned to the hour with within-hour medians and out-of-range values dropped,
    exactly as chartevents are. Columns the file does not have are absent, so they come back
    imputed; blank cells are unobserved.
    """
    names=[f.name for f in features]
    with open(path,newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle); columns=set(reader.fieldnames or ())
        if "time" not in columns: raise SystemExit(f"{path}: needs a 'time' column (ISO format, e.g. 2024-05-01 08:00)")
        unknown=columns-set(names)-{"time"}
        if unknown: raise SystemExit(f"{path}: unknown columns {sorted(unknown)}; allowed: time, {', '.join(names)}")
        supplied=[n for n in names if n in columns]
        if not supplied: raise SystemExit(f"{path}: no vital columns; allowed: {', '.join(names)}")
        buckets=defaultdict(list)
        for line,row in enumerate(reader,start=2):
            try: hour=_floor_hour(_parse_time(row["time"]))
            except (TypeError,ValueError): raise SystemExit(f"{path}:{line}: bad time {row['time']!r}")
            for index,spec in enumerate(features):
                cell=(row.get(spec.name) or "").strip()
                if not cell: continue
                try: value=float(cell)
                except ValueError: raise SystemExit(f"{path}:{line}: {spec.name} is not a number: {cell!r}")
                if math.isfinite(value) and spec.minimum<=value<=spec.maximum: buckets[(hour,index)].append(value)
    if not buckets: raise SystemExit(f"{path}: no in-range observations")
    timestamps=list(_hours(min(h for h,_ in buckets),max(h for h,_ in buckets)))
    if len(timestamps)<min_hours: raise SystemExit(f"{path}: spans {len(timestamps)} hours, the model needs at least {min_hours}")
    values=[[float(median(buckets[(h,i)])) if (h,i) in buckets else math.nan for i in range(len(names))] for h in timestamps]
    masks=[[int((h,i) in buckets) for i in range(len(names))] for h in timestamps]
    return HourlyStay("patient","patient",timestamps,tuple(names),values,masks,[int(n in columns) for n in names]),supplied

def resolve_keep(feature_names,requested):
    """Turn a requested column subset into a keep mask, rejecting names not in the registry."""
    if not requested: return np.ones(len(feature_names),dtype=np.float32)
    wanted={name.strip() for name in requested if name.strip()}
    unknown=wanted.difference(feature_names)
    if unknown: raise SystemExit(f"not in the feature registry: {sorted(unknown)}; have {list(feature_names)}")
    if not wanted: raise SystemExit("--features needs at least one column")
    return np.asarray([float(name in wanted) for name in feature_names],dtype=np.float32)

def forecast(model,stay,stats,features,context,max_gap,keep=None):
    """Run the last `context` hourly bins of one stay through every head, in original units.

    A hidden column is zeroed in all four input channels and its staleness is set to the
    maximum, so it is indistinguishable from a column the source never had.
    """
    raw=np.asarray(stay.values,dtype=np.float32); mask=np.asarray(stay.observed_mask,dtype=np.float32)
    if len(raw)<context: raise ValueError(f"stay {stay.stay_id} has {len(raw)} hourly bins, needs {context}")
    filled,age=causal_impute(raw,mask,stats.median,max_gap)
    window=slice(len(raw)-context,len(raw))
    present=np.asarray(stay.feature_present_mask,dtype=np.float32)
    values,observed,ages=transform(filled[window],stats),mask[window],age[window]
    if keep is not None:
        present=present*keep
        values,observed,ages=values*keep,observed*keep,ages*keep+(1.0-keep)
    batch=[torch.from_numpy(a[None]).float() for a in (values,observed,present,ages)]
    with torch.no_grad(): out=model(*batch)
    result={"forecast":out["forecast"][0].numpy()*stats.scale+stats.median,
            "impute":out["impute"][0].numpy()*stats.scale+stats.median}
    if "drug_logits" in out: result["drug_probability"]=torch.sigmoid(out["drug_logits"])[0].numpy()
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--stay-id"); parser.add_argument("--limit-stays",type=int)
    parser.add_argument("--patient-csv",help="predict for your own patient: CSV with a 'time' column and any subset of the vitals")
    parser.add_argument("--features",help="comma-separated columns to supply; every other configured vital is treated as absent and imputed")
    parser.add_argument("--drug-threshold",type=float,default=0.5,help="probability above which a medication class is listed as predicted")
    args=parser.parse_args(); cfg=load_config(args.config); artifact=resolve_path(cfg,cfg["project"]["artifact_dir"])
    _,features,registry_digest=load_registry(resolve_path(cfg,cfg["data"]["feature_registry"]))
    _,stats,meta=load_prepared(artifact/"prepared")
    if meta["features"]!=[f.name for f in features]: raise ValueError("registry feature order differs from the prepared artifacts")
    drug_names=meta.get("drugs",[])
    drug_path=cfg["data"].get("drug_registry")
    drugs=load_drug_registry(resolve_path(cfg,drug_path))[1] if drug_path else ()
    model,checkpoint=load_checkpoint(cfg,artifact,len(features),registry_digest,len(drug_names))
    context=int(cfg["task"]["context_hours"]); frequency=int(cfg["data"]["frequency_hours"]); results=[]
    if args.patient_csv:
        stay,supplied_in_file=load_patient_csv(args.patient_csv,features,context); stays=[stay]
        args.features=args.features or ",".join(supplied_in_file)
    else: stays=load_sources(cfg,features,args.limit_stays,drugs)
    if args.stay_id and not args.patient_csv: stays=[s for s in stays if s.stay_id==args.stay_id] or _die(args.stay_id)
    keep=resolve_keep(meta["features"],args.features.split(",") if args.features else None)
    supplied=[n for n,k in zip(meta["features"],keep) if k]; withheld=[n for n,k in zip(meta["features"],keep) if not k]
    for stay in stays:
        if len(stay.values)<context: continue
        out=forecast(model,stay,stats,features,context,int(cfg["data"]["max_forward_fill_hours"]),keep)
        last=stay.timestamps[-1]; prediction=out["forecast"]
        entry={"stay_id":stay.stay_id,"subject_id":stay.subject_id,"last_observed_hour":last.isoformat(),
            "forecast":[{"hour":(last+timedelta(hours=(step+1)*frequency)).isoformat(),
                "values":{name:round(float(prediction[step,i]),3) for i,name in enumerate(meta["features"])}} for step in range(prediction.shape[0])],
            "imputed_columns":{name:round(float(out["impute"][i]),3) for i,name in enumerate(meta["features"]) if not keep[i]}}
        if "drug_probability" in out:
            probability={name:round(float(out["drug_probability"][i]),4) for i,name in enumerate(drug_names)}
            entry["medications"]={"predicted":sorted([n for n,p in probability.items() if p>=args.drug_threshold],
                                                     key=lambda n:-probability[n]),"probability":probability}
        results.append(entry)
    if not results: raise ValueError("no stay had enough history to forecast")
    print(json.dumps({"val_loss":checkpoint.get("val_loss",checkpoint.get("val_mse")),"stays":len(results),
        "supplied_columns":supplied,"withheld_columns":withheld,
        "note":"imputed_columns are model estimates, not measurements" if withheld else "all configured columns were supplied",
        "drug_threshold":args.drug_threshold,"predictions":results},indent=2))

def _die(stay_id): raise SystemExit(f"stay_id not found in the configured source: {stay_id}")

if __name__=="__main__": main()
