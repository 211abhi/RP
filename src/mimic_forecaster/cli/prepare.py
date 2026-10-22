"""Prepare fixed-width masked windows, vital targets, and medication labels."""
import argparse,json
import numpy as np
from mimic_forecaster.config import config_digest,load_config,resolve_path
from mimic_forecaster.data.artifacts import save_prepared
from mimic_forecaster.data.mimic import load_sources
from mimic_forecaster.data.prescriptions import load_drug_registry
from mimic_forecaster.data.registry import fill_limits,load_registry
from mimic_forecaster.data.windows import build_windows,drug_prevalence,fit_training_stats
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--limit-stays",type=int); args=parser.parse_args(); cfg=load_config(args.config)
    version,features,registry_digest=load_registry(resolve_path(cfg,cfg["data"]["feature_registry"]))
    drug_path=cfg["data"].get("drug_registry")
    drug_version,drugs,drug_digest=(load_drug_registry(resolve_path(cfg,drug_path)) if drug_path else ("none",(),""))
    stays=load_sources(cfg,features,args.limit_stays,drugs); ratios=cfg["data"]["split_ratios"]; seed=int(cfg["project"]["seed"])
    stats=fit_training_stats(stays,seed,ratios); windows=build_windows(stays,stats,context=int(cfg["task"]["context_hours"]),horizon=int(cfg["task"]["horizon_hours"]),stride=int(cfg["task"]["stride_hours"]),max_gap=fill_limits(features,int(cfg["data"]["max_forward_fill_hours"])),seed=seed,ratios=ratios,n_drugs=len(drugs),start_horizon=int(cfg["task"].get("start_horizon_hours",24)))
    counts={name:int((windows.split==name).sum()) for name in ("train","val","test")}
    if any(not count for count in counts.values()): raise ValueError(f"one or more splits are empty: {counts}")
    prevalence=drug_prevalence(windows); start_prevalence=drug_prevalence(windows,start=True)
    meta={"config_digest":config_digest(cfg),"registry_version":version,"registry_digest":registry_digest,
          "features":[f.name for f in features],"drug_registry_version":drug_version,"drug_registry_digest":drug_digest,
          "drugs":[d.name for d in drugs],"drug_prevalence":[round(float(p),5) for p in prevalence],
          "start_prevalence":[round(float(p),5) for p in start_prevalence],
          "stays":len(stays),"windows":counts}
    output=resolve_path(cfg,cfg["project"]["artifact_dir"])/"prepared"; save_prepared(output,windows,stats,meta)
    learnable=sum(1 for p in prevalence if p>0)
    print(json.dumps({**meta,"drugs_with_positive_examples":learnable,"output":str(output)},indent=2))
if __name__=="__main__": main()
