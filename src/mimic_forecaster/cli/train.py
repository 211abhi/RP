"""Train the shared encoder with the forecast, medication, and imputation heads."""
import argparse,json
import numpy as np
from mimic_forecaster.config import config_digest,load_config,resolve_path
from mimic_forecaster.data.artifacts import load_prepared
from mimic_forecaster.pipeline import build_forecaster
from mimic_forecaster.training import train_model

def positive_weight(prevalence,cap=20.0):
    """BCE pos_weight = (1-p)/p per drug class, capped so a 1-in-5000 class cannot dominate."""
    p=np.asarray(prevalence,dtype=np.float32)
    weight=np.ones_like(p)
    valid=(p>0)&(p<1)
    weight[valid]=np.clip((1.0-p[valid])/p[valid],1.0,cap)
    return weight

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    parser.add_argument("--stage",default="multitask",choices=("multitask","head","encoder","soft-prompt"))
    parser.add_argument("--epochs",type=int,help="override training.epochs without editing the config")
    parser.add_argument("--json",action="store_true",help="print the metrics blob instead of a progress table")
    args=parser.parse_args()
    if args.stage in ("encoder","soft-prompt"): raise SystemExit("alignment/soft-prompt require an approved ablation result")
    cfg=load_config(args.config)
    # The digest is taken from the config as written on disk: --epochs is an operational
    # knob, and letting it change the contract would make the checkpoint unevaluatable.
    digest=config_digest(cfg)
    if args.epochs: cfg["training"]["epochs"]=args.epochs
    artifact=resolve_path(cfg,cfg["project"]["artifact_dir"]); windows,_,meta=load_prepared(artifact/"prepared")
    drugs=meta.get("drugs",[]); n_drugs=len(drugs)
    model=build_forecaster(cfg,windows.values.shape[-1],n_drugs)
    contract={"config_digest":digest,"registry_digest":meta["registry_digest"],
              "drug_registry_digest":meta.get("drug_registry_digest",""),"features":meta["features"],
              "drugs":drugs,"context":int(cfg["task"]["context_hours"]),"horizon":int(cfg["task"]["horizon_hours"])}
    weight=positive_weight(meta.get("drug_prevalence",[])) if n_drugs else None
    # New starts are rarer than active orders, so they get a higher cap.
    start_weight=positive_weight(meta.get("start_prevalence",[]),cap=50.0) if n_drugs and meta.get("start_prevalence") else None
    result=train_model(model,windows,cfg,artifact/"checkpoints"/"multitask.pt",contract,weight,verbose=not args.json,start_weight=start_weight)
    (artifact/"training_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    if args.json: print(json.dumps(result,indent=2))
if __name__=="__main__": main()
