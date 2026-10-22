"""Inspect source schema, canonical feature coverage, and medication-label coverage."""
import argparse, json
from mimic_forecaster.config import load_config, resolve_path
from mimic_forecaster.data.mimic import load_sources
from mimic_forecaster.data.prescriptions import load_drug_registry
from mimic_forecaster.data.registry import load_registry

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--limit-stays",type=int); args=parser.parse_args()
    cfg=load_config(args.config); version,features,digest=load_registry(resolve_path(cfg,cfg["data"]["feature_registry"]))
    drug_path=cfg["data"].get("drug_registry")
    drug_version,drugs,drug_digest=(load_drug_registry(resolve_path(cfg,drug_path)) if drug_path else ("none",(),""))
    stays=load_sources(cfg,features,args.limit_stays,drugs)
    cells=sum(len(s.values)*len(features) for s in stays); observed=sum(sum(map(sum,s.observed_mask)) for s in stays)
    drug_hours=sum(sum(map(sum,s.drug_active)) for s in stays if s.drug_active)
    labelled=sum(len(s.drug_active) for s in stays if s.drug_active)
    charted=sorted({name for s in stays for name,flag in zip(s.drug_names,s.drug_present_mask) if flag})
    print(json.dumps({"registry_version":version,"registry_digest":digest,"stays":len(stays),"hours":sum(len(s.timestamps) for s in stays),"features":[f.name for f in features],"observed_cells":observed,"total_cells":cells,"coverage":observed/cells if cells else 0,
        "drug_registry_version":drug_version,"drug_registry_digest":drug_digest,"drug_classes":len(drugs),
        "drug_classes_charted":charted,"stay_hours_with_drug_labels":labelled,
        "active_drug_hours":drug_hours,"mean_active_drugs_per_hour":round(drug_hours/labelled,3) if labelled else 0},indent=2))
if __name__=="__main__": main()
