"""Evaluate a compatible checkpoint: forecast, medications (active and new starts), imputation.

Medication heads are scored twice: with the patient's current medications supplied, and
without them. The first is compared against carrying the current medications forward,
which is the baseline any "what will they be on" model has to beat. Per-class decision
thresholds are fitted on the validation split and written next to the checkpoint.
"""
import argparse,json,numpy as np,torch
from torch.utils.data import DataLoader
from mimic_forecaster.config import config_digest,load_config,resolve_path
from mimic_forecaster.data.artifacts import load_prepared
from mimic_forecaster.pipeline import build_forecaster
from mimic_forecaster.training import ArrayDataset,model_inputs,resolve_device

def bootstrap_skill(model_error,persistence_error,mask,seed=0,draws=2000):
    """Paired bootstrap over test windows: % macro-MAE improvement over persistence.

    Windows from one stay are correlated, so the plain metric difference overstates its own
    precision. Resampling whole windows gives an interval a reviewer can trust.
    """
    rng=np.random.default_rng(seed); n=len(mask); skills=[]
    def macro(error,index):
        counts=mask[index].sum((0,1)); keep=counts>0
        return float(np.mean((error[index].sum((0,1))[keep])/counts[keep]))
    for _ in range(draws):
        index=rng.integers(0,n,n)
        base=macro(persistence_error,index)
        skills.append(100.0*(base-macro(model_error,index))/base if base else 0.0)
    low,high=np.percentile(skills,[2.5,97.5])
    return {"skill_vs_persistence_pct":round(float(np.mean(skills)),2),
            "ci95":[round(float(low),2),round(float(high),2)]}


def roc_auc(scores,labels):
    """Rank-based AUROC. No sklearn dependency for one formula."""
    labels=np.asarray(labels); positives=labels.sum(); negatives=len(labels)-positives
    if not positives or not negatives: return None
    order=np.argsort(scores,kind="mergesort"); ranks=np.empty(len(scores),dtype=np.float64)
    ranks[order]=np.arange(1,len(scores)+1)
    # average ranks within ties, or AUROC is biased by the sort order of equal scores
    sorted_scores=scores[order]
    start=0
    for index in range(1,len(scores)+1):
        if index==len(scores) or sorted_scores[index]!=sorted_scores[start]:
            if index-start>1: ranks[order[start:index]]=ranks[order[start:index]].mean()
            start=index
    return float((ranks[labels.astype(bool)].sum()-positives*(positives+1)/2)/(positives*negatives))


def average_precision(scores,labels):
    """Area under the precision-recall curve, tie-aware, so a 0/1 baseline scores honestly."""
    labels=np.asarray(labels,dtype=np.float64); positives=labels.sum()
    if not positives: return None
    order=np.argsort(-scores,kind="mergesort"); sorted_scores=scores[order]; hits=labels[order].cumsum()
    last=np.r_[np.flatnonzero(np.diff(sorted_scores)),len(scores)-1]  # end of each tie group
    precision=hits[last]/(last+1); recall=hits[last]/positives
    return float(np.sum(np.diff(np.r_[0.0,recall])*precision))


def best_threshold(scores,labels):
    """The probability cut that maximises F1 on this split; 0.5 when the class has no positives."""
    if labels.sum()==0 or labels.sum()==len(labels): return 0.5
    order=np.argsort(-scores,kind="mergesort"); hits=labels[order].cumsum()
    f1=2*hits/(np.arange(1,len(labels)+1)+labels.sum())
    return float(scores[order][int(np.argmax(f1))])


LAB_UNITS={"glucose":"mg/dL","potassium":"mEq/L","sodium":"mEq/L","bicarbonate":"mEq/L","bun":"mg/dL",
           "creatinine":"mg/dL","magnesium":"mg/dL","calcium":"mg/dL","wbc":"K/uL","hemoglobin":"g/dL",
           "platelets":"K/uL","inr":"ratio","bilirubin":"mg/dL","lactate":"mmol/L","ph":""}
UNITS={"heart_rate":"bpm","respiratory_rate":"breaths/min","spo2":"%","systolic_bp":"mmHg",
       "diastolic_bp":"mmHg","mean_bp":"mmHg","temperature_f":"°F",**LAB_UNITS}


def run_split(model,windows,split,device,batch_size,rate,med_rate,seed,meds):
    """Every head's output over one split, in scaled units, plus the column-dropout draw."""
    loader=DataLoader(ArrayDataset(windows,split),batch_size=batch_size)
    generator=torch.Generator(device=device).manual_seed(seed); out={}
    with torch.no_grad():
        for raw in loader:
            batch=tuple(item.to(device) for item in raw)
            full,_=model_inputs(batch,0.0,0.0,None,meds)
            heads=model(*full); parts={"forecast":heads["forecast"]}
            for key in ("drug_logits","start_logits"):
                if key in heads: parts[key]=torch.sigmoid(heads[key])
            if rate>0:
                hidden_inputs,keep=model_inputs(batch,rate,0.0,generator,meds)
                parts["impute"]=model(*hidden_inputs)["impute"]; parts["keep"]=keep
            for key,value in parts.items(): out.setdefault(key,[]).append(value.float().cpu().numpy())
    return {key:np.concatenate(value) for key,value in out.items()}


def score_classes(probability,labels,masks,names,baseline=None):
    classes={}
    for index,name in enumerate(names):
        keep=masks[:,index].astype(bool)
        if keep.sum()<2: continue
        y=labels[keep,index]; s=probability[keep,index]; prevalence=float(y.mean())
        if prevalence in (0.0,1.0): continue  # a class with one label cannot be scored
        classes[name]={"auroc":roc_auc(s,y),"ap":average_precision(s,y),"prevalence":round(prevalence,4),"n":int(keep.sum())}
        if baseline is not None: classes[name]["carry_forward_ap"]=average_precision(baseline[keep,index],y)
    def macro(key): values=[v[key] for v in classes.values() if v.get(key) is not None]; return float(np.mean(values)) if values else 0.0
    summary={"classes":classes,"macro_auroc":macro("auroc"),"macro_ap":macro("ap"),"baseline_ap":macro("prevalence"),
             "scored_classes":len(classes),"unscorable_classes":len(names)-len(classes)}
    if baseline is not None: summary["carry_forward_ap"]=macro("carry_forward_ap")
    return summary


def report(result,meta,cfg):
    """Human-readable test report: what it predicts, how wrong it is, and against what."""
    stays=meta.get("stays"); n=sum(v["n"] for v in result["forecast"]["features"].values())
    print(f"\nTest report  -  {meta.get('registry_version','?')}  -  {stays} ICU stays, "
          f"{len(result['forecast']['features'])} features, {n:,} observed target values")
    print(f"Forecasting the next {cfg['task']['horizon_hours']}h from {cfg['task']['context_hours']}h of history\n")
    print("1. FORECAST")
    print(f"  {'feature':<20} {'MAE':>8} {'RMSE':>8}  {'unit':<14} {'n':>6}")
    print("  " + "-"*60)
    for name,m in result["forecast"]["features"].items():
        print(f"  {name:<20} {m['mae']:>8.2f} {m['rmse']:>8.2f}  {UNITS.get(name,''):<14} {m['n']:>6,}")
    b=result["forecast"]["baselines"]; skill=result["forecast"]["skill"]; lo,hi=skill["ci95"]
    print(f"\n  Skill over repeat-last-value: {skill['skill_vs_persistence_pct']:+.2f}%  (95% CI {lo:+.2f}% to {hi:+.2f}%)"
          f"   macro MAE {result['forecast']['macro_mae']:.3f} vs {b['persistence']:.3f}")
    for key,title in (("medication",f"MEDICATIONS ACTIVE in the next {cfg['task']['horizon_hours']}h"),
                      ("new_start",f"NEW STARTS within {cfg['task'].get('start_horizon_hours',24)}h")):
        section=result.get(key)
        if not section: continue
        known,unknown=section["with_current_meds"],section["without_current_meds"]
        print(f"\n{'2' if key=='medication' else '3'}. {title}")
        print(f"  {'drug class':<20} {'AUROC':>7} {'AP':>7} {'AP no-meds':>11} {'prevalence':>11}"+(f" {'carry-fwd AP':>13}" if key=="medication" else ""))
        print("  " + "-"*(72 if key=="medication" else 60))
        for name,m in sorted(known["classes"].items(),key=lambda kv:-(kv[1]["ap"] or 0)):
            other=unknown["classes"].get(name,{}).get("ap")
            line=f"  {name:<20} {m['auroc'] or 0:>7.3f} {m['ap'] or 0:>7.3f} {other or 0:>11.3f} {m['prevalence']:>11.3f}"
            if key=="medication": line+=f" {m.get('carry_forward_ap') or 0:>13.3f}"
            print(line)
        print(f"\n  macro AP with current meds    {known['macro_ap']:.3f}   AUROC {known['macro_auroc']:.3f}")
        print(f"  macro AP without current meds {unknown['macro_ap']:.3f}   AUROC {unknown['macro_auroc']:.3f}")
        if key=="medication": print(f"  carry current meds forward    {known['carry_forward_ap']:.3f}")
        print(f"  predict-prevalence            {known['baseline_ap']:.3f}")
    impute=result.get("imputation")
    if impute:
        print(f"\n4. COLUMN IMPUTATION  -  value of a withheld column ({impute['dropout_rate']:.0%} hidden)")
        print(f"  {'feature':<20} {'MAE':>8} {'median MAE':>11} {'unit':<14} {'n':>6}")
        print("  " + "-"*62)
        for name,m in impute["features"].items():
            print(f"  {name:<20} {m['mae']:>8.2f} {m['baseline_mae']:>11.2f}  {UNITS.get(name,''):<14} {m['n']:>6,}")
        print(f"\n  Skill over median fill: {impute['skill_vs_median_pct']:+.2f}%")
    print()


def baseline_maes(windows,stats):
    """Doc-required reference points: last-observation-carried-forward and training median."""
    test=windows.split=="test"; targets,mask=windows.targets[test],windows.target_mask[test]
    persistence=np.repeat(windows.values[test][:,-1:,:],targets.shape[1],axis=1)
    out={}
    for name,prediction in (("persistence",persistence),("training_median",np.zeros_like(targets))):
        counts=mask.sum((0,1)); keep=counts>0
        error=(np.abs((prediction-targets)*stats.scale)*mask).sum((0,1))[keep]/counts[keep]
        out[name]=float(np.mean(error))
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--json",action="store_true")
    args=parser.parse_args(); cfg=load_config(args.config); artifact=resolve_path(cfg,cfg["project"]["artifact_dir"])
    windows,stats,meta=load_prepared(artifact/"prepared")
    checkpoint=torch.load(artifact/"checkpoints"/"multitask.pt",map_location="cpu",weights_only=False)
    for key,value in {"config_digest":config_digest(cfg),"registry_digest":meta["registry_digest"]}.items():
        if checkpoint["contract"].get(key)!=value: raise ValueError(f"checkpoint contract mismatch: {key}")
    drugs=meta.get("drugs",[]); n_features=windows.values.shape[-1]
    device=resolve_device(cfg["project"].get("device"))
    model=build_forecaster(cfg,n_features,len(drugs)); model.load_state_dict(checkpoint["model"]); model.eval(); model.to(device)
    rate=float(cfg["task"].get("column_dropout_rate",0.0))
    # A head with zero loss weight was never trained, so scoring it would publish noise.
    weights=cfg["task"].get("loss_weights",{}) or {}
    if rate>0 and not float(weights.get("impute",1.0))>0: rate=0.0
    meds_input=bool(cfg["task"].get("medication_inputs"))
    seed=int(cfg["project"]["seed"]); size=int(cfg["training"]["batch_size"])
    run=lambda split,meds,r=0.0: run_split(model,windows,split,device,size,r,0.0,seed,meds)
    known=run("test",meds_input,rate); unknown=run("test",False) if meds_input else known
    test=windows.split=="test"

    targets,target_mask=windows.targets[test],windows.target_mask[test]
    error=(known["forecast"]-targets)*stats.scale.reshape(1,1,-1)
    counts=target_mask.sum((0,1))
    metrics={name:{"mae":float((np.abs(error)*target_mask).sum((0,1))[i]/counts[i]),
                   "rmse":float(np.sqrt(((error**2)*target_mask).sum((0,1))[i]/counts[i])),"n":int(counts[i])}
             for i,name in enumerate(meta["features"]) if counts[i]}
    persistence=np.repeat(windows.values[test][:,-1:,:],targets.shape[1],axis=1)
    persistence_error=np.abs((persistence-targets)*stats.scale)*target_mask
    result={"forecast":{"macro_mae":float(np.mean([v["mae"] for v in metrics.values()])),
                        "baselines":baseline_maes(windows,stats),
                        "skill":bootstrap_skill(np.abs(error)*target_mask,persistence_error,target_mask),
                        "features":metrics}}

    thresholds={}
    if drugs and "drug_logits" in known:
        carry=windows.drug_history[test][:,-1,:]  # what the patient is on at the last observed hour
        sections=(("medication","drug_logits",windows.drug_targets,windows.drug_mask,"drug"),
                  ("new_start","start_logits",windows.start_targets,windows.start_mask,"drug_start"))
        val_known=run("val",meds_input); val_unknown=run("val",False) if meds_input else val_known
        val=windows.split=="val"
        for key,head,labels,masks,weight in sections:
            if not float(weights.get(weight,1.0))>0: continue
            result[key]={"with_current_meds":score_classes(known[head],labels[test],masks[test],drugs,carry if key=="medication" else None),
                         "without_current_meds":score_classes(unknown[head],labels[test],masks[test],drugs)}
            for mode,scores in (("with_current_meds",val_known),("without_current_meds",val_unknown)):
                thresholds.setdefault(mode,{})[key]={name:round(best_threshold(scores[head][m.astype(bool),i],labels[val][m.astype(bool),i]),4)
                    for i,name in enumerate(drugs) for m in [masks[val][:,i]]}
        (artifact/"thresholds.json").write_text(json.dumps(thresholds,indent=2)+"\n")

    if rate>0 and "impute" in known:
        keep=known["keep"]; hidden=(1.0-keep)*windows.observed_mask[test][:,-1,:]; truth=windows.values[test][:,-1,:]
        imp_abs=(np.abs(known["impute"]-truth)*stats.scale*hidden).sum(0)
        # Median fill is the reference an imputer must beat: scaled median is 0.
        imp_base=(np.abs(truth)*stats.scale*hidden).sum(0); imp_counts=hidden.sum(0)
        features={name:{"mae":float(imp_abs[i]/imp_counts[i]),"baseline_mae":float(imp_base[i]/imp_counts[i]),
                        "n":int(imp_counts[i])} for i,name in enumerate(meta["features"]) if imp_counts[i]}
        if features:
            macro=float(np.mean([v["mae"] for v in features.values()])); base=float(np.mean([v["baseline_mae"] for v in features.values()]))
            result["imputation"]={"dropout_rate":rate,"macro_mae":macro,"baseline_macro_mae":base,
                                  "skill_vs_median_pct":round(100.0*(base-macro)/base,2) if base else 0.0,"features":features}

    (artifact/"test_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)) if args.json else report(result,meta,cfg)
if __name__=="__main__": main()
