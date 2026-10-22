"""Evaluate a compatible checkpoint: vital forecast, medication prediction, column imputation."""
import argparse,json,numpy as np,torch
from torch.utils.data import DataLoader
from mimic_forecaster.config import config_digest,load_config,resolve_path
from mimic_forecaster.data.artifacts import load_prepared
from mimic_forecaster.pipeline import build_forecaster
from mimic_forecaster.training import ArrayDataset,apply_column_dropout,drop_columns,resolve_device

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
    """Area under the precision-recall curve, the honest metric for rare drug classes."""
    labels=np.asarray(labels); positives=labels.sum()
    if not positives: return None
    order=np.argsort(-scores,kind="mergesort"); hits=labels[order].cumsum()
    precision=hits/np.arange(1,len(labels)+1)
    return float((precision*labels[order]).sum()/positives)


UNITS={"heart_rate":"bpm","respiratory_rate":"breaths/min","spo2":"%","systolic_bp":"mmHg",
       "diastolic_bp":"mmHg","mean_bp":"mmHg","temperature_f":"°F"}

def report(result,meta,cfg):
    """Human-readable test report: what it predicts, how wrong it is, and against what."""
    stays=meta.get("stays"); n=sum(v["n"] for v in result["forecast"]["features"].values())
    print(f"\nTest report  -  {meta.get('registry_version','?')}  -  {stays} ICU stays, "
          f"{len(result['forecast']['features'])} vitals, {n:,} observed target values")
    print(f"Forecasting the next {cfg['task']['horizon_hours']}h from {cfg['task']['context_hours']}h of history\n")

    print("1. VITAL FORECAST")
    print(f"  {'vital':<20} {'MAE':>8} {'RMSE':>8}  {'unit':<14} {'n':>6}")
    print("  " + "-"*60)
    for name,m in result["forecast"]["features"].items():
        print(f"  {name:<20} {m['mae']:>8.2f} {m['rmse']:>8.2f}  {UNITS.get(name,''):<14} {m['n']:>6,}")
    b=result["forecast"]["baselines"]; skill=result["forecast"]["skill"]
    print(f"\n  {'macro-average MAE':<28} {result['forecast']['macro_mae']:>7.3f}   <- this model")
    print(f"  {'vs. repeat-last-value':<28} {b['persistence']:>7.3f}")
    print(f"  {'vs. always-predict-median':<28} {b['training_median']:>7.3f}")
    lo,hi=skill["ci95"]
    verdict="better than repeating the last value" if lo>0 else "NOT distinguishable from repeating the last value"
    print(f"  Skill over repeat-last-value: {skill['skill_vs_persistence_pct']:+.2f}%  (95% CI {lo:+.2f}% to {hi:+.2f}%)")
    print(f"  -> {verdict}.")

    drug=result.get("medication")
    if drug and drug["classes"]:
        print(f"\n2. MEDICATION PREDICTION  -  which classes are active in the next "
              f"{cfg['task']['horizon_hours']}h, from vitals alone")
        print(f"  {'drug class':<20} {'AUROC':>7} {'AP':>7} {'prevalence':>11} {'AP lift':>8}")
        print("  " + "-"*60)
        for name,m in sorted(drug["classes"].items(),key=lambda kv:-(kv[1]["auroc"] or 0)):
            auroc="  n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
            ap="  n/a" if m["ap"] is None else f"{m['ap']:.3f}"
            lift="n/a" if not (m["ap"] and m["prevalence"]) else f"{m['ap'] / m['prevalence']:.2f}x"
            print(f"  {name:<20} {auroc:>7} {ap:>7} {m['prevalence']:>11.3f} {lift:>8}")
        print(f"\n  {'macro AUROC':<28} {drug['macro_auroc']:>7.3f}   (0.500 = no signal)")
        print(f"  {'macro average precision':<28} {drug['macro_ap']:>7.3f}")
        print(f"  {'vs. predict-prevalence AP':<28} {drug['baseline_ap']:>7.3f}")

    impute=result.get("imputation")
    if impute:
        print(f"\n3. COLUMN IMPUTATION  -  value of a withheld column, from the remaining ones "
              f"({impute['dropout_rate']:.0%} hidden)")
        print(f"  {'vital':<20} {'MAE':>8} {'median MAE':>11} {'unit':<14} {'n':>6}")
        print("  " + "-"*62)
        for name,m in impute["features"].items():
            print(f"  {name:<20} {m['mae']:>8.2f} {m['baseline_mae']:>11.2f}  {UNITS.get(name,''):<14} {m['n']:>6,}")
        print(f"\n  {'macro-average MAE':<28} {impute['macro_mae']:>7.3f}   <- this model")
        print(f"  {'vs. training-median fill':<28} {impute['baseline_macro_mae']:>7.3f}")
        print(f"  Skill over median fill: {impute['skill_vs_median_pct']:+.2f}%")
    print()


def baseline_maes(windows,stats):
    """Doc-required reference points: last-observation-carried-forward and training median."""
    test=windows.split=="test"; targets,mask=windows.targets[test],windows.target_mask[test]
    persistence=np.repeat(windows.values[test][:,-1:,:],targets.shape[1],axis=1)
    out={}
    for name,prediction in (("persistence",persistence),("training_median",np.zeros_like(targets))):
        error=(np.abs((prediction-targets)*stats.scale)*mask).sum((0,1))/mask.sum((0,1))
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
    loader=DataLoader(ArrayDataset(windows,"test"),batch_size=int(cfg["training"]["batch_size"]))
    rate=float(cfg["task"].get("column_dropout_rate",0.0))
    # A head with zero loss weight was never trained, so scoring it would publish noise.
    weights=cfg["task"].get("loss_weights",{}) or {}
    score_drugs=bool(drugs) and float(weights.get("drug",1.0))>0
    if rate>0 and not float(weights.get("impute",1.0))>0: rate=0.0
    generator=torch.Generator(device=device).manual_seed(int(cfg["project"]["seed"]))

    absolute=np.zeros(n_features); squared=np.zeros_like(absolute); counts=np.zeros_like(absolute); per_window=[]
    drug_scores=[]; drug_labels=[]; drug_masks=[]
    imp_abs=np.zeros(n_features); imp_base=np.zeros(n_features); imp_counts=np.zeros(n_features)
    with torch.no_grad():
        for raw in loader:
            batch=tuple(item.to(device) for item in raw)
            values,observed,present,age,targets,target_mask,drug_targets,drug_mask=batch
            out=model(values,observed,present,age)
            error=(out["forecast"].cpu().numpy()-targets.cpu().numpy())*stats.scale.reshape(1,1,-1); m=target_mask.cpu().numpy()
            absolute+=(np.abs(error)*m).sum((0,1)); squared+=((error**2)*m).sum((0,1)); counts+=m.sum((0,1))
            per_window.append(np.abs(error)*m)
            if score_drugs and "drug_logits" in out:
                drug_scores.append(torch.sigmoid(out["drug_logits"]).cpu().numpy())
                drug_labels.append(drug_targets.cpu().numpy()); drug_masks.append(drug_mask.cpu().numpy())
            if rate>0:
                keep=drop_columns(present,rate,generator)
                hidden=((1.0-keep)*observed[:,-1,:]).cpu().numpy()
                truth=values[:,-1,:].cpu().numpy()
                prediction=model(*apply_column_dropout(values,observed,present,age,keep))["impute"].cpu().numpy()
                imp_abs+=(np.abs(prediction-truth)*stats.scale*hidden).sum(0)
                # Median fill is the reference an imputer must beat: scaled median is 0.
                imp_base+=(np.abs(0.0-truth)*stats.scale*hidden).sum(0)
                imp_counts+=hidden.sum(0)

    metrics={name:{"mae":float(absolute[i]/counts[i]),"rmse":float(np.sqrt(squared[i]/counts[i])),"n":int(counts[i])}
             for i,name in enumerate(meta["features"]) if counts[i]}
    test=windows.split=="test"; target_mask=windows.target_mask[test]
    persistence=np.repeat(windows.values[test][:,-1:,:],windows.targets.shape[1],axis=1)
    persistence_error=np.abs((persistence-windows.targets[test])*stats.scale)*target_mask
    result={"forecast":{"macro_mae":float(np.mean([v["mae"] for v in metrics.values()])),
                        "baselines":baseline_maes(windows,stats),
                        "skill":bootstrap_skill(np.concatenate(per_window),persistence_error,target_mask),
                        "features":metrics}}

    if drug_scores:
        scores=np.concatenate(drug_scores); labels=np.concatenate(drug_labels); masks=np.concatenate(drug_masks)
        classes={}; 
        for index,name in enumerate(drugs):
            keep=masks[:,index].astype(bool)
            if keep.sum()<2: continue
            y=labels[keep,index]; s=scores[keep,index]
            prevalence=float(y.mean())
            if prevalence in (0.0,1.0): continue  # a class with one label cannot be scored
            classes[name]={"auroc":roc_auc(s,y),"ap":average_precision(s,y),
                           "prevalence":round(prevalence,4),"n":int(keep.sum())}
        aurocs=[v["auroc"] for v in classes.values() if v["auroc"] is not None]
        aps=[v["ap"] for v in classes.values() if v["ap"] is not None]
        result["medication"]={"classes":classes,
                             "macro_auroc":float(np.mean(aurocs)) if aurocs else 0.0,
                             "macro_ap":float(np.mean(aps)) if aps else 0.0,
                             "baseline_ap":float(np.mean([v["prevalence"] for v in classes.values()])) if classes else 0.0,
                             "scored_classes":len(classes),"unscorable_classes":len(drugs)-len(classes)}

    if rate>0 and imp_counts.sum():
        features={name:{"mae":float(imp_abs[i]/imp_counts[i]),"baseline_mae":float(imp_base[i]/imp_counts[i]),
                        "n":int(imp_counts[i])} for i,name in enumerate(meta["features"]) if imp_counts[i]}
        macro=float(np.mean([v["mae"] for v in features.values()]))
        base=float(np.mean([v["baseline_mae"] for v in features.values()]))
        result["imputation"]={"dropout_rate":rate,"macro_mae":macro,"baseline_macro_mae":base,
                              "skill_vs_median_pct":round(100.0*(base-macro)/base,2) if base else 0.0,
                              "features":features}

    (artifact/"test_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)) if args.json else report(result,meta,cfg)
if __name__=="__main__": main()
