"""Random search over model and optimiser settings; keeps the lowest validation loss.

Every trial trains on the same prepared windows (symlinked, never rebuilt) and is judged by
the same weighted validation loss, so trials are comparable and the test split is never
looked at. Trial 0 is the base config unchanged, so the winner is never worse than it.
Writes `configs/<base>_best.yaml`, whose artifact_dir holds the winning checkpoint.
"""
import argparse,copy,json,os,random,subprocess,sys
from pathlib import Path
import yaml
from mimic_forecaster.config import load_config,resolve_path

SPACE={("tokenizer","embed_dim"):[64,128,192],("tokenizer","patch_size"):[2,3,4,6],
       ("encoder","n_layers"):[2,3,4],("encoder","dropout"):[0.1,0.2,0.3],
       ("training","learning_rate"):[3e-4,1e-3,2e-3],("training","weight_decay"):[1e-5,1e-3,1e-2]}

def sample(base,rng):
    cfg=copy.deepcopy(base)
    for (section,key),choices in SPACE.items(): cfg[section][key]=rng.choice(choices)
    cfg["encoder"]["ffn_dim"]=2*cfg["tokenizer"]["embed_dim"]
    return cfg

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--trials",type=int,default=12)
    args=parser.parse_args(); base_path=Path(args.config).resolve(); cfg=load_config(base_path)
    prepared=resolve_path(cfg,cfg["project"]["artifact_dir"])/"prepared"
    if not (prepared/"windows.npz").exists(): raise SystemExit(f"run mimic-prepare first: {prepared} is empty")
    raw=yaml.safe_load(base_path.read_text()); rng=random.Random(int(cfg["project"]["seed"])); results=[]
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    for trial in range(args.trials):
        trial_cfg=copy.deepcopy(raw) if trial==0 else sample(raw,rng)
        trial_cfg["project"]["artifact_dir"]=f"{cfg['project']['artifact_dir']}/sweep/trial_{trial:02d}"
        # Trial configs sit in configs/ so the project root resolves exactly as the base's does.
        path=base_path.parent/f"sweep_{base_path.stem}_{trial:02d}.yaml"; path.write_text(yaml.safe_dump(trial_cfg,sort_keys=False))
        artifact=resolve_path(cfg,trial_cfg["project"]["artifact_dir"]); artifact.mkdir(parents=True,exist_ok=True)
        if not (artifact/"prepared").exists(): (artifact/"prepared").symlink_to(prepared.resolve(),target_is_directory=True)
        print(f"trial {trial:02d}: "+", ".join(f"{k}={trial_cfg[s][k]}" for s,k in SPACE),flush=True)
        done=subprocess.run([sys.executable,"-m","mimic_forecaster.cli.train","--config",str(path),"--json"],env=env,capture_output=True,text=True)
        if done.returncode:
            print(f"  failed: {done.stderr.strip().splitlines()[-1] if done.stderr.strip() else done.returncode}",flush=True); continue
        metrics=json.loads((artifact/"training_metrics.json").read_text())
        results.append({"trial":trial,"config":str(path),"val_loss":metrics["best_val_loss"],"epochs":metrics["epochs_ran"]})
        print(f"  val_loss {metrics['best_val_loss']:.4f} after {metrics['epochs_ran']} epochs",flush=True)
    if not results: raise SystemExit("every trial failed")
    results.sort(key=lambda r:r["val_loss"]); best=results[0]
    best_path=base_path.parent/f"{base_path.stem}_best.yaml"; best_path.write_text(Path(best["config"]).read_text())
    print(json.dumps({"best":best,"best_config":str(best_path),"leaderboard":results},indent=2))

if __name__=="__main__": main()
