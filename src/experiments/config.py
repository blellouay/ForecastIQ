"""Central reproducible experiment configuration."""
from __future__ import annotations
from dataclasses import dataclass,asdict,field
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,platform,subprocess
from importlib.metadata import PackageNotFoundError,version

@dataclass(frozen=True)
class ExperimentConfig:
    horizon:int=28; total_days:int=1913; test_origin:int=1885; validation_folds:int=3
    random_seeds:tuple=(7,21,42,73,101,123,202,314,777,2026)
    sample_sizes:tuple=(400,2000,5000,10000)
    adi_threshold:float=1.32; cv2_threshold:float=0.49
    leading_zero_policy:str="preserve"
    alphas:tuple=(0.1,0.2,0.3)
    metrics:tuple=("MAE","MASE","WAPE","RMSSE","bias","inventory_cost_proxy")
    underforecast_penalty:float=5.0; overforecast_penalty:float=1.0
    evaluation_panel:str="data/processed/evaluation_panel.parquet"
    scenarios:tuple=("seen","unseen_product_store")
    validation_origins:tuple=(1801,1829,1857)
    backtest_origins:tuple=(1717,1745,1773,1801,1829,1857)
    dl_mode:str="fast"
    dl_fast_max_steps:int=20
    dl_research_max_steps:int=750
    dl_fast_input_size:int=112
    dl_research_input_size:int=168
    dl_fast_batch_size:int=16
    dl_research_batch_size:int=8
    dl_learning_rate:float=0.0005
    dl_validation_check_steps:int=50

    def __post_init__(self):
        if self.test_origin+self.horizon != self.total_days: raise ValueError("test boundary does not match total days")

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def save_manifest(path, experiment_name, config, *, sources=(), extra=None):
    try: commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip() or None
    except OSError: commit=None
    stamp=datetime.now(timezone.utc)
    packages={}
    for package in ("numpy","pandas","scikit-learn","lightgbm","torch","neuralforecast","pyarrow"):
        try: packages[package]=version(package)
        except PackageNotFoundError: packages[package]=None
    payload={"experiment_name":experiment_name,"experiment_id":f"{experiment_name}-{stamp:%Y%m%dT%H%M%SZ}",
             "timestamp_utc":stamp.isoformat(),"config":asdict(config),"python":platform.python_version(),
             "platform":platform.platform(),"package_versions":packages,"git_commit":commit,
             "source_hashes":{str(p):sha256(p) for p in sources if Path(p).exists()},"extra":extra or {}}
    destination=Path(path); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload
