"""
notebooks/utils/nb_setup.py
===========================
Shared setup utilities for all training notebooks (02–05).

Eliminates the 12 patterns duplicated across every notebook:
  - Project root bootstrap
  - MLflow SQLite backend + experiment management
  - Gold layer parquet discovery
  - Device setup (MPS / CPU)
  - HuggingFace cache configuration
  - TrainingArguments factory
  - Warmup steps calculation
  - TensorBoard + MLflow monitoring print block
  - Active run guard
  - Registry promotion

Usage (Phase 1 of any notebook):
    from notebooks.utils.nb_setup import setup_experiment
    ctx = setup_experiment(cfg)
    # ctx.PROJECT_ROOT, ctx.device, ctx.GOLD_PARQUET_PATH,
    # ctx.EXP_DIR, ctx.mlflow_run all available immediately

See individual function docstrings for standalone usage.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import torch


# ---------------------------------------------------------------------------
# Keys excluded from MLflow param logging
# ---------------------------------------------------------------------------
# - Internal keys (prefixed with _) are implementation details, not params
# - warmup_ratio is converted to warmup_steps inside make_training_args()
#   and must not be re-logged by MLflow (causes "param already logged" error)
_MLFLOW_EXCLUDE_KEYS = {"warmup_ratio"}


# ---------------------------------------------------------------------------
# Experiment context dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentContext:
    """
    All resolved paths and objects produced by setup_experiment().
    Replaces ~80 lines of boilerplate in every notebook Phase 1.
    """
    PROJECT_ROOT:       Path
    GOLD_PARQUET_PATH:  Path
    EXP_DIR:            Path
    HF_CACHE_DIR:       Path
    DB_PATH:            Path
    device:             torch.device
    mlflow_run:         Any          # mlflow.ActiveRun
    cfg:                dict         = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Bootstrap
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Walk up from cwd until artifacts.yaml is found.
    Adds the root to sys.path so `import src` works.
    Raises FileNotFoundError if artifacts.yaml is not found.
    """
    current = Path.cwd()
    while current != current.parent:
        if (current / "artifacts.yaml").exists():
            root = current.resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
                print(f"   📦 Project root added to sys.path: .../{root.name}")
            else:
                print(f"   📦 Project root already in sys.path: .../{root.name}")
            return root
        current = current.parent
    raise FileNotFoundError(
        "Could not find artifacts.yaml in current or parent directories.\n"
        "Ensure you are running from within the project tree."
    )


# ---------------------------------------------------------------------------
# 2. Gold layer discovery
# ---------------------------------------------------------------------------

def find_gold_parquet(project_root: Path) -> Path:
    """
    Finds the most recent medsynth_gold_apso_*.parquet in data/gold/.
    Returns the path. Raises FileNotFoundError if none found.
    """
    from src.config import config  # safe — project root already in sys.path
    gold_dir = config.resolve_path("data", "gold")
    parquet_files = sorted(gold_dir.glob("medsynth_gold_apso_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No Gold layer Parquet found in {gold_dir}\n"
            "Please run the EDA notebook (Phase 4) first."
        )
    path = parquet_files[-1]
    print(f"   📁 Gold layer: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 3. Device setup
# ---------------------------------------------------------------------------

def setup_device() -> torch.device:
    """
    Selects MPS (Apple Silicon) or CPU. Sets PYTORCH_ENABLE_MPS_FALLBACK.
    Returns torch.device.
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"   🚀 Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"   ⚠️  Device: CPU (MPS not available)")
    return device


# ---------------------------------------------------------------------------
# 4. HuggingFace cache
# ---------------------------------------------------------------------------

def setup_hf_cache(project_root: Path) -> Path:
    """
    Points HF_HOME and HF_HUB_CACHE to data/cache/ within the project.
    Sets read/connect timeouts. Returns the cache directory Path.
    """
    hf_cache = project_root / "data" / "cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]                = str(hf_cache)
    os.environ["HF_HUB_CACHE"]           = str(hf_cache)
    os.environ["HF_HUB_READ_TIMEOUT"]    = "120"
    os.environ["HF_HUB_CONNECT_TIMEOUT"] = "60"
    print(f"   📁 HF cache: {hf_cache}")
    return hf_cache


# ---------------------------------------------------------------------------
# 5. MLflow setup
# ---------------------------------------------------------------------------

def _mlflow_safe_params(cfg: dict) -> dict:
    """
    Filter cfg to only the params safe to log to MLflow.

    Excludes:
    - Lists and dicts (MLflow only accepts scalar params)
    - Keys starting with '_' (internal helpers: _notebook, _train_size)
    - Keys in _MLFLOW_EXCLUDE_KEYS (warmup_ratio — converted to
      warmup_steps by make_training_args; re-logging causes conflict)
    """
    return {
        k: v for k, v in cfg.items()
        if not isinstance(v, (list, dict))
        and not k.startswith("_")
        and k not in _MLFLOW_EXCLUDE_KEYS
    }


def setup_mlflow(
    project_root:    Path,
    experiment_name: str,
    run_name:        str,
    cfg:             dict,
) -> Any:
    """
    Configures the MLflow SQLite backend, closes any stale active run,
    opens a new run, and logs all scalar cfg values.

    Returns the active mlflow.ActiveRun object.
    """
    db_path = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(experiment_name)

    if mlflow.active_run():
        print(f"   🔄 Closing stale run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    run = mlflow.start_run(run_name=run_name)

    # Log only safe scalar params — excludes internal keys and warmup_ratio
    mlflow.log_params(_mlflow_safe_params(cfg))

    print(f"   📊 MLflow backend: {db_path.name}")
    print(f"   📊 Run:            {run.info.run_id[:8]}...")
    return run


def end_mlflow_run(final_metrics: dict | None = None) -> None:
    """
    Logs optional final_metrics dict and closes the active MLflow run.
    Safe to call even if no run is active.
    """
    if mlflow.active_run() is None:
        print("   ⚠️  No active MLflow run to close")
        return
    if final_metrics:
        mlflow.log_metrics(final_metrics)
    mlflow.end_run()
    print("   ✅ MLflow run closed")


def check_mlflow_run() -> bool:
    """
    Checks whether an MLflow run is active. Prints a warning if not.
    Returns True if active, False otherwise.
    """
    run = mlflow.active_run()
    if run is None:
        print("⚠️  WARNING: No active MLflow run. Metrics will not be logged.")
        print("   Re-run Phase 1 to initialise a fresh run before continuing.")
        return False
    print(f"   ✅ Active MLflow run: {run.info.run_id[:8]}...")
    return True


# ---------------------------------------------------------------------------
# 6. TrainingArguments factory
# ---------------------------------------------------------------------------

def make_training_args(
    output_dir:    Path | str,
    cfg:           dict,
    epoch_key:     str = "num_epochs",
    lr_key:        str = "learning_rate",
    batch_key:     str = "batch_size",
    report_to:     list[str] | None = None,
    save_limit:    int = 3,
    disable_tqdm:  bool = False,
    log_level:     str = "passive",
) -> "transformers.TrainingArguments":  # type: ignore[name-defined]
    """
    Creates a HuggingFace TrainingArguments object from a cfg dict.

    Handles:
    - warmup_steps calculation from warmup_ratio (deprecation fix)
    - MPS compatibility (fp16=False, dataloader_pin_memory=False)
    - Consistent eval_strategy / save_strategy / metric_for_best_model
    - TENSORBOARD_LOGGING_DIR env var (replaces deprecated logging_dir)

    Parameters
    ----------
    output_dir  : checkpoint output directory
    cfg         : experiment config dict containing training hyperparameters
    epoch_key   : key in cfg for num_epochs (default "num_epochs")
    lr_key      : key in cfg for learning_rate
    batch_key   : key in cfg for batch_size
    report_to   : list of reporters; defaults to ["tensorboard", "mlflow"]
    save_limit  : max checkpoints to retain
    disable_tqdm: suppress per-step progress bars (useful for Stage-2 loops)
    log_level   : HF Trainer log verbosity
    """
    from transformers import TrainingArguments

    if report_to is None:
        report_to = ["tensorboard", "mlflow"]

    num_epochs   = cfg[epoch_key]
    batch_size   = cfg[batch_key]
    lr           = cfg[lr_key]
    weight_decay = cfg.get("weight_decay", 0.01)
    seed         = cfg.get("seed", 42)
    warmup_ratio = cfg.get("warmup_ratio", 0.1)

    # Calculate warmup_steps — warmup_ratio is deprecated in HF >=5.2
    # Derived manually from total training steps to avoid the deprecation
    # warning and the MLflow "param already logged" conflict.
    train_size = cfg.get("_train_size", 0)
    if train_size > 0:
        total_steps  = (train_size // batch_size) * num_epochs
        warmup_steps = max(1, int(warmup_ratio * total_steps))
    else:
        warmup_steps = 0  # HF will use 0 — caller can patch if needed

    # Set TensorBoard log dir via env var (avoids HF deprecation warning)
    tb_dir = Path(output_dir).parent / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)

    return TrainingArguments(
        output_dir                   = str(output_dir),
        eval_strategy                = "epoch",
        save_strategy                = "epoch",
        load_best_model_at_end       = True,
        metric_for_best_model        = "macro_f1",
        greater_is_better            = True,
        save_total_limit             = save_limit,
        num_train_epochs             = num_epochs,
        per_device_train_batch_size  = batch_size,
        learning_rate                = lr,
        weight_decay                 = weight_decay,
        warmup_steps                 = warmup_steps,
        logging_steps                = 20,
        report_to                    = report_to,
        seed                         = seed,
        fp16                         = False,
        dataloader_pin_memory        = False,
        disable_tqdm                 = disable_tqdm,
        log_level                    = log_level,
    )


# ---------------------------------------------------------------------------
# 7. Monitoring print block
# ---------------------------------------------------------------------------

def print_monitoring_urls(
    project_root: Path,
    tb_dir:       Path,
    tb_port:      int = 6006,
    mlflow_port:  int = 5001,
) -> None:
    """
    Prints the TensorBoard and MLflow UI commands for this experiment.
    Call at the end of any trainer configuration phase.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"📈 TENSORBOARD:")
    print(f"   tensorboard --logdir '{tb_dir}' --port {tb_port}")
    print(f"\n📊 MLFLOW UI:")
    print(f"   mlflow ui --backend-store-uri sqlite:///{project_root}/mlflow.db --port {mlflow_port}")
    print(f"{sep}")


# ---------------------------------------------------------------------------
# 8. Registry promotion
# ---------------------------------------------------------------------------

def promote_to_registry(
    cfg:              dict,
    trainer:          Any,
    tokenizer:        Any,
    project_root:     Path,
    train_result:     Any,
    extra_metrics:    dict | None = None,
) -> Path:
    """
    Promotes the best checkpoint to the permanent registry.

    Saves:
    - model weights + tokenizer -> registry/{experiment_name}/model/
    - label_mapping.json        -> registry/{experiment_name}/
    - final_metrics.json        -> registry/{experiment_name}/
    - experiment_config.json    -> registry/{experiment_name}/
    - training_dashboard.png    -> registry/{experiment_name}/ (if exists)

    Returns the registry directory Path.
    """
    from src.config import config

    registry_base = config.resolve_path("outputs", "evaluations") / "registry"
    registry_dir  = registry_base / cfg["experiment_name"]
    registry_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Promoting {cfg['experiment_name']} to registry...")

    # 1. Save model + tokenizer
    model_dir = registry_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"   ✅ Model + tokenizer saved: {model_dir.name}/")

    # 2. Copy training dashboard if it exists
    exp_dir = config.resolve_path("outputs", "evaluations") / cfg["experiment_name"]
    dashboard_src = exp_dir / f"{cfg['experiment_name']}_dashboard.png"
    if dashboard_src.exists():
        shutil.copy(dashboard_src, registry_dir / "training_dashboard.png")
        print(f"   ✅ Dashboard copied")
    else:
        print(f"   ⚠️  Dashboard not found — skipping")

    # 3. Save label mapping
    model_config = trainer.model.config
    if hasattr(model_config, "label2id") and model_config.label2id:
        mapping_path = registry_dir / "label_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump({
                "label2id": model_config.label2id,
                "id2label":  {str(k): v for k, v in model_config.id2label.items()},
            }, f, indent=4)
        print(f"   ✅ Label mapping saved")
    else:
        print(f"   ⚠️  No label2id in model config — label_mapping.json not written")

    # 4. Capture best epoch metrics
    final_metrics: dict = {}
    if train_result is not None:
        final_metrics.update({k: v for k, v in train_result.metrics.items()})

    eval_logs = [
        log for log in trainer.state.log_history if "eval_macro_f1" in log
    ]
    best_epoch = None
    if eval_logs:
        best_eval_log = max(eval_logs, key=lambda x: x["eval_macro_f1"])
        final_metrics.update(best_eval_log)
        best_epoch = best_eval_log.get("epoch", "unknown")
        print(f"   📊 Best epoch: {best_epoch}")

    if extra_metrics:
        final_metrics.update(extra_metrics)

    metrics_path = registry_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "experiment":   cfg["experiment_name"],
            "model":        cfg.get("model_name", "unknown"),
            "label_scheme": cfg.get("label_scheme", "unknown"),
            "num_labels":   getattr(trainer.model.config, "num_labels", None),
            "num_epochs":   cfg.get("num_epochs", None),
            "best_epoch":   best_epoch,
            **final_metrics,
        }, f, indent=4)
    print(f"   ✅ Final metrics saved")
    print(f"   📊 Val Macro F1:  {final_metrics.get('eval_macro_f1', 0):.4f}")
    print(f"   📊 Val Accuracy:  {final_metrics.get('eval_accuracy', 0):.4f}")

    # 5. Save experiment config — use same exclusion rules as MLflow logging
    config_path = registry_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(_mlflow_safe_params(cfg), f, indent=4)
    print(f"   ✅ Experiment config saved")

    print(f"\n🏆 Registry: {registry_dir.resolve()}")
    return registry_dir


# ---------------------------------------------------------------------------
# 9. Master setup function — replaces the entire Phase 1 cell
# ---------------------------------------------------------------------------

def setup_experiment(cfg: dict) -> ExperimentContext:
    """
    Full Phase 1 setup in one call. Replaces ~80 lines per notebook.

    Performs in order:
    1. Find project root + add to sys.path
    2. Import and verify src.config
    3. Find gold layer parquet
    4. Configure HuggingFace cache
    5. Set up device (MPS / CPU)
    6. Configure MLflow + open run
    7. Log cfg to audit trail

    Returns ExperimentContext with all resolved paths and objects.
    """
    print("🔍 Setting up experiment environment...")

    # 1. Bootstrap
    print("\n[1/6] Project root...")
    project_root = find_project_root()

    # 2. Verify config singleton
    from src.config import config
    if Path(config.project_root) != project_root:
        raise RuntimeError(
            f"Project root mismatch!\n"
            f"  Bootstrap found: {project_root}\n"
            f"  Config reports:  {config.project_root}\n"
            f"  Ensure only one artifacts.yaml exists in your directory hierarchy."
        )
    print(f"   ✅ Config: {config.config_path}")

    # 3. Gold layer
    print("\n[2/6] Gold layer...")
    gold_path = find_gold_parquet(project_root)

    # 4. HF cache
    print("\n[3/6] HuggingFace cache...")
    hf_cache = setup_hf_cache(project_root)

    # 5. Device
    print("\n[4/6] Device...")
    device = setup_device()

    # 6. MLflow
    print("\n[5/6] MLflow...")
    exp_name = cfg["experiment_name"]
    run_name = f"{cfg['experiment_id']}_{cfg.get('description', 'run')[:30]}"
    db_path  = project_root / "mlflow.db"
    run      = setup_mlflow(project_root, exp_name, run_name, cfg)
    mlflow.log_param("gold_parquet", gold_path.name)
    mlflow.log_param("device",       str(device))

    # 7. Experiment output dir
    print("\n[6/6] Output directories...")
    exp_dir = config.resolve_path("outputs", "evaluations") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"   📁 Experiment dir: {exp_dir}")

    # 8. Audit trail
    config.log_event(
        phase="Phase 1: Experiment Configuration",
        action=f"{cfg['experiment_id'].lower()}_config_initialised",
        details={
            "experiment_id":   cfg["experiment_id"],
            "experiment_name": cfg["experiment_name"],
            "model_name":      cfg.get("model_name", "unknown"),
            "gold_parquet":    gold_path.name,
            "device":          str(device),
            "mlflow_db":       str(db_path),
        },
        notebook=cfg.get("_notebook", "unknown"),
    )

    print(f"\n{'='*60}")
    print(f"🔒 Experiment:  {exp_name}")
    print(f"🚀 Device:      {device.type.upper()}")
    print(f"📊 MLflow:      {db_path.name}")
    print(f"✅ Setup complete")
    print(f"{'='*60}")

    return ExperimentContext(
        PROJECT_ROOT      = project_root,
        GOLD_PARQUET_PATH = gold_path,
        EXP_DIR           = exp_dir,
        HF_CACHE_DIR      = hf_cache,
        DB_PATH           = db_path,
        device            = device,
        mlflow_run        = run,
        cfg               = cfg,
    )