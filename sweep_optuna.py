import json
import logging
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import optuna
from optuna.samplers import TPESampler, NSGAIISampler

from src.utils import setup_logging
from src.config import Config
from main import run_full_test_suite

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

OPTUNA_DB      = "sqlite:///optuna_hpo.db"
STUDY_NAME     = "pinn_hpo_relative_v2"
MLF_EXPERIMENT = "pinn_optuna_hpo"

# Reduce eval cost during HPO: fewer seeds, same omegas
HPO_NUM_TEST_SEEDS = 3


def _tag_latest_mlflow_run(experiment_name: str, trial_number: int) -> None:
    """Tag the most recent MLflow run in experiment with the Optuna trial number."""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            client.set_tag(runs[0].info.run_id, "optuna_trial", str(trial_number))
    except Exception:
        pass  # MLflow tagging is best-effort


def objective(trial: optuna.Trial) -> float:
    cfg = Config()

    # ── Hyperparameters ───────────────────────────────────────────────────────
    cfg.PHYSICS_LOSS_WEIGHT = trial.suggest_float("physics_loss_weight", 1e-6, 1e-1, log=True)
    cfg.LEARNING_RATE       = trial.suggest_float("learning_rate",       1e-4, 1e-2, log=True)
    cfg.HIDDEN_DIM          = trial.suggest_categorical("hidden_dim",    [64, 128, 256])
    cfg.WINDOW_SIZE         = trial.suggest_int("window_size",           20, 50, step=5)

    # ── Fixed HPO budget ──────────────────────────────────────────────────────
    cfg.NUM_SEGMENTS          = 20
    cfg.TIMESTEPS_PER_SEGMENT = 500
    cfg.NUM_TEST_SEEDS        = HPO_NUM_TEST_SEEDS
    cfg.NUM_WORKERS           = 0    # ensure deterministic DataLoader across trials
    cfg.MLFLOW_EXPERIMENT     = MLF_EXPERIMENT

    cfg.RESULTS_DIR = os.path.join("results", "optuna", f"trial_{trial.number:03d}")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    try:
        run_full_test_suite(cfg, train_again=True)
    except Exception as e:
        logger.warning("Trial %d failed: %s", trial.number, e)
        raise optuna.exceptions.TrialPruned()

    _tag_latest_mlflow_run(MLF_EXPERIMENT, trial.number)

    # ── Objective 1: detection quality (WHERE) ────────────────────────────────
    agg = pd.read_csv(os.path.join(cfg.RESULTS_DIR, "multi_eval_aggregated.csv"))
    pinn_auc = float(agg[agg["model"] == "physics_informed"]["auc_mean"].mean())
    std_auc  = float(agg[agg["model"] == "standard"]["auc_mean"].mean())

    # ── Objective 2: type-separation quality (WHAT) ───────────────────────────
    with open(os.path.join(cfg.RESULTS_DIR, "quantitative_summary.json")) as f:
        qsum = json.load(f)
    gmm_micro_acc = qsum["pinn_gmm_micro_acc"]
    trial.set_user_attr("std_auc",           std_auc)
    trial.set_user_attr("physics_premium",   pinn_auc - std_auc)
    trial.set_user_attr("pinn_knn_micro",    qsum["pinn_knn_micro_acc"])
    trial.set_user_attr("std_gmm_micro_acc", qsum["std_gmm_micro_acc"])
    trial.set_user_attr("gmm_micro_gap",     gmm_micro_acc - qsum["std_gmm_micro_acc"])

    logger.info(
        "Trial %d | PLW=%.2e LR=%.2e HD=%d WS=%d | "
        "AUC=%.4f(+%.4f)  GMM-micro=%.4f(gap=%.4f)",
        trial.number,
        cfg.PHYSICS_LOSS_WEIGHT, cfg.LEARNING_RATE, cfg.HIDDEN_DIM, cfg.WINDOW_SIZE,
        pinn_auc, pinn_auc - std_auc,
        gmm_micro_acc, gmm_micro_acc - qsum["std_gmm_micro_acc"],
    )

    # Return relative metrics: how much does physics training help over standard?
    # f1: physics premium on detection (pinn_auc - std_auc)
    # f2: GMM micro gap (pinn_gmm_micro - std_gmm_micro)
    return pinn_auc - std_auc, gmm_micro_acc - qsum["std_gmm_micro_acc"]


if __name__ == "__main__":
    # Multi-objective: NSGA-II is the standard sampler for Pareto optimisation
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=OPTUNA_DB,
        directions=["maximize", "maximize"],   # (detection AUC, GMM micro accuracy)
        sampler=NSGAIISampler(seed=42),
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=50, show_progress_bar=True)

    pareto = study.best_trials
    logger.info("=" * 60)
    logger.info("PARETO FRONT (%d trials)", len(pareto))
    logger.info("  %-6s  %-10s  %-10s  %-6s  %-6s  %-6s  %-6s",
                "Trial", "AUC-premium", "GMM-gap", "PLW", "LR", "HD", "WS")
    for t in sorted(pareto, key=lambda t: t.values[0], reverse=True):
        logger.info("  %-6d  %.4f      %.4f      %.0e  %.0e  %-6d  %-6d",
                    t.number, t.values[0], t.values[1],
                    t.params["physics_loss_weight"], t.params["learning_rate"],
                    t.params["hidden_dim"], t.params["window_size"])
    logger.info("=" * 60)
    logger.info("To inspect all trials:")
    logger.info("  optuna-dashboard sqlite:///optuna_hpo.db")
    logger.info("  mlflow ui --backend-store-uri mlruns")
