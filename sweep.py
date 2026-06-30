import logging
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging
from src.config import Config
from main import run_full_test_suite

setup_logging(level=logging.INFO)

# ── Phase 2: Hyperparameter search ───────────────────────────────────────────────
# Trains on the small dataset (20 segments / 500 ts) to find the best
# PHYSICS_LOSS_WEIGHT before committing to long data-efficiency runs.
# After running PHASE=2, open `mlflow ui` and note the best PLW.
HYPERPARAM_SWEEP = [
    {"NUM_SEGMENTS": 20, "TIMESTEPS_PER_SEGMENT": 500, "PHYSICS_LOSS_WEIGHT": 1e-5},
    {"NUM_SEGMENTS": 20, "TIMESTEPS_PER_SEGMENT": 500, "PHYSICS_LOSS_WEIGHT": 1e-4},
    {"NUM_SEGMENTS": 20, "TIMESTEPS_PER_SEGMENT": 500, "PHYSICS_LOSS_WEIGHT": 5e-4},
    {"NUM_SEGMENTS": 20, "TIMESTEPS_PER_SEGMENT": 500, "PHYSICS_LOSS_WEIGHT": 1e-3},
    {"NUM_SEGMENTS": 20, "TIMESTEPS_PER_SEGMENT": 500, "PHYSICS_LOSS_WEIGHT": 1e-2},
]

# ── Phase 3: Final model run ──────────────────────────────────────────────────────
# Trains the final model at T48 Optuna-optimal hyperparameters on the full dataset.
# These come from pinn_hpo_relative_v2 Trial 48 (best Pareto point: detection
# parity with standard, +17pp GMM classification gap, +75% Mahalanobis).
BEST_PHYSICS_LOSS_WEIGHT = 2.71e-05
BEST_LEARNING_RATE       = 1.52e-03
BEST_HIDDEN_DIM          = 256
BEST_WINDOW_SIZE         = 30

DATA_EFF_SWEEP = [
    {"NUM_SEGMENTS": 20, "TIMESTEPS_PER_SEGMENT": 500},
]

# ── Select which phase to run ─────────────────────────────────────────────────────
# PHASE = 2  →  hyperparameter search (run this first)
# PHASE = 3  →  final model on full dataset at Optuna-optimal hyperparameters
PHASE = 3

# ─────────────────────────────────────────────────────────────────────────────────

if PHASE == 2:
    sweep_configs = HYPERPARAM_SWEEP
    result_prefix = os.path.join("results", "hyperparam")
elif PHASE == 3:
    sweep_configs = [
        {
            **entry,
            "PHYSICS_LOSS_WEIGHT":    BEST_PHYSICS_LOSS_WEIGHT,
            "LEARNING_RATE":          BEST_LEARNING_RATE,
            "HIDDEN_DIM":             BEST_HIDDEN_DIM,
            "WINDOW_SIZE":            BEST_WINDOW_SIZE,
            "NUM_WORKERS":            0,
        }
        for entry in DATA_EFF_SWEEP
    ]
    result_prefix = os.path.join("results", "final")
else:
    raise ValueError(f"Unknown PHASE={PHASE}. Set PHASE=2 or PHASE=3.")

for overrides in sweep_configs:
    cfg = Config()
    for key, val in overrides.items():
        setattr(cfg, key, val)

    seg = cfg.NUM_SEGMENTS
    ts  = cfg.TIMESTEPS_PER_SEGMENT
    plw = cfg.PHYSICS_LOSS_WEIGHT

    if PHASE == 2:
        cfg.RESULTS_DIR = os.path.join(result_prefix, f"plw{plw:.0e}")
    else:
        cfg.RESULTS_DIR = os.path.join(result_prefix, f"seg{seg}_ts{ts}_plw{plw:.0e}")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    run_full_test_suite(cfg, train_again=True)

