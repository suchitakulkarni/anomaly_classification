"""
Regenerate all plots for both final datasets without retraining.
Loads saved models from results/final_*/ and reruns inference + visualisation only.

Usage:
    python replot.py
"""
import logging
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging
from src.config import Config
from main import run_full_test_suite

setup_logging(level=logging.INFO)

BEST_PHYSICS_LOSS_WEIGHT = 2.71e-05
BEST_LEARNING_RATE       = 1.52e-03
BEST_HIDDEN_DIM          = 256
BEST_WINDOW_SIZE         = 30

CONFIGS = [
    {"NUM_SEGMENTS": 20,  "TIMESTEPS_PER_SEGMENT": 500,  "label": "small"},
    {"NUM_SEGMENTS": 200, "TIMESTEPS_PER_SEGMENT": 2000, "label": "large"},
]

for entry in CONFIGS:
    cfg = Config()
    cfg.PHYSICS_LOSS_WEIGHT    = BEST_PHYSICS_LOSS_WEIGHT
    cfg.LEARNING_RATE          = BEST_LEARNING_RATE
    cfg.HIDDEN_DIM             = BEST_HIDDEN_DIM
    cfg.WINDOW_SIZE            = BEST_WINDOW_SIZE
    cfg.NUM_SEGMENTS           = entry["NUM_SEGMENTS"]
    cfg.TIMESTEPS_PER_SEGMENT  = entry["TIMESTEPS_PER_SEGMENT"]
    cfg.NUM_WORKERS            = 0

    seg = cfg.NUM_SEGMENTS
    ts  = cfg.TIMESTEPS_PER_SEGMENT
    plw = cfg.PHYSICS_LOSS_WEIGHT
    cfg.RESULTS_DIR = os.path.join("results", f"final_{entry['label']}_dataset",
                                   f"seg{seg}_ts{ts}_plw{plw:.0e}")

    print(f"\n{'='*60}")
    print(f"Replotting {entry['label']} dataset → {cfg.RESULTS_DIR}")
    print(f"{'='*60}")
    run_full_test_suite(cfg, train_again=False)
