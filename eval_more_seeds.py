"""
Parallel multi-seed, multi-frequency evaluation of saved T48 models.

For each (omega, seed) pair, in parallel:
  - runs the anomaly test suite on both models
  - computes detection AUC  (via run_detection_evaluation)
  - computes GMM micro accuracy + kNN micro accuracy (via compare_*_accuracy)

Reports mean ± SE for all three metrics across 30 seeds × 4 omegas.

Run:
    python eval_seeds.py
"""
import os
import logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── User config ───────────────────────────────────────────────────────────────
PHYSICS_LOSS_WEIGHT = 2.71e-05
LEARNING_RATE       = 1.52e-03
HIDDEN_DIM          = 256
WINDOW_SIZE         = 30
NUM_TEST_SEEDS      = 30
N_WORKERS           = 20

RUNS = [
    {
        "label":       "small (20 seg × 500 ts)",
        "results_dir": "results/final_small_dataset/seg20_ts500_plw3e-05",
    },
    {
        "label":       "large (200 seg × 2000 ts)",
        "results_dir": "results/final_large_dataset/seg200_ts2000_plw3e-05",
    },
]


# ── Worker — must be top-level for ProcessPoolExecutor + spawn ────────────────
def _eval_one_seed(args):
    """
    Evaluate one (omega, seed_idx) pair — runs in a subprocess.
    Returns (detection_df, clf_dict) with plain-Python/numpy types only.
    """
    omega, seed_idx, model_dir, cfg_kwargs, tb_pinn, tb_std = args

    # Suppress INFO chatter from inside the worker
    logging.disable(logging.INFO)

    import torch
    from src.config import Config
    from src.dataset import simulate_harmonic_oscillator
    from src.test_suite_runner import run_anomaly_test_suite
    from src.detection_metrics import run_detection_evaluation
    from src.quantitative_metrics import (
        compare_classification_accuracy,
        compare_gmm_classification_accuracy,
    )
    from main import load_model

    cfg = Config()
    for k, v in cfg_kwargs.items():
        setattr(cfg, k, v)

    device = torch.device("cpu")
    model_std,  scaler, _ = load_model(
        os.path.join(model_dir, "lstm_autoencoder_standard.pt"), device)
    model_pinn, _,      _ = load_model(
        os.path.join(model_dir, "lstm_autoencoder_pinn.pt"), device)

    x_clean = simulate_harmonic_oscillator(
        timesteps=cfg.TIMESTEPS, dt=cfg.DT, omega=omega,
        noise_std=cfg.TEST_NOISE,
        seed=cfg.RANDOM_STATE + 100 + seed_idx,
        amplitude=3.0, phase=0.5,
    )

    seed_offset = seed_idx * 100
    pinn_res = run_anomaly_test_suite(
        cfg, model_pinn, x_clean, scaler, device,
        model_name="Physics-Informed", omega=omega, dt=cfg.DT,
        seed_offset=seed_offset, verbose=False,
    )
    std_res = run_anomaly_test_suite(
        cfg, model_std, x_clean, scaler, device,
        model_name="Standard", omega=omega, dt=cfg.DT,
        seed_offset=seed_offset, verbose=False,
    )

    # ── Detection AUC ─────────────────────────────────────────────────────────
    det_df = run_detection_evaluation(
        pinn_results=pinn_res, standard_results=std_res,
        threshold_bundle_pinn=tb_pinn, threshold_bundle_std=tb_std,
        config=cfg, save=False, verbose=False,
    )
    det_df["omega"] = omega
    det_df["seed"]  = seed_idx

    # ── Classification: GMM micro + kNN micro ─────────────────────────────────
    knn_data = compare_classification_accuracy(pinn_res, std_res,
                                               window_size=cfg.WINDOW_SIZE)
    gmm_res  = compare_gmm_classification_accuracy(knn_data)

    clf = {
        "omega":          omega,
        "seed":           seed_idx,
        "pinn_gmm_micro": float(gmm_res["micro"]["metrics_pinn"]["accuracy"]),
        "std_gmm_micro":  float(gmm_res["micro"]["metrics_std"]["accuracy"]),
        "pinn_knn_micro": float(gmm_res["micro"]["knn_pinn_acc"]),
        "std_knn_micro":  float(gmm_res["micro"]["knn_std_acc"]),
    }

    return det_df, clf


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    import torch
    import pandas as pd
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from src.utils import setup_logging
    from src.config import Config
    from src.threshold import set_threshold
    from main import load_model, make_calibration_loader

    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    for run in RUNS:
        results_dir = run["results_dir"]
        model_dir   = os.path.join(results_dir, "saved_models")

        if not os.path.isdir(model_dir):
            logger.warning("saved_models not found at %s — skipping", model_dir)
            continue

        print(f"\n{'='*62}")
        print(f"Run : {run['label']}")
        print(f"Dir : {model_dir}")
        print(f"Jobs: {NUM_TEST_SEEDS} seeds × {4} omegas = "
              f"{NUM_TEST_SEEDS*4}  |  workers: {N_WORKERS}")
        print(f"{'='*62}\n")

        # ── Config ────────────────────────────────────────────────────────────
        cfg = Config()
        cfg.PHYSICS_LOSS_WEIGHT = PHYSICS_LOSS_WEIGHT
        cfg.LEARNING_RATE       = LEARNING_RATE
        cfg.HIDDEN_DIM          = HIDDEN_DIM
        cfg.WINDOW_SIZE         = WINDOW_SIZE
        cfg.NUM_TEST_SEEDS      = NUM_TEST_SEEDS
        cfg.NUM_WORKERS         = 0
        cfg.RESULTS_DIR         = results_dir

        cfg_kwargs = {k: getattr(cfg, k) for k in (
            "PHYSICS_LOSS_WEIGHT", "LEARNING_RATE", "HIDDEN_DIM",
            "WINDOW_SIZE", "TIMESTEPS", "DT", "TEST_NOISE",
            "RANDOM_STATE", "NUM_WORKERS",
        )}

        # ── Load models once (for scaler + threshold computation) ─────────────
        device = torch.device("cpu")
        model_std,  scaler, _ = load_model(
            os.path.join(model_dir, "lstm_autoencoder_standard.pt"), device)
        model_pinn, _,      _ = load_model(
            os.path.join(model_dir, "lstm_autoencoder_pinn.pt"), device)

        # ── Pre-compute threshold bundles (one per omega) ─────────────────────
        logger.info("Computing threshold bundles for %d omegas …", len(cfg.TEST_OMEGAS))
        threshold_bundles = {}
        for omega in cfg.TEST_OMEGAS:
            cal_loader = make_calibration_loader(omega, cfg, scaler, device)
            tb_p = set_threshold(model_pinn, cal_loader, omega, cfg.DT, scaler, device,
                                 cfg.WINDOW_SIZE, physics_loss_weight=cfg.PHYSICS_LOSS_WEIGHT)
            tb_s = set_threshold(model_std,  cal_loader, omega, cfg.DT, scaler, device,
                                 cfg.WINDOW_SIZE, physics_loss_weight=0.0)
            threshold_bundles[omega] = (tb_p, tb_s)

        # ── Build job list ────────────────────────────────────────────────────
        jobs = [
            (omega, seed_idx, model_dir, cfg_kwargs,
             threshold_bundles[omega][0], threshold_bundles[omega][1])
            for omega in cfg.TEST_OMEGAS
            for seed_idx in range(NUM_TEST_SEEDS)
        ]
        n_jobs = len(jobs)
        logger.info("Submitting %d jobs to %d workers …", n_jobs, N_WORKERS)

        # ── Parallel evaluation ───────────────────────────────────────────────
        det_records, clf_records = [], []
        done = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_eval_one_seed, job): job for job in jobs}
            for f in as_completed(futures):
                det_df, clf = f.result()
                det_records.append(det_df)
                clf_records.append(clf)
                done += 1
                if done % 20 == 0 or done == n_jobs:
                    logger.info("  %d / %d done", done, n_jobs)

        # ── Aggregate detection ───────────────────────────────────────────────
        all_det = pd.concat(det_records, ignore_index=True)
        det_agg = (
            all_det
            .groupby(["model", "omega", "anomaly_type"], sort=False)
            .agg(
                auc_mean=("auc", "mean"),
                auc_se  =("auc", lambda x: x.std() / len(x) ** 0.5),
            )
            .reset_index()
        )
        all_det.to_csv( os.path.join(results_dir, "multi_eval_all_seeds.csv"),  index=False)
        det_agg.to_csv(os.path.join(results_dir, "multi_eval_aggregated.csv"), index=False)

        # ── Aggregate classification ──────────────────────────────────────────
        clf_df = pd.DataFrame(clf_records)
        clf_df.to_csv(os.path.join(results_dir, "clf_all_seeds.csv"), index=False)

        # ── Summary ───────────────────────────────────────────────────────────
        def _stats(series):
            n = len(series)
            return series.mean(), series.std(), series.std() / n**0.5

        # Per-seed mean over omegas
        auc_per_seed = all_det.groupby(["model", "seed"])["auc"].mean().unstack("model")
        gap_auc = auc_per_seed["physics_informed"] - auc_per_seed["standard"]

        gap_gmm = clf_df["pinn_gmm_micro"] - clf_df["std_gmm_micro"]
        gap_knn = clf_df["pinn_knn_micro"] - clf_df["std_knn_micro"]

        print(f"\n─── {run['label']} ({NUM_TEST_SEEDS} seeds × {len(cfg.TEST_OMEGAS)} omegas) ───")

        for label, pinn_col, std_col, gap in [
            ("AUC (detection)",
             auc_per_seed["physics_informed"], auc_per_seed["standard"], gap_auc),
            ("GMM micro (classification)",
             clf_df["pinn_gmm_micro"], clf_df["std_gmm_micro"], gap_gmm),
            ("kNN micro (classification)",
             clf_df["pinn_knn_micro"], clf_df["std_knn_micro"], gap_knn),
        ]:
            pm, ps, pse = _stats(pinn_col)
            sm, ss, sse = _stats(std_col)
            gm, gs, gse = _stats(gap)
            t = gm / gse if gse > 0 else float("inf")
            print(f"\n  {label}")
            print(f"    PINN : {pm:.4f}  SD={ps:.4f}  SE={pse:.4f}")
            print(f"    Std  : {sm:.4f}  SD={ss:.4f}  SE={sse:.4f}")
            print(f"    Gap  : {gm:+.4f}  SD={gs:.4f}  SE={gse:.4f}  t={t:.1f}")

        print()

