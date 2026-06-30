import json
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
import os, logging

logger = logging.getLogger(__name__)

# Grouping derived from empirical centroid angles of ground-truth anomalous windows
# in the PINN 2D (log MSE, log physics) space, measured from the normal centroid:
#   mse_dominant    34–37°: amplitude_spikes, frequency_violations
#   both_fail       45–51°: phase_discontinuities, damping_violations, white_noise_bursts
#   physics_dominant 61–79°: harmonic_contamination, missing_data, dc_offset_shifts,
#                            amplitude_modulation
# Boundaries chosen at midpoints between groups: 42° and 57°.
MACRO_CLASS_MAP = {
    'baseline':              'normal',
    'amplitude_spikes':      'mse_dominant',
    'frequency_violations':  'mse_dominant',
    'phase_discontinuities': 'both_fail',
    'damping_violations':    'both_fail',
    'white_noise_bursts':    'both_fail',
    'harmonic_contamination':'physics_dominant',
    'missing_data':          'physics_dominant',
    'dc_offset_shifts':      'physics_dominant',
    'amplitude_modulation':  'physics_dominant',
}
MACRO_CLASS_ORDER = ['mse_dominant', 'both_fail', 'physics_dominant']


def compute_physics_loss_reduction(pinn_results, standard_results):
    logger.info("=== Physics Loss Reduction Analysis ===")

    for anom_type in pinn_results.keys():
        pinn_mean = np.mean(pinn_results[anom_type].physics_values)
        std_mean  = np.mean(standard_results[anom_type].physics_values)
        reduction_factor = std_mean / pinn_mean if pinn_mean > 0 else np.inf
        reduction_pct    = ((std_mean - pinn_mean) / std_mean) * 100 if std_mean > 0 else 0
        logger.info("%-25s: %8.1fx reduction (%5.1f%% decrease)", anom_type, reduction_factor, reduction_pct)

    all_pinn_phy = np.concatenate([r.physics_values for r in pinn_results.values()])
    all_std_phy  = np.concatenate([r.physics_values for r in standard_results.values()])
    overall_reduction = np.mean(all_std_phy) / np.mean(all_pinn_phy)
    overall_pct       = ((np.mean(all_std_phy) - np.mean(all_pinn_phy)) / np.mean(all_std_phy)) * 100
    logger.info("%-25s: %8.1fx reduction (%5.1f%% decrease)", "OVERALL", overall_reduction, overall_pct)


def compare_classification_accuracy(pinn_results, standard_results, window_size=30):
    """
    Classifies anomaly macro-type from (log MSE, log physics) features.

    Uses only ground-truth anomalous windows (those overlapping the injected event)
    rather than all windows — clean windows collapse all anomaly types onto the
    normal mean and dilute the per-type signal.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    MICRO_CLASS_ORDER = [t for t in pinn_results if t != 'baseline']

    def prepare_data(results):
        X_list, y_macro, y_micro = [], [], []
        for anom_type, result in results.items():
            if anom_type == 'baseline':
                continue
            macro_name = MACRO_CLASS_MAP.get(anom_type, 'physics_dominant')
            if macro_name == 'normal':
                continue
            macro_idx = MACRO_CLASS_ORDER.index(macro_name)
            micro_idx = MICRO_CLASS_ORDER.index(anom_type)

            anom_start = int(result.anomaly_indices[0])
            duration   = result.anomaly_duration
            n_windows  = len(result.mse_values)

            i_lo = max(0, anom_start - window_size + 1)
            i_hi = min(n_windows, anom_start + max(duration, 1))
            anom_idx = list(range(i_lo, i_hi))
            if not anom_idx:
                continue

            mse_w = np.array(result.mse_values)[anom_idx]
            phy_w = np.array(result.physics_values)[anom_idx]
            X = np.column_stack([np.log10(mse_w + 1e-10), np.log10(phy_w + 1e-10)])
            X_list.append(X)
            y_macro.append(np.full(len(anom_idx), macro_idx))
            y_micro.append(np.full(len(anom_idx), micro_idx))

        return np.vstack(X_list), np.concatenate(y_macro), np.concatenate(y_micro)

    X_pinn, y_pinn_macro, y_pinn_micro = prepare_data(pinn_results)
    X_std,  y_std_macro,  y_std_micro  = prepare_data(standard_results)

    n_macro  = len(MACRO_CLASS_ORDER)
    n_splits = min(5, min(np.bincount(y_pinn_macro).min(), np.bincount(y_std_macro).min()))
    knn = KNeighborsClassifier(n_neighbors=5)

    pinn_scores = cross_val_score(knn, X_pinn, y_pinn_macro, cv=n_splits)
    std_scores  = cross_val_score(knn, X_std,  y_std_macro,  cv=n_splits)

    logger.info("=== Macro-class Classification Accuracy (anomalous windows only) ===")
    logger.info("%d classes, random baseline %.0f%%", n_macro, 100 / n_macro)
    logger.info("Physics-Informed: %.3f +/- %.3f", pinn_scores.mean(), pinn_scores.std())
    logger.info("Standard:         %.3f +/- %.3f", std_scores.mean(), std_scores.std())
    logger.info("Improvement:      %.1f percentage points",
                (pinn_scores.mean() - std_scores.mean()) * 100)

    return {
        'X_pinn': X_pinn, 'y_pinn_macro': y_pinn_macro, 'y_pinn_micro': y_pinn_micro,
        'X_std':  X_std,  'y_std_macro':  y_std_macro,  'y_std_micro':  y_std_micro,
        'macro_order': MACRO_CLASS_ORDER,
        'micro_order': MICRO_CLASS_ORDER,
    }


class GMMClassifierWrapper:
    """
    Wraps a fitted GaussianMixture so it plugs into run_e2e_classification.
    Predictions with max posterior < confidence_threshold are returned as -1 (uncertain).
    """
    def __init__(self, gmm, component_to_class, confidence_threshold=0.7):
        self.gmm = gmm
        self.component_to_class = component_to_class
        self.confidence_threshold = confidence_threshold

    def predict(self, X):
        probs = self.gmm.predict_proba(X)
        confidence = probs.max(axis=1)
        raw = probs.argmax(axis=1)
        mapped = np.array([self.component_to_class.get(int(c), 0) for c in raw])
        mapped[confidence < self.confidence_threshold] = -1
        return mapped

    def predict_proba(self, X):
        return self.gmm.predict_proba(X)


def fit_gmm_classifier(X, y_true, n_components, random_state=42):
    """
    Fits a GaussianMixture unsupervised on X, then uses the Hungarian algorithm to
    align arbitrary component IDs to ground-truth class IDs from y_true.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        n_init=10,
        max_iter=500,
        reg_covar=1e-4,
    )
    gmm.fit(X)

    cluster_labels = gmm.predict(X)

    # cost[i, j] = number of samples in GMM component i belonging to ground-truth class j
    cost = np.zeros((n_components, n_components))
    for c, t in zip(cluster_labels, y_true):
        if int(t) < n_components:
            cost[int(c), int(t)] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    component_to_class = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    return gmm, component_to_class


def evaluate_gmm(gmm, component_to_class, X, y_true):
    """
    Evaluates a fitted GMM against ground-truth labels after Hungarian alignment.
    Returns dict with accuracy.
    """
    cluster_labels = gmm.predict(X)
    y_pred = np.array([component_to_class.get(int(c), 0) for c in cluster_labels])
    acc = float((y_pred == np.asarray(y_true)).mean())
    return {'accuracy': acc}


def gmm_precision_coverage_curve(gmm, component_to_class, X, y_true):
    """
    Sweeps confidence threshold θ from 0 to 0.99 and records the precision-coverage tradeoff.

    At each θ:
      - coverage  = fraction of samples with max(posterior) >= θ
      - precision = accuracy on that classified subset

    Returns (thresholds, coverages, precisions) as numpy arrays of length 100.
    """
    probs = gmm.predict_proba(X)
    confidences = probs.max(axis=1)
    raw_preds = np.array([component_to_class.get(int(c), 0) for c in probs.argmax(axis=1)])
    correct = (raw_preds == np.asarray(y_true))

    thresholds = np.linspace(0.0, 0.99, 100)
    coverages, precisions = [], []
    for theta in thresholds:
        mask = confidences >= theta
        coverages.append(float(mask.mean()))
        precisions.append(float(correct[mask].mean()) if mask.sum() > 0 else np.nan)

    return thresholds, np.array(coverages), np.array(precisions)


def compare_gmm_classification_accuracy(knn_data, random_state=42):
    """
    Fits GMM classifiers at macro (3-class) and micro (N-class) level.
    Logs accuracy vs. kNN supervised upper bound.

    Returns nested dict keyed by 'macro'/'micro', each containing fitted GMMs,
    evaluation metrics, kNN reference accuracy, and precision-coverage curve data.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    n_macro = len(knn_data['macro_order'])
    n_micro = len(knn_data['micro_order'])

    gmm_results = {}

    for level, n_comp, X_pinn, y_pinn, X_std, y_std in [
        ('macro', n_macro,
         knn_data['X_pinn'], knn_data['y_pinn_macro'],
         knn_data['X_std'],  knn_data['y_std_macro']),
        ('micro', n_micro,
         knn_data['X_pinn'], knn_data['y_pinn_micro'],
         knn_data['X_std'],  knn_data['y_std_micro']),
    ]:
        logger.info("=== GMM %s-class Classification (%d components) ===", level, n_comp)
        logger.info("Random baseline: %.0f%%", 100.0 / n_comp)

        gmm_pinn, map_pinn = fit_gmm_classifier(X_pinn, y_pinn, n_comp, random_state)
        gmm_std,  map_std  = fit_gmm_classifier(X_std,  y_std,  n_comp, random_state)

        m_pinn = evaluate_gmm(gmm_pinn, map_pinn, X_pinn, y_pinn)
        m_std  = evaluate_gmm(gmm_std,  map_std,  X_std,  y_std)

        logger.info("Physics-Informed — acc=%.3f", m_pinn['accuracy'])
        logger.info("Standard         — acc=%.3f", m_std['accuracy'])

        knn = KNeighborsClassifier(n_neighbors=5)
        n_splits = min(5, min(np.bincount(y_pinn).min(), np.bincount(y_std).min()))
        knn_pinn_acc = float(cross_val_score(knn, X_pinn, y_pinn, cv=n_splits).mean())
        knn_std_acc  = float(cross_val_score(knn, X_std,  y_std,  cv=n_splits).mean())

        logger.info("kNN supervised upper bound — PINN: %.3f  Std: %.3f", knn_pinn_acc, knn_std_acc)
        logger.info("GMM gap — PINN: %.1f pp  |  Std: %.1f pp",
                    (knn_pinn_acc - m_pinn['accuracy']) * 100,
                    (knn_std_acc  - m_std['accuracy'])  * 100)

        gmm_results[level] = {
            'gmm_pinn':     gmm_pinn,
            'map_pinn':     map_pinn,
            'gmm_std':      gmm_std,
            'map_std':      map_std,
            'metrics_pinn': m_pinn,
            'metrics_std':  m_std,
            'knn_pinn_acc': knn_pinn_acc,
            'knn_std_acc':  knn_std_acc,
            'curve_pinn':   gmm_precision_coverage_curve(gmm_pinn, map_pinn, X_pinn, y_pinn),
            'curve_std':    gmm_precision_coverage_curve(gmm_std,  map_std,  X_std,  y_std),
        }

    return gmm_results


def run_full_quantitative_analysis(pinn_results, standard_results, save_dir="results", window_size=30):
    """
    Runs quantitative analyses: physics loss reduction, kNN and GMM classification.
    """
    logger.info("="*70)
    logger.info("QUANTITATIVE ANALYSIS")
    logger.info("="*70)

    compute_physics_loss_reduction(pinn_results, standard_results)

    knn_data = compare_classification_accuracy(pinn_results, standard_results, window_size=window_size)

    logger.info("-"*70)
    logger.info("GMM UNSUPERVISED CLASSIFICATION")
    logger.info("-"*70)
    gmm_results = compare_gmm_classification_accuracy(knn_data)
    knn_data['gmm_results'] = gmm_results

    summary = {
        "pinn_gmm_micro_acc":  float(gmm_results["micro"]["metrics_pinn"]["accuracy"]),
        "std_gmm_micro_acc":   float(gmm_results["micro"]["metrics_std"]["accuracy"]),
        "pinn_gmm_macro_acc":  float(gmm_results["macro"]["metrics_pinn"]["accuracy"]),
        "std_gmm_macro_acc":   float(gmm_results["macro"]["metrics_std"]["accuracy"]),
        "pinn_knn_micro_acc":  float(gmm_results["micro"]["knn_pinn_acc"]),
        "std_knn_micro_acc":   float(gmm_results["micro"]["knn_std_acc"]),
    }
    summary_path = os.path.join(save_dir, "quantitative_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved quantitative summary to %s", summary_path)

    logger.info("="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)

    return knn_data
