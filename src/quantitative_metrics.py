import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_cluster_separation_metrics(results, model_name="Model"):
    """
    Computes quantitative metrics for how well anomaly types are separated.
    
    Metrics:
    - Silhouette Score: [-1, 1], higher is better (measures cluster cohesion)
    - Davies-Bouldin Index: [0, inf), lower is better (ratio of within/between cluster distances)
    - Calinski-Harabasz Index: [0, inf), higher is better (ratio of between/within cluster dispersion)
    """
    all_mse = []
    all_physics = []
    all_labels = []
    label_names = []
    
    for idx, (anom_type, result) in enumerate(results.items()):
        all_mse.extend(result.mse_values)
        all_physics.extend(result.physics_values)
        all_labels.extend([idx] * len(result.mse_values))
        label_names.append(anom_type)
    
    X = np.column_stack([
        np.log10(np.array(all_mse) + 1e-10),
        np.log10(np.array(all_physics) + 1e-10)
    ])
    
    labels = np.array(all_labels)
    
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    print(f"\n=== {model_name} Separation Metrics ===")
    print(f"Silhouette Score: {silhouette:.4f} (higher is better, range [-1, 1])")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f} (higher is better)")
    
    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "X": X,
        "labels": labels,
        "label_names": label_names
    }


def compute_pairwise_separability(results, model_name="Model"):
    """
    Computes pairwise separability between each anomaly type.
    Returns matrix showing how distinguishable each pair is.
    """
    anomaly_types = list(results.keys())
    n_types = len(anomaly_types)
    
    separability_matrix = np.zeros((n_types, n_types))
    
    for i, type_i in enumerate(anomaly_types):
        for j, type_j in enumerate(anomaly_types):
            if i == j:
                separability_matrix[i, j] = 0
                continue
            
            mse_i = np.log10(results[type_i].mse_values + 1e-10)
            phy_i = np.log10(results[type_i].physics_values + 1e-10)
            X_i = np.column_stack([mse_i, phy_i])
            
            mse_j = np.log10(results[type_j].mse_values + 1e-10)
            phy_j = np.log10(results[type_j].physics_values + 1e-10)
            X_j = np.column_stack([mse_j, phy_j])
            
            mean_i = X_i.mean(axis=0)
            mean_j = X_j.mean(axis=0)
            
            cov_i = np.cov(X_i.T)
            cov_j = np.cov(X_j.T)
            pooled_cov = (cov_i + cov_j) / 2
            
            try:
                inv_cov = np.linalg.inv(pooled_cov)
                mahalanobis_dist = np.sqrt((mean_i - mean_j).T @ inv_cov @ (mean_i - mean_j))
                separability_matrix[i, j] = mahalanobis_dist
            except np.linalg.LinAlgError:
                euclidean_dist = np.linalg.norm(mean_i - mean_j)
                separability_matrix[i, j] = euclidean_dist
    
    print(f"\n=== {model_name} Pairwise Separability (Mahalanobis Distance) ===")
    print("Larger values indicate better separation between anomaly types")
    
    return separability_matrix, anomaly_types


def plot_separability_heatmap(sep_matrix, anomaly_types, model_name, save_dir="results"):
    """
    Plots heatmap of pairwise separability.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    labels = [at.replace("_", "\n") for at in anomaly_types]
    
    sns.heatmap(sep_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Mahalanobis Distance'})
    
    plt.title(f"{model_name} - Pairwise Anomaly Separability", 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"separability_heatmap_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def compute_physics_loss_reduction(pinn_results, standard_results):
    """
    Computes how much physics loss is reduced by physics-informed model.
    """
    print("\n=== Physics Loss Reduction Analysis ===")
    
    for anom_type in pinn_results.keys():
        pinn_mean = np.mean(pinn_results[anom_type].physics_values)
        std_mean = np.mean(standard_results[anom_type].physics_values)
        
        reduction_factor = std_mean / pinn_mean if pinn_mean > 0 else np.inf
        reduction_pct = ((std_mean - pinn_mean) / std_mean) * 100 if std_mean > 0 else 0
        
        print(f"{anom_type:25s}: {reduction_factor:8.1f}x reduction ({reduction_pct:5.1f}% decrease)")
    
    all_pinn_phy = np.concatenate([r.physics_values for r in pinn_results.values()])
    all_std_phy = np.concatenate([r.physics_values for r in standard_results.values()])
    
    overall_reduction = np.mean(all_std_phy) / np.mean(all_pinn_phy)
    overall_pct = ((np.mean(all_std_phy) - np.mean(all_pinn_phy)) / np.mean(all_std_phy)) * 100
    
    print(f"\n{'OVERALL':25s}: {overall_reduction:8.1f}x reduction ({overall_pct:5.1f}% decrease)")


def compare_classification_accuracy(pinn_results, standard_results):
    """
    Simulates classification task: can you identify anomaly type from (MSE, Physics)?
    Uses k-nearest neighbors for simplicity.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    def prepare_data(results):
        X_list = []
        y_list = []
        for idx, (anom_type, result) in enumerate(results.items()):
            X = np.column_stack([
                np.log10(result.mse_values + 1e-10),
                np.log10(result.physics_values + 1e-10)
            ])
            X_list.append(X)
            y_list.append(np.full(len(result.mse_values), idx))
        
        return np.vstack(X_list), np.concatenate(y_list)
    
    X_pinn, y_pinn = prepare_data(pinn_results)
    X_std, y_std = prepare_data(standard_results)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    
    pinn_scores = cross_val_score(knn, X_pinn, y_pinn, cv=5)
    std_scores = cross_val_score(knn, X_std, y_std, cv=5)
    
    print("\n=== Classification Accuracy (5-fold CV with KNN) ===")
    print(f"Physics-Informed: {pinn_scores.mean():.3f} +/- {pinn_scores.std():.3f}")
    print(f"Standard:         {std_scores.mean():.3f} +/- {std_scores.std():.3f}")
    print(f"Improvement:      {(pinn_scores.mean() - std_scores.mean())*100:.1f} percentage points")


def run_full_quantitative_analysis(pinn_results, standard_results, save_dir="results"):
    """
    Runs all quantitative analyses and generates comparison report.
    """
    print("\n" + "="*70)
    print("QUANTITATIVE SEPARATION ANALYSIS")
    print("="*70)
    
    pinn_metrics = compute_cluster_separation_metrics(pinn_results, "Physics-Informed")
    std_metrics = compute_cluster_separation_metrics(standard_results, "Standard")
    
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    silh_improvement = pinn_metrics["silhouette"] - std_metrics["silhouette"]
    db_improvement = std_metrics["davies_bouldin"] - pinn_metrics["davies_bouldin"]
    
    print(f"Silhouette improvement: {silh_improvement:+.4f}")
    print(f"Davies-Bouldin improvement: {db_improvement:+.4f}")
    
    if silh_improvement > 0.05:
        print("Physics-informed model shows BETTER cluster separation")
    elif silh_improvement > 0:
        print("Physics-informed model shows SLIGHT improvement in separation")
    else:
        print("Models show similar separation quality")
    
    pinn_sep, types = compute_pairwise_separability(pinn_results, "Physics-Informed")
    std_sep, _ = compute_pairwise_separability(standard_results, "Standard")
    
    plot_separability_heatmap(pinn_sep, types, "Physics-Informed", save_dir)
    plot_separability_heatmap(std_sep, types, "Standard", save_dir)
    
    avg_pinn_sep = pinn_sep[np.triu_indices_from(pinn_sep, k=1)].mean()
    avg_std_sep = std_sep[np.triu_indices_from(std_sep, k=1)].mean()
    
    print(f"\nAverage pairwise separation:")
    print(f"  Physics-Informed: {avg_pinn_sep:.3f}")
    print(f"  Standard:         {avg_std_sep:.3f}")
    print(f"  Improvement:      {(avg_pinn_sep/avg_std_sep - 1)*100:+.1f}%")
    
    compute_physics_loss_reduction(pinn_results, standard_results)
    
    compare_classification_accuracy(pinn_results, standard_results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")
