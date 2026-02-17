from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, silhouette_samples


# ==========================================================
# PAM (K-MEDOIDS) ON PRECOMPUTED GOWER DISTANCES
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOWER_PATH = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Primary candidate k values + secondary checks
K_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
RANDOM_STATE = 42


def validate_distance_matrix(D: np.ndarray) -> None:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected square n×n distance matrix, got shape {D.shape}")
    if not np.allclose(np.diag(D), 0.0, atol=1e-6):
        raise ValueError("Distance matrix diagonal is not ~0.")
    if not np.allclose(D, D.T, atol=1e-6):
        raise ValueError("Distance matrix is not symmetric.")
    if np.min(D) < -1e-6:
        raise ValueError("Distance matrix has negative entries.")
    # Gower is typically in [0, 1], but do not hard-fail if scaling differs slightly.
    if np.max(D) > 1.0 + 1e-6:
        print(f"WARNING: max(D)={np.max(D):.6f} > 1.0. If you expected Gower, verify scaling.")


def run_pam(D: np.ndarray, k: int) -> dict:
    model = KMedoids(
        n_clusters=k,
        metric="precomputed",
        method="pam",
        init="k-medoids++",
        random_state=RANDOM_STATE,
    )

    labels = model.fit_predict(D)

    sil_avg = float(silhouette_score(D, labels, metric="precomputed"))
    sil = silhouette_samples(D, labels, metric="precomputed")

    sizes = np.bincount(labels, minlength=k)

    return {
        "k": int(k),
        "silhouette_avg": sil_avg,
        "silhouette_min": float(np.min(sil)),
        "silhouette_q25": float(np.percentile(sil, 25)),
        "silhouette_q50": float(np.percentile(sil, 50)),
        "silhouette_q75": float(np.percentile(sil, 75)),
        "min_cluster_size": int(np.min(sizes)),
        "max_cluster_size": int(np.max(sizes)),
        "cluster_sizes": sizes.tolist(),
        "medoid_indices": model.medoid_indices_.tolist(),
        "labels": labels.astype(np.int32),
    }

# visualization (minimal, standard, no annotations)

def _plot_silhouette_vs_k(df: pd.DataFrame, out_path: Path) -> None:
    ks = df["k"].to_numpy(dtype=int)
    sil = df["silhouette_avg"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, sil, marker="o")
    ax.set_title("Silhouette vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Average silhouette")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_cluster_sizes_by_k(df: pd.DataFrame, out_path: Path) -> None:
    ks = df["k"].tolist()
    size_lists = df["cluster_sizes"].tolist()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.boxplot(size_lists, tick_labels=[str(k) for k in ks], showfliers=True)
    ax.set_title("Cluster size distribution by k")
    ax.set_xlabel("k")
    ax.set_ylabel("Cluster size")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# DONT delete below
def main() -> None:
    if not GOWER_PATH.exists():
        raise FileNotFoundError(f"Missing Gower matrix: {GOWER_PATH}")

    D = np.load(GOWER_PATH)
    validate_distance_matrix(D)

    n = D.shape[0]
    print(f"Loaded Gower distance matrix: shape={D.shape}")
    print(f"Distance stats: min={np.min(D):.6f}, max={np.max(D):.6f}, mean={np.mean(D):.6f}")

    rows = []
    for k in K_LIST:
        print(f"\nRunning PAM (k-medoids) for k={k} ...")
        res = run_pam(D, k)

        # Save labels + medoids per k
        np.save(OUT_DIR / f"pam_labels_k{k}.npy", res["labels"])
        np.save(OUT_DIR / f"pam_medoids_k{k}.npy", np.array(res["medoid_indices"], dtype=np.int32))

        rows.append({kk: vv for kk, vv in res.items() if kk not in ("labels",)})

        print(
            f"k={k:>2} | sil_avg={res['silhouette_avg']:.4f} | "
            f"sil_q25/50/75=({res['silhouette_q25']:.4f}, {res['silhouette_q50']:.4f}, {res['silhouette_q75']:.4f}) | "
            f"sizes={res['cluster_sizes']}"
        )

    # Save summary table
    df = pd.DataFrame(rows).sort_values("k")
    out_csv = OUT_DIR / "pam_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary table: {out_csv}")
    print(f"Saved per-k labels/medoids in: {OUT_DIR}")

    # ----------------------------
    # V2 visuals
    # ----------------------------
    sil_plot_path = OUT_DIR / "silhouette_vs_k.png"
    _plot_silhouette_vs_k(df, sil_plot_path)
    print(f"Saved: {sil_plot_path}")

    size_plot_path = OUT_DIR / "cluster_sizes_by_k.png"
    _plot_cluster_sizes_by_k(df, size_plot_path)
    print(f"Saved: {size_plot_path}")


if __name__ == "__main__":
    main()
