# pam_pcoa_k6.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAM_DIR = PROJECT_ROOT / "data" / "out" / "pam"
OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam_post"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOWER_PATH = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"

K = 6
LABELS_PATH = PAM_DIR / f"pam_labels_k{K}.npy"


def load_gower() -> np.ndarray:
    if not GOWER_PATH.exists():
        raise FileNotFoundError(f"Missing Gower distance matrix: {GOWER_PATH}")
    D = np.load(GOWER_PATH)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape {D.shape}")
    # force symmetry (numerical safety)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D.astype(float)


def load_labels() -> np.ndarray:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")
    lab = np.load(LABELS_PATH).astype(int)
    if lab.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {lab.shape}")
    return lab


def pcoa(D: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Principal Coordinates Analysis (PCoA) / Classical MDS.

    Steps:
      1) Square distances: D^2
      2) Double-center: B = -0.5 * J D^2 J, where J = I - 11^T/n
      3) Eigendecompose B
      4) Coordinates = V * sqrt(Lambda) for positive eigenvalues only

    Returns:
      coords: (n, m) array for m = min(n_components, #positive eigenvalues)
      explained: (m,) fraction of total positive eigenvalue mass
    """
    n = D.shape[0]
    D2 = D ** 2

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D2 @ J)

    # Symmetrize B (numerical safety)
    B = 0.5 * (B + B.T)

    # Eigen decomposition (B is symmetric)
    evals, evecs = np.linalg.eigh(B)

    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Keep only positive eigenvalues (PCoA can yield small negatives due to non-Euclidean distances)
    pos_mask = evals > 1e-12
    evals_pos = evals[pos_mask]
    evecs_pos = evecs[:, pos_mask]

    if evals_pos.size == 0:
        raise ValueError("No positive eigenvalues found; cannot construct PCoA coordinates.")

    m = min(n_components, evals_pos.size)
    evals_pos = evals_pos[:m]
    evecs_pos = evecs_pos[:, :m]

    coords = evecs_pos * np.sqrt(evals_pos)

    explained = evals_pos / np.sum(evals_pos) if np.sum(evals_pos) > 0 else np.zeros_like(evals_pos)
    return coords, explained


def plot_pcoa_2d(coords: np.ndarray, explained: np.ndarray, labels: np.ndarray, save_path: Path) -> None:
    x = coords[:, 0]
    y = coords[:, 1]
    clusts = np.unique(labels)

    plt.figure(figsize=(10, 7), dpi=150)
    for c in clusts:
        mask = labels == c
        plt.scatter(x[mask], y[mask], s=10, alpha=0.8, label=f"Cluster {c}")

    xlab = f"PCoA1 ({explained[0]*100:.1f}%)" if explained.size >= 1 else "PCoA1"
    ylab = f"PCoA2 ({explained[1]*100:.1f}%)" if explained.size >= 2 else "PCoA2"
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title("PCoA (Gower) – k=6 clusters (2D)")
    plt.legend(markerscale=1.5, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_pcoa_3d(coords: np.ndarray, explained: np.ndarray, labels: np.ndarray, save_path: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    clusts = np.unique(labels)

    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    for c in clusts:
        mask = labels == c
        ax.scatter(x[mask], y[mask], z[mask], s=10, alpha=0.8, label=f"Cluster {c}")

    xlab = f"PCoA1 ({explained[0]*100:.1f}%)" if explained.size >= 1 else "PCoA1"
    ylab = f"PCoA2 ({explained[1]*100:.1f}%)" if explained.size >= 2 else "PCoA2"
    zlab = f"PCoA3 ({explained[2]*100:.1f}%)" if explained.size >= 3 else "PCoA3"

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    ax.set_title("PCoA (Gower) – k=6 clusters (3D)")
    ax.legend(markerscale=1.5, frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    D = load_gower()
    labels = load_labels()

    if D.shape[0] != labels.shape[0]:
        raise ValueError(f"Row mismatch: D is {D.shape[0]}x{D.shape[1]} but labels has {labels.shape[0]}")

    # Need at least 2D for the 2D plot; ask for 3, but handle if only 2 available
    coords, explained = pcoa(D, n_components=3)

    # 2D
    if coords.shape[1] < 2:
        raise ValueError("PCoA returned <2 components; cannot make 2D plot.")
    out2d = OUT_DIR / "pcoa_gower_k6_2d.png"
    plot_pcoa_2d(coords[:, :2], explained[:2], labels, out2d)
    print(f"Saved: {out2d}")

    # 3D
    if coords.shape[1] >= 3:
        out3d = OUT_DIR / "pcoa_gower_k6_3d.png"
        plot_pcoa_3d(coords[:, :3], explained[:3], labels, out3d)
        print(f"Saved: {out3d}")
    else:
        print("PCoA returned only 2 positive components; 3D plot skipped.")

    # Quick console summary
    pct = (explained[: min(3, explained.size)] * 100.0).round(2)
    print(f"Explained (positive-eigenvalue mass): {pct.tolist()} % for first components")


if __name__ == "__main__":
    main()

