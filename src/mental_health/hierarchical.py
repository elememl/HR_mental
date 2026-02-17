from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOWER_PATH = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
OUT_DIR = PROJECT_ROOT / "data" / "out" / "hierarchical"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Validation / utilities
# ----------------------------
def validate_distance_matrix(D: np.ndarray) -> None:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected square n×n matrix, got shape {D.shape}")

    if not np.isfinite(D).all():
        bad = int(np.size(D) - np.isfinite(D).sum())
        raise ValueError(f"Distance matrix has non-finite values (NaN/inf): {bad} entries")

    if not np.allclose(np.diag(D), 0.0, atol=1e-6):
        raise ValueError("Distance matrix diagonal is not ~0; check saved file.")

    if not np.allclose(D, D.T, atol=1e-6):
        raise ValueError("Distance matrix is not symmetric; check computation.")

    if np.min(D) < -1e-6:
        raise ValueError("Distance matrix has negative entries; invalid distances.")

    if np.max(D) > 1.0 + 1e-6:
        # Gower should be in [0,1], but allow tiny numeric drift.
        print(
            f"WARNING: max(D)={np.max(D):.6f} > 1.0. "
            "If you expected Gower, verify your computation."
        )


def tier_label(abs_corr: float) -> str:
    if abs_corr >= 0.8:
        return "STRONG (>= 0.8)"
    if abs_corr >= 0.6:
        return "MODERATE (0.6–0.8)"
    return "weak (< 0.6)"


def big_jumps_to_candidate_k(
    Z: np.ndarray,
    top_m: int = 12,
    min_k: int = 2,
    max_k: int = 30,
):
    """
    Find large increases in linkage distance (merge heights).

    heights has length n-1 where n is #points.
    diffs[j] = heights[j+1] - heights[j] corresponds to the "jump" into merge (j+1).

    Cutting BEFORE merge (j+1) yields:
        #clusters = n - (j+1)

    Returns a list of candidate k values with associated jump statistics.
    """
    n = Z.shape[0] + 1
    heights = Z[:, 2]
    if len(heights) < 2:
        return []

    diffs = np.diff(heights)  # length n-2
    idx = np.argsort(diffs)[::-1]  # largest first

    picks = []
    for j in idx[:top_m]:
        k = n - (j + 1)
        if not (min_k <= k <= max_k):
            continue

        lo = float(heights[j])
        hi = float(heights[j + 1])
        jump = float(diffs[j])
        cut_height = (lo + hi) / 2.0

        picks.append(
            {
                "k": int(k),
                "jump": jump,
                "cut_between": (lo, hi),
                "cut_height": float(cut_height),
                "merge_index": int(j + 1),
            }
        )

    # de-duplicate ks while preserving order
    seen = set()
    uniq = []
    for p in picks:
        if p["k"] not in seen:
            uniq.append(p)
            seen.add(p["k"])
    return uniq


def plot_truncated_dendrogram(
    Z: np.ndarray,
    title: str,
    p: int = 30,
    truncate_mode: str = "lastp",
    save_path: Path | None = None,
) -> None:
    plt.figure(figsize=(12, 6), dpi=150)
    dendrogram(
        Z,
        truncate_mode=truncate_mode,  # "lastp" (default) or "level"
        p=p,
        show_leaf_counts=True,
        leaf_rotation=0,
        leaf_font_size=10,
    )
    plt.title(title)
    plt.xlabel("Cluster (truncated)")
    plt.ylabel("Linkage distance (precomputed Gower)")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved dendrogram: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_linkage_heights(Z: np.ndarray, title: str, save_path: Path) -> None:
    """
    Plot the merge heights (Z[:,2]) across merge steps.
    Large upward jumps suggest a natural cut just before the jump.
    """
    h = Z[:, 2]
    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(h)
    plt.title(title)
    plt.xlabel("Merge step")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved linkage heights: {save_path}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if not GOWER_PATH.exists():
        raise FileNotFoundError(f"Missing: {GOWER_PATH}")

    D = np.load(GOWER_PATH)
    validate_distance_matrix(D)

    n = D.shape[0]
    print(f"Loaded distance matrix: shape={D.shape}, dtype={D.dtype}")
    print(f"min(D)={np.min(D):.6f}, max(D)={np.max(D):.6f}, mean(D)={np.mean(D):.6f}")

    # SciPy linkage expects condensed distances
    d_condensed = squareform(D, checks=False)

    # Linkage methods to compare
    methods = ["average", "complete", "single"]

    Zs: dict[str, np.ndarray] = {}
    for m in methods:
        print(f"\nComputing linkage: {m}")
        Zs[m] = linkage(d_condensed, method=m)

    # ---- Dendrograms (truncated for readability) ----
    print("\n=== DENDROGRAMS (TRUNCATED) ===")
    for m in methods:
        out_png = OUT_DIR / f"dendrogram_{m}_trunc30_lastp.png"
        plot_truncated_dendrogram(
            Zs[m],
            title=f"Hierarchical clustering ({m} linkage, truncated)",
            p=30,
            truncate_mode="lastp",
            save_path=out_png,
        )

        # Optional second view (often clearer branching)
        out_png2 = OUT_DIR / f"dendrogram_{m}_trunc5_level.png"
        plot_truncated_dendrogram(
            Zs[m],
            title=f"Hierarchical clustering ({m} linkage, level-truncated)",
            p=5,
            truncate_mode="level",
            save_path=out_png2,
        )

    # ---- Linkage heights plots ----
    print("\n=== LINKAGE HEIGHTS (MERGE DISTANCES) ===")
    for m in methods:
        out_png = OUT_DIR / f"linkage_heights_{m}.png"
        plot_linkage_heights(Zs[m], title=f"Linkage heights ({m})", save_path=out_png)

    # ---- Identify large linkage jumps & propose candidate k ----
    print("\n=== LARGE LINKAGE JUMPS (heuristic for candidate k) ===")
    for m in methods:
        picks = big_jumps_to_candidate_k(Zs[m], top_m=12, min_k=2, max_k=25)

        print(f"\nMethod: {m}")
        if not picks:
            print("  No candidate jumps found in the specified k range.")
            continue

        # Print top few
        for p_ in picks[:6]:
            lo, hi = p_["cut_between"]
            print(
                f"  k≈{p_['k']:>2}  jump={p_['jump']:.4f}  "
                f"cut_height≈{p_['cut_height']:.4f}  "
                f"between heights [{lo:.4f}, {hi:.4f}]"
            )

        suggested_ks = [p_["k"] for p_ in picks[:5]]
        print("  Suggested candidate k values (for PAM follow-up):", suggested_ks)

if __name__ == "__main__":
    main()
