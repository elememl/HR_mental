# pam_post_validate.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
from itertools import combinations

# NEW: suppress Matplotlib deprecation spam + support panel plots
import warnings
from matplotlib import MatplotlibDeprecationWarning

# Plotly for Sankey diagrams (recommended)
import plotly.graph_objects as go


# ==========================================================
# POST-CLUSTER VALIDATION (STABILITY):
# - ARI table (printed)
# - Adjacent-transition stability metrics table (single)
# - Sankey diagrams: adjacent + global (k4→k5→k6→k7)
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAM_DIR = PROJECT_ROOT / "data" / "out" / "pam"
OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam_post"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS = [4, 5, 6, 7]

# For split detection:
SPLIT_THRESHOLD = 0.80

# If you want row-wise metrics weighted by row sizes:
WEIGHTED = True


# ==========================================================
# FEATURE IMPORTANCE HEATMAPS BY FEATURE TYPE
# - Uses the same precomputed Gower distance matrix as PAM
# - Evaluates k = 5, 6, 7
#
# Medoid selection:
# - Uses the precomputed Gower distance matrix (drivers_gower.npy)
#
# Importance definitions:
# - Ordinal/non-binary features:
#     mean pairwise |medoid values| normalized by feature IQR on full dataset
# - Binary features (including rare one-hot indicators):
#     mean pairwise |medoid values| (no IQR normalization; scale naturally in [0,1])
#
# Visualization:
# - Two separate heatmaps:
#     1) Ordinal/non-binary heatmap (single consistent scale across k)
#     2) Binary heatmap (fixed scale [0,1] across k)
# - Rows sorted by importance at k=5 (descending) within each heatmap
# - No clustering of rows/columns
# ==========================================================

GOWER_PATH = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
ENCODED_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"
FEATURE_IMPORTANCE_KS = [5, 6, 7]

# NEW: how many features to include in the combined “panel” plots
TOP_PANEL_N = 10


def load_encoded_drivers() -> tuple[np.ndarray, list[str]]:
    """
    Load the final encoded dataset (observations × features).

    Expected: numeric encoded matrix (standardized numeric and/or one-hot).
    If respondent_id exists, it is dropped.
    """
    if not ENCODED_PATH.exists():
        raise FileNotFoundError(f"Missing encoded drivers file: {ENCODED_PATH}")

    df = pd.read_csv(ENCODED_PATH)
    if "respondent_id" in df.columns:
        df = df.drop(columns=["respondent_id"])

    X = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    return X, df.columns.tolist()


def compute_cluster_medoids_from_gower(D: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """
    Compute medoid indices for each cluster using the same precomputed Gower
    distance matrix used by PAM (k-medoids).

    Medoid definition: point minimizing the sum of distances to all other points
    within its cluster (classic PAM medoid criterion).
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape {D.shape}")
    if labels.ndim != 1 or labels.shape[0] != D.shape[0]:
        raise ValueError("Label vector length must match distance matrix size.")

    medoids: dict[int, int] = {}
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        if idx.size == 1:
            medoids[int(c)] = int(idx[0])
            continue

        subD = D[np.ix_(idx, idx)]
        sums = subD.sum(axis=1)
        medoids[int(c)] = int(idx[int(np.argmin(sums))])
    return medoids


def mean_pairwise_abs_diff(values: np.ndarray) -> float:
    """Mean pairwise absolute difference (upper triangle) for a 1D array."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    diffs = [abs(a - b) for a, b in combinations(v.tolist(), 2)]
    return float(np.mean(diffs)) if diffs else 0.0


def split_feature_types(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Split features into:
      - binary: all finite values in {0,1} (including rare one-hot indicators)
      - ordinal/non-binary: everything else

    Returns:
      (binary_mask, ordinal_mask) boolean arrays of length p
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    _, p = X.shape
    if len(feature_names) != p:
        raise ValueError("feature_names length must match X columns.")

    binary_mask = np.zeros(p, dtype=bool)

    for j in range(p):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            # Treat fully-missing columns as non-binary (keeps them out of binary heatmap)
            binary_mask[j] = False
            continue
        uniq = np.unique(col)
        # binary if subset of {0,1}
        binary_mask[j] = np.all(np.isin(uniq, [0.0, 1.0]))

    ordinal_mask = ~binary_mask
    return binary_mask, ordinal_mask


def compute_feature_importance_matrix_ordinal(
    X: np.ndarray,
    feature_names: list[str],
    D: np.ndarray,
    label_map: dict[int, np.ndarray],
    ks: list[int],
    ordinal_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Ordinal/non-binary feature importance:
      mean pairwise |medoid values| normalized by IQR(feature) on full dataset
    """
    n, p = X.shape
    if D.shape[0] != n:
        raise ValueError("Distance matrix size must match X rows.")
    if ordinal_mask.shape[0] != p:
        raise ValueError("ordinal_mask length must match #features.")

    Xo = X[:, ordinal_mask]
    names_o = [feature_names[i] for i in np.flatnonzero(ordinal_mask)]

    if Xo.shape[1] == 0:
        return pd.DataFrame(index=[], columns=[f"k{k}" for k in ks], dtype=float)

    q25 = np.nanpercentile(Xo, 25, axis=0)
    q75 = np.nanpercentile(Xo, 75, axis=0)
    iqr = q75 - q25

    # if IQR is 0 or invalid, normalized importance is undefined -> NaN
    iqr = np.where(np.isfinite(iqr) & (iqr > 0), iqr, np.nan)

    out = pd.DataFrame(index=names_o, columns=[f"k{k}" for k in ks], dtype=float)

    for k in ks:
        labels = label_map[k]
        medoid_idx = compute_cluster_medoids_from_gower(D, labels)
        medoid_rows = np.array([medoid_idx[c] for c in sorted(medoid_idx.keys())], dtype=int)
        medoid_vals = Xo[medoid_rows, :]

        raw = np.array(
            [mean_pairwise_abs_diff(medoid_vals[:, j]) for j in range(medoid_vals.shape[1])],
            dtype=float,
        )
        out[f"k{k}"] = raw / iqr

    return out


def compute_feature_importance_matrix_binary(
    X: np.ndarray,
    feature_names: list[str],
    D: np.ndarray,
    label_map: dict[int, np.ndarray],
    ks: list[int],
    binary_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Binary feature importance (includes rare features):
      mean pairwise |medoid values|, no IQR normalization
    This stays interpretable for sparse one-hot columns and is naturally in [0,1].
    """
    n, p = X.shape
    if D.shape[0] != n:
        raise ValueError("Distance matrix size must match X rows.")
    if binary_mask.shape[0] != p:
        raise ValueError("binary_mask length must match #features.")

    Xb = X[:, binary_mask]
    names_b = [feature_names[i] for i in np.flatnonzero(binary_mask)]

    if Xb.shape[1] == 0:
        return pd.DataFrame(index=[], columns=[f"k{k}" for k in ks], dtype=float)

    out = pd.DataFrame(index=names_b, columns=[f"k{k}" for k in ks], dtype=float)

    for k in ks:
        labels = label_map[k]
        medoid_idx = compute_cluster_medoids_from_gower(D, labels)
        medoid_rows = np.array([medoid_idx[c] for c in sorted(medoid_idx.keys())], dtype=int)
        medoid_vals = Xb[medoid_rows, :]

        raw = np.array(
            [mean_pairwise_abs_diff(medoid_vals[:, j]) for j in range(medoid_vals.shape[1])],
            dtype=float,
        )
        out[f"k{k}"] = raw

    return out


def plot_heatmap(
    mat: pd.DataFrame,
    save_path: Path,
    title: str,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Generic heatmap:
      - sorts rows by k5 descending (if k5 exists)
      - uses a single consistent color scale across all k in this heatmap
      - does not cluster rows/columns
    """
    if mat.empty:
        raise ValueError(f"Heatmap matrix is empty: {title}")

    mat2 = mat.copy()
    if "k5" in mat2.columns:
        mat2 = mat2.sort_values("k5", ascending=False)

    data = mat2.to_numpy(dtype=float)
    finite = np.isfinite(data)
    auto_vmin = float(np.nanmin(data)) if finite.any() else 0.0
    auto_vmax = float(np.nanmax(data)) if finite.any() else 1.0

    if vmin is None:
        vmin = auto_vmin
    if vmax is None:
        vmax = auto_vmax

    fig_h = max(6.0, 0.18 * mat2.shape[0])
    fig_w = 7.0
    plt.figure(figsize=(fig_w, fig_h), dpi=150)
    im = plt.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)
    plt.xticks(range(mat2.shape[1]), mat2.columns.tolist(), rotation=0)
    plt.yticks(range(mat2.shape[0]), mat2.index.tolist(), fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ==========================================================
# NEW: combined “panel” plots (rows=features, cols=k=5/6/7)
# ==========================================================

def _cluster_ids(labels: np.ndarray) -> list[int]:
    return sorted(np.unique(labels).astype(int).tolist())


def _add_left_row_titles(fig: plt.Figure, axes: np.ndarray, row_titles: list[str], fontsize: int = 8) -> None:
    """
    Add one left-side row title per row (outside the axes), aligned to the vertical
    center of that row. Assumes axes is 2D: [nrows, ncols].
    """
    nrows = axes.shape[0]
    for ridx in range(nrows):
        # Use first column axis position to compute y center for the row
        bbox = axes[ridx, 0].get_position()
        y_mid = 0.5 * (bbox.y0 + bbox.y1)
        # Place text slightly left of the subplot area (in figure coords)
        fig.text(0.01, y_mid, row_titles[ridx], ha="left", va="center", fontsize=fontsize)


def plot_panel_ordinal_boxplots_by_k(
    X: np.ndarray,
    feature_names: list[str],
    label_map: dict[int, np.ndarray],
    ks: list[int],
    features: list[str],
    save_path: Path,
) -> None:
    """
    One figure:
      rows = features
      cols = ks (k5,k6,k7)
      each cell = boxplot across clusters for that (feature, k)

    UPDATED: Remove per-row y-axis labels and instead add one left-side row title
    per feature (outside the axes) to avoid stacked y-label overlap.
    """
    if len(features) == 0:
        raise ValueError("No features provided for ordinal panel plot.")

    nrows = len(features)
    ncols = len(ks)

    fig_w = 4.2 * ncols
    fig_h = max(6.0, 0.75 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=150, squeeze=False)

    for cidx, k in enumerate(ks):
        labels = label_map[k]
        clusts = _cluster_ids(labels)

        for ridx, feat in enumerate(features):
            ax = axes[ridx, cidx]
            if feat not in feature_names:
                ax.axis("off")
                continue

            j = feature_names.index(feat)
            data_by_cluster = []
            for c in clusts:
                vals = X[labels == c, j]
                vals = vals[np.isfinite(vals)]
                data_by_cluster.append(vals)

            ax.boxplot(
                data_by_cluster,
                tick_labels=[str(c) for c in clusts],  # modern name (no deprecation)
                showfliers=False,
            )

            if ridx == 0:
                ax.set_title(f"k{k}", fontsize=10)

            # UPDATED: remove per-axis y-labels (feature names) to avoid overlap
            ax.set_ylabel("")

            if ridx != nrows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Cluster", fontsize=8)

            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Cluster-wise distributions (ordinal): top features across k", fontsize=12)

    # UPDATED: create room for left-side row titles, then add them
    fig.subplots_adjust(left=0.14)
    _add_left_row_titles(fig, axes, features, fontsize=8)

    plt.tight_layout(rect=[0.14, 0, 1, 0.98])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_panel_binary_bars_by_k(
    X: np.ndarray,
    feature_names: list[str],
    label_map: dict[int, np.ndarray],
    ks: list[int],
    features: list[str],
    save_path: Path,
) -> None:
    """
    One figure:
      rows = features
      cols = ks (k5,k6,k7)
      each cell = bar chart of proportion(1) by cluster for that (feature, k)

    UPDATED: Remove per-row y-axis labels and instead add one left-side row title
    per feature (outside the axes) to avoid stacked y-label overlap.
    """
    if len(features) == 0:
        raise ValueError("No features provided for binary panel plot.")

    nrows = len(features)
    ncols = len(ks)

    fig_w = 4.2 * ncols
    fig_h = max(6.0, 0.75 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=150, squeeze=False)

    for cidx, k in enumerate(ks):
        labels = label_map[k]
        clusts = _cluster_ids(labels)

        for ridx, feat in enumerate(features):
            ax = axes[ridx, cidx]
            if feat not in feature_names:
                ax.axis("off")
                continue

            j = feature_names.index(feat)
            props = []
            for c in clusts:
                vals = X[labels == c, j]
                vals = vals[np.isfinite(vals)]
                props.append(float(np.mean(vals)) if vals.size else np.nan)

            ax.bar([str(c) for c in clusts], props)
            ax.set_ylim(0, 1)

            if ridx == 0:
                ax.set_title(f"k{k}", fontsize=10)

            # UPDATED: remove per-axis y-labels (feature names) to avoid overlap
            ax.set_ylabel("")

            if ridx != nrows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Cluster", fontsize=8)

            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Cluster-wise distributions (binary): top features across k", fontsize=12)

    # UPDATED: create room for left-side row titles, then add them
    fig.subplots_adjust(left=0.14)
    _add_left_row_titles(fig, axes, features, fontsize=8)

    plt.tight_layout(rect=[0.14, 0, 1, 0.98])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def load_labels(k: int) -> np.ndarray:
    path = PAM_DIR / f"pam_labels_k{k}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")
    labels = np.load(path)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels for k={k}, got shape {labels.shape}")
    return labels.astype(int)


def contingency_matrix(labels_from: np.ndarray, labels_to: np.ndarray) -> pd.DataFrame:
    """
    Counts: rows=clusters at k_from, cols=clusters at k_to.
    """
    return pd.crosstab(
        pd.Series(labels_from, name="from"),
        pd.Series(labels_to, name="to"),
        normalize=False,
        dropna=False,
    )


def row_normalized_percentages(ct: pd.DataFrame) -> pd.DataFrame:
    """
    Row-normalized percentages (0..1). Handles zero rows safely.
    """
    row_sums = ct.sum(axis=1).replace(0, np.nan)
    row_pct = ct.div(row_sums, axis=0).astype(float).fillna(0.0)
    return row_pct


def row_entropy(row_probs: np.ndarray) -> float:
    """
    Entropy (natural log) for a single probability vector.
    Zeros ignored. If all zeros -> entropy=0.
    """
    p = row_probs[row_probs > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def stability_metrics_for_transition(
    k_from: int,
    k_to: int,
    labels_from: np.ndarray,
    labels_to: np.ndarray,
    split_threshold: float = 0.80,
    weighted: bool = True,
) -> dict:
    ct = contingency_matrix(labels_from, labels_to)
    row_pct = row_normalized_percentages(ct)

    # per-row: max overlap, entropy, effective #targets (perplexity)
    max_row = row_pct.max(axis=1).to_numpy(dtype=float)

    ent = np.array([row_entropy(row_pct.loc[i].to_numpy(dtype=float)) for i in row_pct.index], dtype=float)
    eff_targets = np.exp(ent)

    # weights are cluster sizes (row totals)
    weights = ct.sum(axis=1).to_numpy(dtype=float)
    weights_sum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0

    if weighted:
        avg_max = float((max_row * weights).sum() / weights_sum)
    else:
        avg_max = float(max_row.mean()) if max_row.size else 0.0

    med_max = float(np.median(max_row)) if max_row.size else 0.0
    med_entropy = float(np.median(ent)) if ent.size else 0.0
    med_eff_targets = float(np.median(eff_targets)) if eff_targets.size else 1.0

    pct_split = float((max_row < split_threshold).mean() * 100.0) if max_row.size else 0.0

    return {
        "transition": f"{k_from}->{k_to}",
        "avg_max_row_overlap": avg_max,                 # 0..1
        "median_max_row_overlap": med_max,              # 0..1
        "median_row_entropy": med_entropy,              # >=0
        "pct_clusters_split": pct_split,                # 0..100
        "effective_num_targets_median": med_eff_targets # >=1
    }


def make_sankey_adjacent(
    k_from: int,
    k_to: int,
    labels_from: np.ndarray,
    labels_to: np.ndarray,
    out_html: Path,
) -> None:
    ct = contingency_matrix(labels_from, labels_to)

    left_nodes = [f"k{k_from}_c{int(i)}" for i in ct.index.tolist()]
    right_nodes = [f"k{k_to}_c{int(j)}" for j in ct.columns.tolist()]
    labels = left_nodes + right_nodes

    idx_left = {name: i for i, name in enumerate(left_nodes)}
    idx_right = {name: i + len(left_nodes) for i, name in enumerate(right_nodes)}

    sources = []
    targets = []
    values = []

    for i in ct.index:
        for j in ct.columns:
            v = int(ct.loc[i, j])
            if v <= 0:
                continue
            sources.append(idx_left[f"k{k_from}_c{int(i)}"])
            targets.append(idx_right[f"k{k_to}_c{int(j)}"])
            values.append(v)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=15, thickness=15),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title_text=f"Sankey: k{k_from} → k{k_to} (cluster transitions)", font_size=12)
    fig.write_html(str(out_html))


def make_sankey_global(
    ks: list[int],
    label_map: dict[int, np.ndarray],
    out_html: Path,
) -> None:
    """
    Global Sankey with layers: k4 -> k5 -> k6 -> k7 (adjacent links only).
    Nodes labeled "k4_c0", etc.
    """
    # Build node labels in layer order
    node_labels: list[str] = []
    node_index: dict[str, int] = {}

    def add_node(name: str) -> int:
        if name in node_index:
            return node_index[name]
        node_index[name] = len(node_labels)
        node_labels.append(name)
        return node_index[name]

    sources = []
    targets = []
    values = []

    for a, b in zip(ks[:-1], ks[1:]):
        ct = contingency_matrix(label_map[a], label_map[b])
        for i in ct.index:
            for j in ct.columns:
                v = int(ct.loc[i, j])
                if v <= 0:
                    continue
                s = add_node(f"k{a}_c{int(i)}")
                t = add_node(f"k{b}_c{int(j)}")
                sources.append(s)
                targets.append(t)
                values.append(v)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=node_labels, pad=15, thickness=14),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(
        title_text=f"Global Sankey: {' → '.join([f'k{k}' for k in ks])} (adjacent transitions)",
        font_size=11,
    )
    fig.write_html(str(out_html))


def main() -> None:
    # NEW: silence the Matplotlib “labels -> tick_labels” deprecation spam
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

    # -------------------------
    # Load labels
    # -------------------------
    label_map = {k: load_labels(k) for k in KS}
    n = len(next(iter(label_map.values())))
    for k, lab in label_map.items():
        if len(lab) != n:
            raise ValueError(f"Label length mismatch: k={k} has {len(lab)} labels, expected {n}")

    print("\n=== POST-CLUSTER STABILITY (k=4..7) ===")
    print(f"n respondents: {n}")

    # -------------------------
    # ARI table (console + optional save)
    # -------------------------
    ari_rows = []
    for i in range(len(KS)):
        for j in range(i + 1, len(KS)):
            k1, k2 = KS[i], KS[j]
            ari = float(adjusted_rand_score(label_map[k1], label_map[k2]))
            ari_rows.append({"k1": k1, "k2": k2, "ARI": ari})

    ari_df = pd.DataFrame(ari_rows).sort_values(["k1", "k2"]).reset_index(drop=True)
    print("\n--- ARI between k solutions (agreement; higher = more stable) ---")
    print(ari_df.to_string(index=False))

    # Optional: keep a single CSV artifact (not many files)
    ari_df.to_csv(OUT_DIR / "robustness_ari.csv", index=False)

    # -------------------------
    # Stability summary metrics (adjacent only)
    # -------------------------
    stability_rows = []
    for k_from, k_to in zip(KS[:-1], KS[1:]):
        metrics = stability_metrics_for_transition(
            k_from=k_from,
            k_to=k_to,
            labels_from=label_map[k_from],
            labels_to=label_map[k_to],
            split_threshold=SPLIT_THRESHOLD,
            weighted=WEIGHTED,
        )
        stability_rows.append(metrics)

    stability_df = pd.DataFrame(stability_rows)
    # nicer formatting in console
    print("\n--- Stability summary metrics (adjacent transitions) ---")
    show = stability_df.copy()
    for c in ["avg_max_row_overlap", "median_max_row_overlap"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["median_row_entropy"] = show["median_row_entropy"].map(lambda x: f"{x:.3f}")
    show["effective_num_targets_median"] = show["effective_num_targets_median"].map(lambda x: f"{x:.3f}")
    show["pct_clusters_split"] = show["pct_clusters_split"].map(lambda x: f"{x:.1f}%")
    print(show.to_string(index=False))

    stability_df.to_csv(OUT_DIR / "stability_summary_adjacent.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'stability_summary_adjacent.csv'}")
    print(f"Saved: {OUT_DIR / 'robustness_ari.csv'}")

    # -------------------------
    # Sankey diagrams (adjacent + global)
    # -------------------------
    for k_from, k_to in zip(KS[:-1], KS[1:]):
        out_html = OUT_DIR / f"sankey_k{k_from}_to_k{k_to}.html"
        make_sankey_adjacent(k_from, k_to, label_map[k_from], label_map[k_to], out_html)
        print(f"Saved: {out_html}")

    global_html = OUT_DIR / "sankey_global_k4_k5_k6_k7.html"
    make_sankey_global(KS, label_map, global_html)
    print(f"Saved: {global_html}")

    # -------------------------
    # Feature importance heatmaps by type (k=5,6,7)
    # -------------------------
    if not GOWER_PATH.exists():
        raise FileNotFoundError(f"Missing Gower distance matrix: {GOWER_PATH}")

    X, feature_names = load_encoded_drivers()
    D = np.load(GOWER_PATH)

    # --- DIAGNOSTIC: inspect problematic ordinal feature ---
    feat = "bad_conseq_mh_prev_boss_ord"
    if feat in feature_names:
        j = feature_names.index(feat)
        col = X[:, j]
        vals, counts = np.unique(col, return_counts=True)
        print(f"\nDiagnostic for {feat}:")
        for v, c in zip(vals, counts):
            print(f"  value={v}, count={c}")
    else:
        print(f"\nFeature {feat} not found in encoded data.")
    # checking ends

    if D.shape[0] != X.shape[0]:
        raise ValueError(
            f"Row mismatch: Gower is {D.shape[0]}x{D.shape[1]}, but encoded X has {X.shape[0]} rows"
        )

    labels_by_k = {k: label_map[k] for k in FEATURE_IMPORTANCE_KS}

    binary_mask, ordinal_mask = split_feature_types(X, feature_names)

    # Ordinal/non-binary heatmap matrix (IQR-normalized)
    fi_ord = compute_feature_importance_matrix_ordinal(
        X=X,
        feature_names=feature_names,
        D=D,
        label_map=labels_by_k,
        ks=FEATURE_IMPORTANCE_KS,
        ordinal_mask=ordinal_mask,
    )
    fi_ord = fi_ord.sort_values("k5", ascending=False) if not fi_ord.empty and "k5" in fi_ord.columns else fi_ord
    fi_ord_csv = OUT_DIR / "feature_importance_ordinal_k5_k6_k7.csv"
    fi_ord.to_csv(fi_ord_csv)
    print(f"Saved ordinal feature-importance matrix: {fi_ord_csv}")

    # Use a single consistent color scale across k (within ordinal heatmap)
    if not fi_ord.empty:
        ord_png = OUT_DIR / "feature_importance_heatmap_ordinal.png"
        plot_heatmap(
            mat=fi_ord,
            save_path=ord_png,
            title="Feature importance (ordinal/non-binary): medoid separation normalized by IQR",
            colorbar_label="Mean pairwise medoid |Δ| / IQR",
        )
        print(f"Saved ordinal feature-importance heatmap: {ord_png}")
    else:
        print("Ordinal/non-binary heatmap skipped: no ordinal/non-binary features detected.")

    # Binary heatmap matrix (no IQR normalization; includes rare one-hot features)
    fi_bin = compute_feature_importance_matrix_binary(
        X=X,
        feature_names=feature_names,
        D=D,
        label_map=labels_by_k,
        ks=FEATURE_IMPORTANCE_KS,
        binary_mask=binary_mask,
    )
    fi_bin = fi_bin.sort_values("k5", ascending=False) if not fi_bin.empty and "k5" in fi_bin.columns else fi_bin
    fi_bin_csv = OUT_DIR / "feature_importance_binary_k5_k6_k7.csv"
    fi_bin.to_csv(fi_bin_csv)
    print(f"Saved binary feature-importance matrix: {fi_bin_csv}")

    # Fixed scale [0,1] across k for binary heatmap
    if not fi_bin.empty:
        bin_png = OUT_DIR / "feature_importance_heatmap_binary.png"
        plot_heatmap(
            mat=fi_bin,
            save_path=bin_png,
            title="Feature importance (binary): medoid separation (includes rare features)",
            colorbar_label="Mean pairwise medoid |Δ| (binary; scale 0..1)",
            vmin=0.0,
            vmax=1.0,
        )
        print(f"Saved binary feature-importance heatmap: {bin_png}")
    else:
        print("Binary heatmap skipped: no binary features detected.")

    # -------------------------
    # NEW: Combined distribution panels (top N from k=5 ranking)
    # -------------------------
    if not fi_ord.empty:
        top_ord_feats = (
            fi_ord["k5"].dropna().sort_values(ascending=False).head(TOP_PANEL_N).index.tolist()
            if "k5" in fi_ord.columns else fi_ord.index.tolist()[:TOP_PANEL_N]
        )
        out_panel_ord = OUT_DIR / "panel_distributions_ordinal_top10_k5_k6_k7.png"
        plot_panel_ordinal_boxplots_by_k(
            X=X,
            feature_names=feature_names,
            label_map=labels_by_k,
            ks=FEATURE_IMPORTANCE_KS,
            features=top_ord_feats,
            save_path=out_panel_ord,
        )
        print(f"Saved combined ordinal distribution panel: {out_panel_ord}")
    else:
        print("Ordinal distribution panel skipped: no ordinal features detected.")

    if not fi_bin.empty:
        top_bin_feats = (
            fi_bin["k5"].dropna().sort_values(ascending=False).head(TOP_PANEL_N).index.tolist()
            if "k5" in fi_bin.columns else fi_bin.index.tolist()[:TOP_PANEL_N]
        )
        out_panel_bin = OUT_DIR / "panel_distributions_binary_top10_k5_k6_k7.png"
        plot_panel_binary_bars_by_k(
            X=X,
            feature_names=feature_names,
            label_map=labels_by_k,
            ks=FEATURE_IMPORTANCE_KS,
            features=top_bin_feats,
            save_path=out_panel_bin,
        )
        print(f"Saved combined binary distribution panel: {out_panel_bin}")
    else:
        print("Binary distribution panel skipped: no binary features detected.")

    # -------------------------
    # NEW (additional): Combined distribution panels (top N from k=6 ranking)
    # -------------------------
    if not fi_ord.empty:
        top_ord_feats_k6 = (
            fi_ord["k6"].dropna().sort_values(ascending=False).head(TOP_PANEL_N).index.tolist()
            if "k6" in fi_ord.columns else fi_ord.index.tolist()[:TOP_PANEL_N]
        )
        out_panel_ord_k6 = OUT_DIR / "panel_distributions_ordinal_top10_by_k6_k5_k6_k7.png"
        plot_panel_ordinal_boxplots_by_k(
            X=X,
            feature_names=feature_names,
            label_map=labels_by_k,
            ks=FEATURE_IMPORTANCE_KS,
            features=top_ord_feats_k6,
            save_path=out_panel_ord_k6,
        )
        print(f"Saved combined ordinal distribution panel (features chosen by k6): {out_panel_ord_k6}")
    else:
        print("Ordinal distribution panel (k6-selected) skipped: no ordinal features detected.")

    if not fi_bin.empty:
        top_bin_feats_k6 = (
            fi_bin["k6"].dropna().sort_values(ascending=False).head(TOP_PANEL_N).index.tolist()
            if "k6" in fi_bin.columns else fi_bin.index.tolist()[:TOP_PANEL_N]
        )
        out_panel_bin_k6 = OUT_DIR / "panel_distributions_binary_top10_by_k6_k5_k6_k7.png"
        plot_panel_binary_bars_by_k(
            X=X,
            feature_names=feature_names,
            label_map=labels_by_k,
            ks=FEATURE_IMPORTANCE_KS,
            features=top_bin_feats_k6,
            save_path=out_panel_bin_k6,
        )
        print(f"Saved combined binary distribution panel (features chosen by k6): {out_panel_bin_k6}")
    else:
        print("Binary distribution panel (k6-selected) skipped: no binary features detected.")



if __name__ == "__main__":
    main()
