"""
cluster5_friend_family_mhd_comfort_percentages.py

Prints (within cluster 5) the % who answered:
  - "Somewhat open"
  - "Very open"
to friend_family_mhd_comfort (or friends_family_mhd_comfort), using FULL cluster size
(including NaNs) as the denominator.

ALSO prints (within cluster 2) the % who answered:
  - prev_benefits: "Yes, they all did"
  - prev_mh_options_known: "I was aware of some"
  - prev_mh_options_known: "Yes, I was aware of all of them"
  - prev_resources: "Yes, they all did"
  - prev_resources: "Some did"
  - bad_conseq_mh_prev_boss: "Yes, all of them"
using FULL cluster size (including NaNs) as the denominator.

ALSO prints (within cluster 4) the % who answered:
  - prev_resources: "None did"
using FULL cluster size (including NaNs) as the denominator.

ALSO creates cluster-vs-cluster medoid-difference heatmaps for k=6.

Expected inputs:
  - data/out/drivers_encoded.csv
  - data/out/pam/pam_labels_k6.npy
"""

from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from encoding import ORDINAL_MAPS, MIXED_SPECS, MIXED_SPECS_V2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODED_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "out" / "pam" / "pam_labels_k6.npy"

OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam_post" / "pairwise_feature_separation_k6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_ID = 5


FEATURE_CANDIDATES = [
    "friend_family_mhd_comfort_ord",
    "friends_family_mhd_comfort_ord",
    "friend_family_mhd_comfort",
    "friends_family_mhd_comfort",
]


def load_encoded_df() -> pd.DataFrame:
    return pd.read_csv(ENCODED_PATH)


def load_labels() -> np.ndarray:
    return np.load(LABELS_PATH).astype(int)


def _find_feature_column(df: pd.DataFrame) -> str:
    for c in FEATURE_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError("friend_family_mhd_comfort column not found")


def _build_forward_map() -> dict[str, dict[str, float]]:
    fwd = {}
    for feat, mapping in ORDINAL_MAPS.items():
        fwd[f"{feat}_ord"] = {str(k): float(v) for k, v in mapping.items()}
    for feat, spec in MIXED_SPECS.items():
        fwd[f"{feat}_ord"] = {str(k): float(v) for k, v in spec["ord_mapping"].items()}
    for feat, spec in MIXED_SPECS_V2.items():
        fwd[f"{feat}_ord"] = {str(k): float(v) for k, v in spec["ord_mapping"].items()}
    return fwd


FWD_MAP = _build_forward_map()


def _get_code_for_label(feature_col: str, label: str) -> float | None:
    mapping = FWD_MAP.get(feature_col)
    if mapping is None:
        return None
    target = label.lower()
    for k, v in mapping.items():
        if k.lower() == target:
            return v
    return None


def _pct_and_count_over_full(values, value):
    total = len(values)
    count = np.sum(values == value)
    return count / total, count, total


def fmt_pct(x):
    return f"{x*100:.1f}%"


def _find_any_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError


def _print_cluster_label_pct(df, labels, cluster_id, feature_candidates, label_text):
    feature = _find_any_column(df, feature_candidates)
    values = df[feature].values[labels == cluster_id]
    lookup = feature if feature.endswith("_ord") else f"{feature}_ord"
    code = _get_code_for_label(lookup, label_text)
    if code is None:
        print("NA")
        return
    pct, cnt, total = _pct_and_count_over_full(values, code)
    print(f'{feature}: "{label_text}" -> {fmt_pct(pct)} ({cnt}/{total})')


# ============================================================
# NEW SECTION: PAIRWISE MEDOID DIFFERENCE HEATMAPS
# ============================================================

def compute_cluster_medoids(df, labels):
    medoids = {}
    for k in sorted(np.unique(labels)):
        medoids[k] = df[labels == k].median()
    return pd.DataFrame(medoids).T


def compute_pairwise_differences(medoids_df):
    pairs = list(itertools.combinations(medoids_df.index, 2))
    out = pd.DataFrame(index=medoids_df.columns)
    for a, b in pairs:
        out[f"C{a}-C{b}"] = (medoids_df.loc[a] - medoids_df.loc[b]).abs()
    return out


def plot_heatmap(data, title, path, vmax=None):

    if data.shape[0] > 30:
        data = data.loc[data.max(axis=1).sort_values(ascending=False).head(30).index]

    plt.figure(figsize=(10, max(6, len(data) * 0.35)))
    sns.heatmap(data, cmap="viridis", vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_pairwise_heatmaps(df, labels):

    medoids = compute_cluster_medoids(df, labels)

    ordinal_cols = [c for c in df.columns if c.endswith("_ord")]
    binary_cols = [c for c in df.columns if c not in ordinal_cols]

    # --- raw pairwise diffs (unnormalized) ---
    ordinal_raw = compute_pairwise_differences(medoids[ordinal_cols])
    binary = compute_pairwise_differences(medoids[binary_cols])

    # --- IQR-normalized ordinal ---
    iqr = df[ordinal_cols].quantile(0.75) - df[ordinal_cols].quantile(0.25)
    ordinal_iqr = ordinal_raw.divide(iqr.replace(0, np.nan), axis=0)

    # --- save tables ---
    ordinal_raw.to_csv(OUT_DIR / "pairwise_medoid_diff_ordinal_raw_k6.csv")
    ordinal_iqr.to_csv(OUT_DIR / "pairwise_medoid_diff_ordinal_iqr_k6.csv")
    binary.to_csv(OUT_DIR / "pairwise_medoid_diff_binary_k6.csv")

    # --- plots ---
    plot_heatmap(
        ordinal_iqr,
        "Ordinal feature separation (IQR-normalized) — k=6",
        OUT_DIR / "pairwise_medoid_diff_ordinal_iqr_k6.png",
    )

    plot_heatmap(
        ordinal_raw,
        "Ordinal feature separation (unnormalized) — k=6",
        OUT_DIR / "pairwise_medoid_diff_ordinal_raw_k6.png",
    )

    plot_heatmap(
        binary,
        "Binary feature separation — k=6",
        OUT_DIR / "pairwise_medoid_diff_binary_k6.png",
        vmax=1,
    )


# ============================================================


def main():

    df = load_encoded_df()
    labels = load_labels()

    feature = _find_feature_column(df)

    values = df[feature].values[labels == CLUSTER_ID]

    code_some = _get_code_for_label(feature, "Somewhat open") or 4
    code_very = _get_code_for_label(feature, "Very open") or 5

    pct_some, cnt_some, total = _pct_and_count_over_full(values, code_some)
    pct_very, cnt_very, _ = _pct_and_count_over_full(values, code_very)

    print(f"\nCluster {CLUSTER_ID}")
    print("Somewhat open:", fmt_pct(pct_some))
    print("Very open:", fmt_pct(pct_very))


    print("\nCreating cluster-pair heatmaps...")
    run_pairwise_heatmaps(df, labels)
    print("Done.")


if __name__ == "__main__":
    main()
