# pam_cluster_profiles_k6_console.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Import ordinal maps directly from encoding.py
from encoding import ORDINAL_MAPS, MIXED_SPECS, MIXED_SPECS_V2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAM_DIR = PROJECT_ROOT / "data" / "out" / "pam"
ENCODED_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"
DRIVERS_RAW_PATH = PROJECT_ROOT / "data" / "out" / "survey_drivers.csv"  # before encoding, after preprocessing

K = 6
LABELS_PATH = PAM_DIR / f"pam_labels_k{K}.npy"

TOP_N = 10
THRESHOLD = 0.45  # requested cutoff

# NEW: output folder for raw (pre-encoding) driver plots
RAW_PLOTS_DIR = PROJECT_ROOT / "data" / "out" / "pam_post" / "raw_driver_distributions_k6"
RAW_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_encoded_drivers() -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(ENCODED_PATH)
    if "respondent_id" in df.columns:
        df = df.drop(columns=["respondent_id"])
    X = df.to_numpy(dtype=float)
    return X, df.columns.tolist()


def load_labels() -> np.ndarray:
    lab = np.load(LABELS_PATH)
    return lab.astype(int)


def load_preprocessed_drivers_raw() -> pd.DataFrame:
    """
    Loads drivers BEFORE encoding but AFTER preprocessing (skip-logic / cleaning).
    Expected file: data/out/survey_drivers.csv
    """
    if not DRIVERS_RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw drivers file: {DRIVERS_RAW_PATH}")
    return pd.read_csv(DRIVERS_RAW_PATH)


def align_labels_to_raw(df_raw: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """
    Ensures labels align with df_raw row order.

    If respondent_id is available in both df_raw and drivers_encoded.csv:
      - create (respondent_id -> label) from encoded row order
      - map onto df_raw order

    Otherwise:
      - assume df_raw order == encoded order and lengths match.
    """
    if "respondent_id" in df_raw.columns:
        df_enc = pd.read_csv(ENCODED_PATH)
        if "respondent_id" in df_enc.columns and len(df_enc) == len(labels):
            id_to_label = pd.Series(labels, index=df_enc["respondent_id"]).to_dict()
            mapped = df_raw["respondent_id"].map(id_to_label).to_numpy()
            if np.any(pd.isna(mapped)):
                missing = int(pd.isna(mapped).sum())
                raise ValueError(
                    f"Could not align {missing} rows via respondent_id. "
                    f"Check that survey_drivers.csv and drivers_encoded.csv contain the same respondent_id set."
                )
            return mapped.astype(int)

    # fallback: positional alignment
    if len(df_raw) != len(labels):
        raise ValueError(
            f"Row count mismatch: raw drivers n={len(df_raw)} vs labels n={len(labels)}. "
            f"Cannot align without respondent_id."
        )
    return labels.astype(int)


# ------------------------------------------------------------
# Build reverse ordinal mapping
# ------------------------------------------------------------

def build_reverse_ordinal_map() -> dict[str, dict[float, str]]:
    reverse_map = {}

    for feat, mapping in ORDINAL_MAPS.items():
        reverse_map[f"{feat}_ord"] = {float(v): k for k, v in mapping.items()}

    for feat, spec in MIXED_SPECS.items():
        mapping = spec["ord_mapping"]
        reverse_map[f"{feat}_ord"] = {float(v): k for k, v in mapping.items()}

    for feat, spec in MIXED_SPECS_V2.items():
        mapping = spec["ord_mapping"]
        reverse_map[f"{feat}_ord"] = {float(v): k for k, v in mapping.items()}

    return reverse_map


REVERSE_ORDINAL_MAP = build_reverse_ordinal_map()


def ordinal_label(feat: str, value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    mapping = REVERSE_ORDINAL_MAP.get(feat, {})
    return mapping.get(float(value), str(value))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def fmt_pct_or_na(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x*100:.1f}%"


def safe_median_and_pct_at_median(values: np.ndarray) -> tuple[float, float | None]:
    """
    Returns:
      (median_value, pct_at_median_valid)

    - filters to finite values for the median
    - if no finite values, returns (nan, None)
    - pct_at_median_valid is fraction of FINITE values equal to the median
      (we keep this internally, but tables will show % over FULL cluster size)
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan"), None
    med = float(np.median(v))
    pct_valid = float(np.mean(v == med)) if np.isfinite(med) else None
    return med, pct_valid


def pct_at_value_over_full(values: np.ndarray, value: float) -> float | None:
    """
    Fraction over FULL group size (including NaNs) that equals `value`.
    If value is not finite, returns None.
    """
    if not np.isfinite(value):
        return None
    if values.size == 0:
        return None
    return float(np.mean(values == value))


# ------------------------------------------------------------
# Feature type split
# ------------------------------------------------------------

def split_feature_types(X: np.ndarray, feature_names: list[str]):
    binary = []
    ordinal = []

    for j, feat in enumerate(feature_names):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            ordinal.append(feat)
            continue
        uniq = np.unique(col)
        if np.all(np.isin(uniq, [0.0, 1.0])):
            binary.append(feat)
        else:
            ordinal.append(feat)

    return binary, ordinal


# ------------------------------------------------------------
# Cluster vs Rest Tables (UNCHANGED for binary; ordinal % now over FULL group size)
# ------------------------------------------------------------

def build_binary_table(X, features, labels, cluster_id):
    rows = []
    mask_c = labels == cluster_id
    mask_r = ~mask_c

    for feat in features:
        j = feature_names.index(feat)
        col = X[:, j]

        c_vals = col[mask_c]
        r_vals = col[mask_r]

        c_vals = c_vals[np.isfinite(c_vals)]
        r_vals = r_vals[np.isfinite(r_vals)]

        if c_vals.size == 0 or r_vals.size == 0:
            continue

        p_c = float(np.mean(c_vals))
        p_r = float(np.mean(r_vals))

        rows.append({
            "feature": feat,
            "cluster_%": f"{p_c*100:.1f}%",
            "rest_%": f"{p_r*100:.1f}%",
            "_rank": abs(p_c - p_r)
        })

    df = pd.DataFrame(rows)
    return df.sort_values("_rank", ascending=False).head(TOP_N).drop(columns="_rank")


def build_ordinal_table(X, features, labels, cluster_id):
    rows = []
    mask_c = labels == cluster_id
    mask_r = ~mask_c

    for feat in features:
        j = feature_names.index(feat)
        col = X[:, j]

        c_raw = col[mask_c]  # includes NaNs
        r_raw = col[mask_r]  # includes NaNs

        med_c, _pct_c_valid = safe_median_and_pct_at_median(c_raw)
        med_r, _pct_r_valid = safe_median_and_pct_at_median(r_raw)

        # % at CLUSTER median, over FULL group size (includes NaNs)
        pct_c_full = pct_at_value_over_full(c_raw, med_c)

        # % in REST at CLUSTER median, over FULL rest size
        pct_r_full_at_cluster_med = pct_at_value_over_full(r_raw, med_c)

        rows.append({
            "feature": feat,
            "cluster_median": ordinal_label(feat, med_c),
            "rest_median": ordinal_label(feat, med_r),
            "cluster_%_at_cluster_median": fmt_pct_or_na(pct_c_full),
            "rest_%_at_cluster_median": fmt_pct_or_na(pct_r_full_at_cluster_med),
            "_rank": abs(med_c - med_r) if (np.isfinite(med_c) and np.isfinite(med_r)) else -1.0
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("_rank", ascending=False)
    df = df[df["_rank"] >= 0] if (df["_rank"] >= 0).any() else df
    return df.head(TOP_N).drop(columns="_rank")


# ------------------------------------------------------------
# Pairwise Cluster Comparison Tables (ordinal % now over FULL group size)
# ------------------------------------------------------------

def build_binary_pairwise(X, features, labels, c1, c2):
    rows = []
    mask1 = labels == c1
    mask2 = labels == c2

    for feat in features:
        j = feature_names.index(feat)
        col = X[:, j]

        v1 = col[mask1]
        v2 = col[mask2]

        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]

        if v1.size == 0 or v2.size == 0:
            continue

        p1 = float(np.mean(v1))
        p2 = float(np.mean(v2))

        rows.append({
            "feature": feat,
            f"cluster_{c1}": f"{p1*100:.1f}%",
            f"cluster_{c2}": f"{p2*100:.1f}%",
            "_rank": abs(p1 - p2)
        })

    df = pd.DataFrame(rows)
    return df.sort_values("_rank", ascending=False).head(TOP_N).drop(columns="_rank")


def build_ordinal_pairwise(X, features, labels, c1, c2):
    rows = []
    mask1 = labels == c1
    mask2 = labels == c2

    for feat in features:
        j = feature_names.index(feat)
        col = X[:, j]

        v1_raw = col[mask1]  # includes NaNs
        v2_raw = col[mask2]  # includes NaNs

        med1, _pct1_valid = safe_median_and_pct_at_median(v1_raw)
        med2, _pct2_valid = safe_median_and_pct_at_median(v2_raw)

        # % at each cluster's OWN median, over FULL cluster size
        pct1_full = pct_at_value_over_full(v1_raw, med1)
        pct2_full = pct_at_value_over_full(v2_raw, med2)

        rows.append({
            "feature": feat,
            f"cluster_{c1}_median": ordinal_label(feat, med1),
            f"cluster_{c2}_median": ordinal_label(feat, med2),
            f"cluster_{c1}_%_at_own_median": fmt_pct_or_na(pct1_full),
            f"cluster_{c2}_%_at_own_median": fmt_pct_or_na(pct2_full),
            "_rank": abs(med1 - med2) if (np.isfinite(med1) and np.isfinite(med2)) else -1.0
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("_rank", ascending=False)
    df = df[df["_rank"] >= 0] if (df["_rank"] >= 0).any() else df
    return df.head(TOP_N).drop(columns="_rank")


# ------------------------------------------------------------
# NEW: High-prevalence feature lists (>= THRESHOLD)
# ------------------------------------------------------------

def list_binary_features_ge_threshold(X, features, labels, cluster_id, threshold: float):
    """
    Lists binary features with proportion(==1) >= threshold within the cluster.
    Returns a DataFrame: feature, value (always "1"), cluster_%.
    """
    rows = []
    mask_c = labels == cluster_id

    for feat in features:
        j = feature_names.index(feat)
        col = X[:, j]
        c_vals = col[mask_c]
        c_vals = c_vals[np.isfinite(c_vals)]
        if c_vals.size == 0:
            continue
        p = float(np.mean(c_vals))
        if p >= threshold:
            rows.append({
                "feature": feat,
                "value": "1",
                "cluster_%": f"{p*100:.1f}%"
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_share"] = df["cluster_%"].str.replace("%", "", regex=False).astype(float)
    df = df.sort_values("_share", ascending=False).drop(columns="_share")
    return df


def list_ordinal_values_ge_threshold(X, features, labels, cluster_id, threshold: float):
    """
    For each ordinal feature, finds any single value whose share (over FULL cluster size)
    is >= threshold. If multiple values qualify, keeps the highest-share value.
    Returns DataFrame: feature, value_label, cluster_%.
    """
    rows = []
    mask_c = labels == cluster_id

    for feat in features:
        j = feature_names.index(feat)
        col = X[:, j]
        c_raw = col[mask_c]  # includes NaNs

        if c_raw.size == 0:
            continue

        finite = c_raw[np.isfinite(c_raw)]
        if finite.size == 0:
            continue

        uniq = np.unique(finite)
        best_val = None
        best_share = -1.0

        for v in uniq:
            share = float(np.mean(c_raw == v))  # denominator includes NaNs
            if share > best_share:
                best_share = share
                best_val = float(v)

        if best_val is None:
            continue

        if best_share >= threshold:
            rows.append({
                "feature": feat,
                "value": ordinal_label(feat, best_val),
                "cluster_%": f"{best_share*100:.1f}%"
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_share"] = df["cluster_%"].str.replace("%", "", regex=False).astype(float)
    df = df.sort_values("_share", ascending=False).drop(columns="_share")
    return df


# ------------------------------------------------------------
# NEW: Raw (pre-encoding) stacked bar plots by cluster
# ------------------------------------------------------------

def plot_raw_feature_stacked_by_cluster(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    feature: str,
    out_path: Path,
) -> None:
    """
    For ONE raw driver feature:
      - x axis: clusters 0..5
      - stacked segments: answer categories (including NA for missing)
      - each bar sums to 1.0 (proportions within cluster)
    """
    import matplotlib.pyplot as plt

    series = df_raw[feature].copy()

    # normalize missing display
    series = series.astype(object)
    series[pd.isna(series)] = "NA"
    series = series.astype(str)

    clusters = sorted(np.unique(labels_aligned).tolist())

    # union of categories across all data (stable order by overall frequency)
    overall_counts = series.value_counts(dropna=False)
    categories = overall_counts.index.tolist()

    # build proportion table: rows=clusters, cols=categories
    mat = []
    for c in clusters:
        s_c = series[labels_aligned == c]
        vc = s_c.value_counts(dropna=False)
        total = len(s_c)
        props = [(float(vc.get(cat, 0)) / total) if total > 0 else 0.0 for cat in categories]
        mat.append(props)

    mat = np.array(mat, dtype=float)  # shape (n_clusters, n_categories)

    # plot
    fig_w = 10
    fig_h = 4.5
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    # NOTE: we do not hardcode specific colors; matplotlib will cycle colors by category.
    for j, cat in enumerate(categories):
        plt.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    plt.xticks(x, [str(c) for c in clusters])
    plt.ylim(0, 1.0)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion within cluster")
    plt.title(f"Raw feature distribution by cluster (k=6): {feature}")

    # keep legend readable (can get long)
    plt.legend(
        title="Answer",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _normalize_benefits_value(x: object) -> str:
    """
    Normalizes raw 'benefits' answers into 4 buckets:
      Y  = Yes
      N  = No
      DK = I don't know
      NE = Not eligible for coverage / N/A (and missing)
    """
    if pd.isna(x):
        return "NE"
    s = str(x).strip()

    s_low = s.lower()
    if s_low in {"yes", "y"}:
        return "Y"
    if s_low in {"no", "n"}:
        return "N"
    if "don't know" in s_low or "dont know" in s_low or s_low in {"dk", "idk"}:
        return "DK"
    if "not eligible" in s_low or "n/a" in s_low or s_low in {"na", "n.a.", "n\\a"}:
        return "NE"

    # fall back to a compact, but explicit, bucket to avoid exploding categories
    return "NE"


def _normalize_resources_value(x: object) -> str:
    """
    Normalizes raw 'resources' answers into 3 buckets:
      Y  = Yes
      N  = No
      DK = I don't know (and missing)
    """
    if pd.isna(x):
        return "DK"
    s = str(x).strip()

    s_low = s.lower()
    if s_low in {"yes", "y"}:
        return "Y"
    if s_low in {"no", "n"}:
        return "N"
    if "don't know" in s_low or "dont know" in s_low or s_low in {"dk", "idk"}:
        return "DK"

    # fall back to DK to keep the combined legend limited to 12 combinations
    return "DK"


def plot_raw_benefits_and_resources_stacked_by_cluster(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    benefits_feature: str,
    resources_feature: str,
    out_path: Path,
) -> None:
    """
    Combined raw feature plot: 'benefits and resources'

    - combines two raw driver features into 12 joint answer categories:
        Benefits:  Y / N / DK / NE
        Resources: Y / N / DK
      -> 4 * 3 = 12 combinations

    - x axis: clusters 0..5
    - stacked segments: joint answer categories
    - each bar sums to 1.0 (proportions within cluster)

    Legend uses compact labels of the form:
      B:Y R:N   (B=benefits, R=resources)
    """
    import matplotlib.pyplot as plt

    b = df_raw[benefits_feature].map(_normalize_benefits_value)
    r = df_raw[resources_feature].map(_normalize_resources_value)

    joint = ("B:" + b + " R:" + r).astype(str)

    clusters = sorted(np.unique(labels_aligned).tolist())

    # stable category order: by overall frequency, then lexicographic to keep ties deterministic
    overall_counts = joint.value_counts(dropna=False)
    categories = overall_counts.sort_values(ascending=False).index.tolist()

    mat = []
    for c in clusters:
        s_c = joint[labels_aligned == c]
        vc = s_c.value_counts(dropna=False)
        total = len(s_c)
        props = [(float(vc.get(cat, 0)) / total) if total > 0 else 0.0 for cat in categories]
        mat.append(props)

    mat = np.array(mat, dtype=float)

    fig_w = 10
    fig_h = 4.5
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    for j, cat in enumerate(categories):
        plt.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    plt.xticks(x, [str(c) for c in clusters])
    plt.ylim(0, 1.0)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion within cluster")
    plt.title("Raw feature distribution by cluster (k=6): benefits and resources")

    plt.legend(
        title="Answer (B=benefits, R=resources)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _normalize_interview_answer(x: object) -> str:
    """
    Normalizes raw interview answers (mh_interview / ph_interview) into 3 buckets:
      Y = Yes
      N = No
      M = Maybe (and missing / unexpected)
    """
    if pd.isna(x):
        return "M"
    s = str(x).strip()
    s_low = s.lower()

    if s_low in {"yes", "y"}:
        return "Y"
    if s_low in {"no", "n"}:
        return "N"
    if s_low in {"maybe", "m"}:
        return "M"

    return "M"


def plot_raw_mh_and_ph_interview_stacked_by_cluster(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    mh_feature: str,
    ph_feature: str,
    out_path: Path,
) -> None:
    """
    Combined raw feature plot: 'mh and ph interview'

    - combines two raw driver features into 9 joint answer categories:
        MH interview: Y / N / M
        PH interview: Y / N / M
      -> 3 * 3 = 9 combinations

    Legend uses compact labels of the form:
      MH:Y PH:M
    """
    import matplotlib.pyplot as plt

    mh = df_raw[mh_feature].map(_normalize_interview_answer)
    ph = df_raw[ph_feature].map(_normalize_interview_answer)

    joint = ("MH:" + mh + " PH:" + ph).astype(str)

    clusters = sorted(np.unique(labels_aligned).tolist())

    overall_counts = joint.value_counts(dropna=False)
    categories = overall_counts.sort_values(ascending=False).index.tolist()

    mat = []
    for c in clusters:
        s_c = joint[labels_aligned == c]
        vc = s_c.value_counts(dropna=False)
        total = len(s_c)
        props = [(float(vc.get(cat, 0)) / total) if total > 0 else 0.0 for cat in categories]
        mat.append(props)

    mat = np.array(mat, dtype=float)

    fig_w = 10
    fig_h = 4.5
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    for j, cat in enumerate(categories):
        plt.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    plt.xticks(x, [str(c) for c in clusters])
    plt.ylim(0, 1.0)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion within cluster")
    plt.title("Raw feature distribution by cluster (k=6): mh and ph interview")

    plt.legend(
        title="Answer (MH=mh_interview, PH=ph_interview)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _normalize_current_anonymity_value(x: object) -> str:
    """
    Normalizes raw 'anonymity_protected' answers into 3 buckets:
      Y  = Yes
      N  = No
      DK = I don't know (and missing)
    """
    if pd.isna(x):
        return "DK"
    s = str(x).strip()
    s_low = s.lower()

    if s_low in {"yes", "y"}:
        return "Y"
    if s_low in {"no", "n"}:
        return "N"
    if "don't know" in s_low or "dont know" in s_low or s_low in {"dk", "idk"}:
        return "DK"

    return "DK"


def _normalize_prev_anonymity_value_merged(x: object) -> str:
    """
    Normalizes raw 'prev_anonymity_protected' answers, but merges:
      - 'Yes, always' and 'Sometimes' -> '≥S' (at least sometimes)

    Buckets returned:
      GE_S = '≥S'  (at least sometimes)
      N    = No
      NA   = Not applicable
      DK   = I don't know (and missing)

    This is only used for the combined "current and previous anonymity" feature.
    """
    if pd.isna(x):
        return "DK"
    s = str(x).strip()
    s_low = s.lower()

    if s_low in {"no", "n"}:
        return "N"
    if "not applicable" in s_low or s_low in {"na", "n/a", "n.a.", "n\\a"}:
        return "NA"
    if "don't know" in s_low or "dont know" in s_low or s_low in {"dk", "idk"}:
        return "DK"

    # merge the two requested categories
    if "yes, always" in s_low or s_low == "yes, always":
        return "≥S"
    if "sometimes" in s_low:
        return "≥S"

    return "DK"


def plot_raw_current_and_previous_anonymity_stacked_by_cluster(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    current_feature: str,
    prev_feature: str,
    out_path: Path,
) -> None:
    """
    Combined raw feature plot: 'current and previous anonymity'

    - current_feature (anonymity_protected): Y / N / DK
    - prev_feature (prev_anonymity_protected), with merge:
        'Yes, always' + 'Sometimes' -> '≥S' (at least sometimes)
      then buckets: ≥S / N / NA / DK

    -> 3 * 4 = 12 joint categories

    Legend uses compact labels of the form:
      C:Y P:≥S
    """
    import matplotlib.pyplot as plt

    cur = df_raw[current_feature].map(_normalize_current_anonymity_value)
    prev = df_raw[prev_feature].map(_normalize_prev_anonymity_value_merged)

    joint = ("C:" + cur + " P:" + prev).astype(str)

    clusters = sorted(np.unique(labels_aligned).tolist())

    overall_counts = joint.value_counts(dropna=False)
    categories = overall_counts.sort_values(ascending=False).index.tolist()

    mat = []
    for c in clusters:
        s_c = joint[labels_aligned == c]
        vc = s_c.value_counts(dropna=False)
        total = len(s_c)
        props = [(float(vc.get(cat, 0)) / total) if total > 0 else 0.0 for cat in categories]
        mat.append(props)

    mat = np.array(mat, dtype=float)

    fig_w = 10
    fig_h = 4.5
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    for j, cat in enumerate(categories):
        plt.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    plt.xticks(x, [str(c) for c in clusters])
    plt.ylim(0, 1.0)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion within cluster")
    plt.title("Raw feature distribution by cluster (k=6): current and previous anonymity")

    plt.legend(
        title="Answer (C=current, P=previous; ≥S=≥ sometimes; NA=not applicable)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _normalize_mhdcoworker_not_reveal(x: object) -> str:
    """
    Normalizes raw 'mhdcoworker_you_not_reveal' answers WITHOUT merging categories.
    We only map each distinct response to a short code (1:1 mapping).
    NOTE: NaN is treated as "NR" (no response) to keep the joint space at 5 reveal categories.

    Codes:
      Y   = Yes
      N   = No
      M   = Maybe
      NAp = Not applicable
      NR  = No response (and missing)
    """
    if pd.isna(x):
        return "NR"
    s = str(x).strip()
    s_low = s.lower()

    if s_low in {"yes", "y"}:
        return "Y"
    if s_low in {"no", "n"}:
        return "N"
    if s_low in {"maybe", "m"}:
        return "M"
    if "not applicable" in s_low:
        return "NAp"
    if "no response" in s_low:
        return "NR"

    return f"X:{s}"


def _normalize_observed_bad_conseq_raw(x: object) -> str:
    """
    Normalizes raw 'observed_mhdcoworker_bad_conseq' answers WITHOUT merging categories.
    This feature is typically 0/1.
    NOTE: NaN is treated as "0" to keep the joint space at 2 consequence categories.

    Codes:
      0 / 1 = as-is
      X:<...> = unexpected non-0/1 strings
    """
    if pd.isna(x):
        return "0"
    s = str(x).strip()
    if s in {"0", "1"}:
        return s
    s_low = s.lower()
    if s_low in {"false"}:
        return "0"
    if s_low in {"true"}:
        return "1"
    return f"X:{s}"


def _normalize_ever_observed_bad_response(x: object) -> str:
    """
    Normalizes raw 'ever_observed_mhd_bad_response' answers WITHOUT merging categories.
    We map each distinct response to a short code (1:1 mapping).
    NOTE: NaN is treated as "NR" (no response) to keep the joint space at 5 observed categories.

    Codes:
      N   = No
      M   = Maybe/Not sure
      YO  = Yes, I observed
      YE  = Yes, I experienced
      NR  = No response (and missing)
    """
    if pd.isna(x):
        return "NR"
    s = str(x).strip()
    s_low = s.lower()

    if s_low in {"no", "n"}:
        return "N"
    if "maybe" in s_low or "not sure" in s_low:
        return "M"
    if "yes" in s_low and "observed" in s_low:
        return "YO"
    if "yes" in s_low and "experienced" in s_low:
        return "YE"
    if "no response" in s_low:
        return "NR"

    return f"X:{s}"


def plot_raw_mhd_coworker_combo_stacked_by_cluster(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    reveal_feature: str,
    conseq_feature: str,
    ever_feature: str,
    out_path: Path,
) -> None:
    """
    Combined raw feature plot for:
      - mhdcoworker_you_not_reveal
      - observed_mhdcoworker_bad_conseq
      - ever_observed_mhd_bad_response

    The full expected joint space is forced to show (5 * 2 * 5 = 50 combos):
      Reveal:  NAp / N / Y / NR / M
      Conseq:  0 / 1
      Ever:    N / M / YO / YE / NR
    """
    import matplotlib.pyplot as plt

    r = df_raw[reveal_feature].map(_normalize_mhdcoworker_not_reveal)
    c = df_raw[conseq_feature].map(_normalize_observed_bad_conseq_raw)
    e = df_raw[ever_feature].map(_normalize_ever_observed_bad_response)

    joint = ("R:" + r + " C:" + c + " E:" + e).astype(str)

    clusters = sorted(np.unique(labels_aligned).tolist())

    R_vals = ["NAp", "N", "Y", "NR", "M"]
    C_vals = ["0", "1"]
    E_vals = ["N", "M", "YO", "YE", "NR"]
    categories = [f"R:{rv} C:{cv} E:{ev}" for rv in R_vals for cv in C_vals for ev in E_vals]

    mat = []
    for cl in clusters:
        s_cl = joint[labels_aligned == cl]
        vc = s_cl.value_counts(dropna=False)
        total = len(s_cl)
        props = [(float(vc.get(cat, 0)) / total) if total > 0 else 0.0 for cat in categories]
        mat.append(props)

    mat = np.array(mat, dtype=float)

    fig_w = 10
    fig_h = 4.5
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    for j, cat in enumerate(categories):
        plt.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    plt.xticks(x, [str(cl) for cl in clusters])
    plt.ylim(0, 1.0)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion within cluster")
    plt.title("Raw feature distribution by cluster (k=6): mhd coworker combo")

    plt.legend(
        title="Answer (R=reveal, C=conseq, E=ever; NAp=not applicable; NR=no response)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_raw_observe_vs_reveal_stacked_by_cluster(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    reveal_feature: str,
    observed_feature: str,
    out_path: Path,
) -> None:
    """
    Combined raw feature plot: 'observe_vs_reveal'

    - reveal_feature: mhdcoworker_you_not_reveal (5 categories)
    - observed_feature: ever_observed_mhd_bad_response (5 categories)
    -> 25 joint categories

    Joint space is forced to show (5 * 5 = 25 combos):
      Reveal:  NAp / N / Y / NR / M
      Observed: N / M / YO / YE / NR
    """
    import matplotlib.pyplot as plt

    r = df_raw[reveal_feature].map(_normalize_mhdcoworker_not_reveal)
    o = df_raw[observed_feature].map(_normalize_ever_observed_bad_response)

    joint = ("R:" + r + " O:" + o).astype(str)

    clusters = sorted(np.unique(labels_aligned).tolist())

    R_vals = ["NAp", "N", "Y", "NR", "M"]
    O_vals = ["N", "M", "YO", "YE", "NR"]
    categories = [f"R:{rv} O:{ov}" for rv in R_vals for ov in O_vals]

    mat = []
    for c in clusters:
        s_c = joint[labels_aligned == c]
        vc = s_c.value_counts(dropna=False)
        total = len(s_c)
        props = [(float(vc.get(cat, 0)) / total) if total > 0 else 0.0 for cat in categories]
        mat.append(props)

    mat = np.array(mat, dtype=float)

    fig_w = 10
    fig_h = 4.5
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    for j, cat in enumerate(categories):
        plt.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    plt.xticks(x, [str(c) for c in clusters])
    plt.ylim(0, 1.0)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion within cluster")
    plt.title("Raw feature distribution by cluster (k=6): observe_vs_reveal")

    plt.legend(
        title="Answer (R=reveal, O=observed; NAp=not applicable; NR=no response)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _normalize_boss_serious_value(x: object) -> str:
    """
    Normalizes raw 'mh_ph_boss_serious' answers into 3 buckets:
      Yes / No / I don't know
    Missing is treated as "I don't know".
    """
    if pd.isna(x):
        return "I don't know"
    s = str(x).strip()
    s_low = s.lower()

    if s_low in {"yes", "y"}:
        return "Yes"
    if s_low in {"no", "n"}:
        return "No"
    if "don't know" in s_low or "dont know" in s_low or s_low in {"dk", "idk"}:
        return "I don't know"

    return "I don't know"


def _normalize_prev_boss_serious_value_merged(x: object) -> str:
    """
    Normalizes raw 'mh_ph_prev_boss_serious' answers, but merges:
      - 'Yes, they all did' and 'Some did' -> 'At least some did'

    Buckets returned:
      At least some did
      None did
      I don't know
      Not applicable

    Missing is treated as "I don't know".
    """
    if pd.isna(x):
        return "I don't know"
    s = str(x).strip()
    s_low = s.lower()

    # merge requested categories
    if "yes, they all did" in s_low or s_low == "yes, they all did":
        return "At least some did"
    if s_low == "some did" or "some did" in s_low:
        return "At least some did"

    if s_low == "none did" or "none did" in s_low:
        return "None did"
    if "don't know" in s_low or "dont know" in s_low or s_low in {"dk", "idk"}:
        return "I don't know"
    if "not applicable" in s_low or s_low in {"na", "n/a", "n.a.", "n\\a"}:
        return "Not applicable"

    return "I don't know"


def plot_raw_boss_and_prev_boss_serious_stacked_by_cluster(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    boss_feature: str,
    prev_boss_feature: str,
    out_path: Path,
) -> None:
    """
    Combined raw feature plot: 'boss serious' + 'previous boss serious' (with merge)

    - boss_feature (mh_ph_boss_serious): Yes / No / I don't know
    - prev_boss_feature (mh_ph_prev_boss_serious), with merge:
        'Yes, they all did' + 'Some did' -> 'At least some did'
      then buckets:
        At least some did / None did / I don't know / Not applicable

    -> 3 * 4 = 12 joint categories

    Legend uses labels of the form:
      B:Yes P:At least some did
    """
    import matplotlib.pyplot as plt

    b = df_raw[boss_feature].map(_normalize_boss_serious_value)
    p = df_raw[prev_boss_feature].map(_normalize_prev_boss_serious_value_merged)

    joint = ("B:" + b + " P:" + p).astype(str)

    clusters = sorted(np.unique(labels_aligned).tolist())

    overall_counts = joint.value_counts(dropna=False)
    categories = overall_counts.sort_values(ascending=False).index.tolist()

    mat = []
    for c in clusters:
        s_c = joint[labels_aligned == c]
        vc = s_c.value_counts(dropna=False)
        total = len(s_c)
        props = [(float(vc.get(cat, 0)) / total) if total > 0 else 0.0 for cat in categories]
        mat.append(props)

    mat = np.array(mat, dtype=float)

    fig_w = 10
    fig_h = 4.5
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    for j, cat in enumerate(categories):
        plt.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    plt.xticks(x, [str(c) for c in clusters])
    plt.ylim(0, 1.0)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion within cluster")
    plt.title("Raw feature distribution by cluster (k=6): boss serious and previous boss serious")

    plt.legend(
        title="Answer (B=boss serious, P=previous boss serious)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def make_all_raw_feature_plots(df_raw: pd.DataFrame, labels_aligned: np.ndarray) -> None:
    """
    Creates one PNG per raw feature in df_raw (excluding respondent_id).
    """
    features = [c for c in df_raw.columns if c != "respondent_id"]

    print(f"\n=== Raw (pre-encoding) driver plots by cluster (k=6) ===")
    print(f"Input file: {DRIVERS_RAW_PATH}")
    print(f"Saving {len(features)} plots to: {RAW_PLOTS_DIR}")

    for i, feat in enumerate(features, start=1):
        safe_name = feat.replace("/", "_").replace("\\", "_").replace(":", "_")
        out_path = RAW_PLOTS_DIR / f"raw_dist_k6_{safe_name}.png"
        plot_raw_feature_stacked_by_cluster(df_raw, labels_aligned, feat, out_path)
        if i <= 5 or i == len(features):
            print(f"Saved: {out_path}")
        elif i == 6:
            print("... (suppressing further per-file logs)")

    # Additional combined plot: benefits + resources (12 joint categories)
    if ("benefits" in df_raw.columns) and ("resources" in df_raw.columns):
        out_path = RAW_PLOTS_DIR / "raw_dist_k6_benefits_and_resources.png"
        plot_raw_benefits_and_resources_stacked_by_cluster(
            df_raw,
            labels_aligned,
            benefits_feature="benefits",
            resources_feature="resources",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")
    else:
        missing = [c for c in ["benefits", "resources"] if c not in df_raw.columns]
        print(f"Skipping combined benefits/resources plot (missing columns: {missing})")

    # Additional combined plot: mh_interview + ph_interview (9 joint categories)
    if ("mh_interview" in df_raw.columns) and ("ph_interview" in df_raw.columns):
        out_path = RAW_PLOTS_DIR / "raw_dist_k6_mh_and_ph_interview.png"
        plot_raw_mh_and_ph_interview_stacked_by_cluster(
            df_raw,
            labels_aligned,
            mh_feature="mh_interview",
            ph_feature="ph_interview",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")
    else:
        missing = [c for c in ["mh_interview", "ph_interview"] if c not in df_raw.columns]
        print(f"Skipping combined mh/ph interview plot (missing columns: {missing})")

    # Additional combined plot: anonymity_protected + prev_anonymity_protected (with merge to ≥S)
    if ("anonymity_protected" in df_raw.columns) and ("prev_anonymity_protected" in df_raw.columns):
        out_path = RAW_PLOTS_DIR / "raw_dist_k6_current_and_previous_anonymity.png"
        plot_raw_current_and_previous_anonymity_stacked_by_cluster(
            df_raw,
            labels_aligned,
            current_feature="anonymity_protected",
            prev_feature="prev_anonymity_protected",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")
    else:
        missing = [c for c in ["anonymity_protected", "prev_anonymity_protected"] if c not in df_raw.columns]
        print(f"Skipping combined current/previous anonymity plot (missing columns: {missing})")

    # Additional combined plot: mhd coworker + consequences + bad response (force 50 combos)
    needed = ["mhdcoworker_you_not_reveal", "observed_mhdcoworker_bad_conseq", "ever_observed_mhd_bad_response"]
    if all(col in df_raw.columns for col in needed):
        out_path = RAW_PLOTS_DIR / "raw_dist_k6_mhd_coworker_combo.png"
        plot_raw_mhd_coworker_combo_stacked_by_cluster(
            df_raw,
            labels_aligned,
            reveal_feature="mhdcoworker_you_not_reveal",
            conseq_feature="observed_mhdcoworker_bad_conseq",
            ever_feature="ever_observed_mhd_bad_response",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")
    else:
        missing = [c for c in needed if c not in df_raw.columns]
        print(f"Skipping combined mhd coworker combo plot (missing columns: {missing})")

    # Additional combined plot: observe_vs_reveal (force 25 combos)
    if ("mhdcoworker_you_not_reveal" in df_raw.columns) and ("ever_observed_mhd_bad_response" in df_raw.columns):
        out_path = RAW_PLOTS_DIR / "raw_dist_k6_observe_vs_reveal.png"
        plot_raw_observe_vs_reveal_stacked_by_cluster(
            df_raw,
            labels_aligned,
            reveal_feature="mhdcoworker_you_not_reveal",
            observed_feature="ever_observed_mhd_bad_response",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")
    else:
        missing = [c for c in ["mhdcoworker_you_not_reveal", "ever_observed_mhd_bad_response"] if c not in df_raw.columns]
        print(f"Skipping combined observe_vs_reveal plot (missing columns: {missing})")

    # NEW combined plot: mh_ph_boss_serious + mh_ph_prev_boss_serious (with merge to "At least some did")
    if ("mh_ph_boss_serious" in df_raw.columns) and ("mh_ph_prev_boss_serious" in df_raw.columns):
        out_path = RAW_PLOTS_DIR / "raw_dist_k6_boss_and_prev_boss_serious.png"
        plot_raw_boss_and_prev_boss_serious_stacked_by_cluster(
            df_raw,
            labels_aligned,
            boss_feature="mh_ph_boss_serious",
            prev_boss_feature="mh_ph_prev_boss_serious",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")
    else:
        missing = [c for c in ["mh_ph_boss_serious", "mh_ph_prev_boss_serious"] if c not in df_raw.columns]
        print(f"Skipping combined boss/prev boss serious plot (missing columns: {missing})")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    X, feature_names = load_encoded_drivers()
    labels = load_labels()

    binary_feats, ordinal_feats = split_feature_types(X, feature_names)

    clusters = sorted(np.unique(labels))

    print("\n=== k=6 Cluster Sizes ===")
    for c in clusters:
        n = np.sum(labels == c)
        print(f"Cluster {c}: {n} ({n/len(labels)*100:.1f}%)")

    # ------------------------
    # ORIGINAL cluster vs rest tables
    # ------------------------
    for c in clusters:
        print("\n" + "="*100)
        print(f"CLUSTER {c}")
        print("="*100)

        print("\nBinary (Top 10 by deviation vs rest):")
        df_bin = build_binary_table(X, binary_feats, labels, c)
        print(df_bin.to_string(index=False))

        print("\nOrdinal (Top 10 by deviation vs rest):")
        df_ord = build_ordinal_table(X, ordinal_feats, labels, c)
        print(df_ord.to_string(index=False))

        # ------------------------
        # NEW: High-prevalence lists
        # ------------------------
        print(f"\nBinary features with proportion >= {THRESHOLD:.2f} in cluster:")
        df_bhi = list_binary_features_ge_threshold(X, binary_feats, labels, c, THRESHOLD)
        if df_bhi.empty:
            print("(none)")
        else:
            print(df_bhi.to_string(index=False))

        print(f"\nOrdinal values with share >= {THRESHOLD:.2f} in cluster (value shown is the qualifying answer):")
        df_ohi = list_ordinal_values_ge_threshold(X, ordinal_feats, labels, c, THRESHOLD)
        if df_ohi.empty:
            print("(none)")
        else:
            print(df_ohi.to_string(index=False))

    # ------------------------
    # Pairwise comparisons
    # ------------------------
    comparisons = [(0, 1), (3, 4), (2, 5)]

    for c1, c2 in comparisons:
        print("\n" + "=" * 110)
        print(f"CLUSTER {c1} vs CLUSTER {c2}")
        print("=" * 110)

        print("\nBinary (Top 10 separators):")
        df_bin = build_binary_pairwise(X, binary_feats, labels, c1, c2)
        print(df_bin.to_string(index=False))

        print("\nOrdinal (Top 10 separators):")
        df_ord = build_ordinal_pairwise(X, ordinal_feats, labels, c1, c2)
        print(df_ord.to_string(index=False))

    # ------------------------
    # NEW: Raw (pre-encoding) plots by cluster
    # ------------------------
    df_raw = load_preprocessed_drivers_raw()
    labels_raw = align_labels_to_raw(df_raw, labels)
    make_all_raw_feature_plots(df_raw, labels_raw)
