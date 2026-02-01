# %% [markdown]
# # Unified report (01–05): HR segmentation pipeline (OSMI 2016)
#
# This notebook merges the workflow previously split across:
#
# - `notebooks/01_data_intake_eda.py`
# - `notebooks/02_cleaning_feature_engineering.py`
# - `notebooks/03_model_comparison.py`
# - `notebooks/04_cluster_interpretation.py`
# - `notebooks/05_predictive_and_actions.py`
#
# ## How to read this notebook
#
# **Goal:** produce an HR-usable segmentation described by interpretable workplace levers, then overlay mental-health burden proxies
# for sizing and prioritization (association, not causation).
#
# **Key design choices (defensible defaults):**
# - **Population restriction:** focus on employees in tech-oriented contexts (HR levers have consistent meaning).
# - **Separation of variables:**
#   - *Drivers* define clusters: workplace levers + scored indices/components.
#   - *Overlays* are reported after: mental-health status/proxies + demographics/context.
# - **Missingness is not “just missing”:** survey routing creates structural missingness; we treat “not applicable” distinctly.
# - **Representation choices:**
#   - One-hot encodes categorical items and scales numeric indices.
#   - TruncatedSVD provides a defensible linear geometry.
#   - UMAP is treated as a structure-revealing tool (not a proof of cluster validity).
# - **Stability + interpretability checks:** stability across seeds and shallow-tree explainability are used as sanity checks.
#
# ## Output style (per your request)
#
# - **DataFrames are displayed** (not written to disk).
# - **Plots are shown inline** (not saved).
# - Markdown is intentionally detailed: it explains why each step exists and how to interpret outputs.
#
# If you want file outputs to `tmp/notebook_outputs/...`, use the individual notebooks.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))

from notebooks.lib import (
    load_df_raw,
    build_df_feat,
    feature_sets,
    build_preprocessor,
    to_object_for_cat,
    embed_svd,
    eval_kmeans,
    eval_gmm,
    eval_hdbscan,
    eval_umap_hdbscan,
    stability_ari,
    decision_tree_explainability,
    top_lift_drivers,
    MISSING_TOKEN,
)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)
PALETTE = sns.color_palette()

try:
    from IPython.display import display
except Exception:

    def display(x):  # type: ignore
        print(x)


def show_df(df: pd.DataFrame, title: str | None = None, head: int | None = None):
    if title:
        print(title)
    display(df.head(head) if head is not None else df)


def _clean_str(series: pd.Series) -> pd.Series:
    return series.astype(object).fillna("").astype(str).str.strip()


# %% [markdown]
# ## 1) Data intake + measurement reality (routing, missingness)
#
# The raw CSV uses full question text as column names. `load_df_raw()`:
# - loads the raw dataset,
# - renames columns to stable short “measure names” via a question→measure map,
# - provides a column mapping table for traceability.
#
# This is not “cosmetic”: it prevents fragile pipelines and makes every result auditable.

# %%
df_raw, df_raw_original, col_map = load_df_raw()
show_df(pd.DataFrame([{"df_raw_shape": df_raw.shape, "df_raw_original_shape": df_raw_original.shape}]))
show_df(col_map, "Question → measure map (head)", head=15)

# %% [markdown]
# ### 1.1 Missingness overview
#
# Why this matters:
# - The survey contains skip logic (routing). Many blanks are “not asked”, not “missing at random”.
# - Without explicitly handling routing, clustering can rediscover eligibility patterns rather than HR-lever differences.
#
# We start with a basic missingness audit: column-wise and row-wise.

# %%
overview = pd.DataFrame(
    [
        {"metric": "rows", "value": int(df_raw.shape[0])},
        {"metric": "cols", "value": int(df_raw.shape[1])},
        {"metric": "missing_cells", "value": int(df_raw.isna().sum().sum())},
        {"metric": "missing_rate_mean_over_cells", "value": float(df_raw.isna().mean().mean())},
        {"metric": "duplicate_rows", "value": int(df_raw.duplicated().sum())},
    ]
)
show_df(overview, "Raw missingness overview")

# %%
missingness = (
    pd.DataFrame(
        {
            "column": df_raw.columns,
            "dtype": [str(t) for t in df_raw.dtypes],
            "missing_rate": df_raw.isna().mean().values,
            "n_unique_non_missing": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
        }
    )
    .sort_values("missing_rate", ascending=False)
    .reset_index(drop=True)
)
show_df(missingness.head(25), "Top 25 columns by missing rate")

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top = missingness.head(25).copy().sort_values("missing_rate", ascending=True)
sns.barplot(data=top, x="missing_rate", y="column", color=PALETTE[0], ax=ax)
ax.set_title("Top 25 columns by missing rate (raw)")
ax.set_xlabel("Missing rate")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 4))
row_missing = df_raw.isna().sum(axis=1)
sns.histplot(row_missing, bins=30, color=PALETTE[0], ax=ax)
ax.set_title("Missing values per respondent (raw)")
ax.set_xlabel("Missing columns (count)")
ax.set_ylabel("Respondents")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.1.1 Redundancy among categorical variables (Cramér’s V heatmap)
#
# A large fraction of HR-lever survey items are categorical (Yes/No/Maybe/Don’t know, buckets, etc.).
# Many of these variables are **redundant** (highly associated), which creates two practical issues:
#
# 1) The one-hot encoded design matrix becomes wide and collinear.
# 2) Distance-based clustering in raw one-hot space becomes noisy and can overweight correlated blocks.
#
# This is one of the motivations for the **TruncatedSVD step** later: it compresses correlated one-hot structure into a smaller number
# of continuous components.
#
# We visualize redundancy using **Cramér’s V** (an association measure for categorical variables):
# - 0 ≈ no association,
# - 1 ≈ strong association.
#
# Note: this is computed on a selected set of “core” categorical workplace/context variables (not every high-cardinality text field).

# %%
try:
    from scipy.stats import chi2_contingency

    def cramers_v(a: pd.Series, b: pd.Series) -> float:
        x = _clean_str(a).replace("", "__MISSING__")
        y = _clean_str(b).replace("", "__MISSING__")
        tab = pd.crosstab(x, y)
        if tab.shape[0] <= 1 or tab.shape[1] <= 1:
            return float("nan")
        chi2 = chi2_contingency(tab.values, correction=False)[0]
        n = tab.values.sum()
        if n <= 0:
            return float("nan")
        r, k = tab.shape
        phi2 = max(0.0, chi2 / n)
        # Bias correction (Bergsma & Wicher) to reduce inflation in small tables.
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
        rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
        kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
        denom = max(min(kcorr - 1, rcorr - 1), 1e-12)
        return float(np.sqrt(phi2corr / denom))


    candidate_cats = [
        "company_size",
        "remote_work",
        "benefits",
        "benefits_options_known",
        "formal_discussion",
        "resources_available",
        "anonymity_protected",
        "leave_ease",
        "employer_serious",
        "mental_health_consequences",
        "physical_health_consequences",
        "coworker_comfort",
        "supervisor_comfort",
        "observed_negative_consequences",
        "career_harm",
        "team_views_negative",
        "tech_company",
        "it_worker",
    ]
    cat_cols = [c for c in candidate_cats if c in df_raw.columns]
    # Keep only columns with manageable cardinality (avoid giant sparse crosstabs).
    cat_cols = [c for c in cat_cols if df_raw[c].nunique(dropna=True) <= 15]

    V = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for i, ci in enumerate(cat_cols):
        for j, cj in enumerate(cat_cols):
            if j < i:
                V.loc[ci, cj] = V.loc[cj, ci]
                continue
            V.loc[ci, cj] = 1.0 if ci == cj else cramers_v(df_raw[ci], df_raw[cj])

    show_df(V, "Cramér’s V association matrix (selected categorical variables)")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(V, vmin=0, vmax=1, cmap="mako", ax=ax, cbar_kws={"label": "Cramér’s V"})
    ax.set_title("Categorical redundancy (Cramér’s V) — motivation for SVD compression")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Cramér’s V heatmap skipped (scipy not available or computation failed):", repr(e))

# %% [markdown]
# ### 1.2 Routing gates and why we restrict the population
#
# The HR deliverable is an intervention plan for organizations. Many employer-policy items are not comparable for self-employed respondents,
# so the main HR segmentation is restricted to employees.
#
# In addition, we focus on tech-oriented contexts (as encoded in the survey) to reduce heterogeneity that would dilute actionability.

# %%
counts = {}
if "self_employed" in df_raw.columns:
    counts["n_total"] = int(len(df_raw))
    counts["n_employees"] = int((df_raw["self_employed"] == 0).sum())
    counts["n_self_employed"] = int((df_raw["self_employed"] == 1).sum())
show_df(pd.DataFrame([counts]), "Routing: self-employed split")

# %% [markdown]
# ### 1.3 Raw burden proxy context (base rates)
#
# Burden proxies (outcomes) are **not used to define clusters**. They are overlaid later.
# Showing their raw distributions early prevents “surprise” when base rates differ after filtering.

# %%
if "current_disorder" in df_raw.columns:
    s = _clean_str(df_raw["current_disorder"]).replace("", "__MISSING__")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    sns.countplot(x=s, order=[o for o in ["Yes", "No", "__MISSING__"] if o in s.unique()], color=PALETTE[0], ax=ax)
    ax.set_title("Current disorder (raw)")
    ax.set_xlabel("")
    ax.set_ylabel("Respondents")
    plt.tight_layout()
    plt.show()

if "treatment_sought" in df_raw.columns:
    s = _clean_str(df_raw["treatment_sought"]).replace("", "__MISSING__")
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    order = [o for o in ["Yes", "No", "1", "0", "__MISSING__"] if o in s.unique()]
    sns.countplot(x=s, order=order, color=PALETTE[0], ax=ax)
    ax.set_title("Treatment sought (raw)")
    ax.set_xlabel("")
    ax.set_ylabel("Respondents")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 1.4 Basic EDA (age outliers, top gender strings, top roles)
#
# These are not “model features” yet; they are reality checks:
# - Age can contain outliers or implausible values in surveys; later we keep only a plausible range.
# - Gender and role fields are high-cardinality / messy text. For modeling we use cleaned `gender_norm` and role indicators, but raw counts
#   help justify those transformations.

# %%
if "age" in df_raw.columns:
    age = pd.to_numeric(df_raw["age"], errors="coerce")
    show_df(
        pd.DataFrame(
            [
                {
                    "min": float(age.min()),
                    "p01": float(age.quantile(0.01)),
                    "median": float(age.quantile(0.50)),
                    "p99": float(age.quantile(0.99)),
                    "max": float(age.max()),
                    "n_lt_16": int((age < 16).sum()),
                    "n_gt_80": int((age > 80).sum()),
                }
            ]
        ),
        "Age summary (raw, highlights outliers)",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(age.dropna(), bins=35, color=PALETTE[0], alpha=0.9, ax=ax)
    ax.set_title("Age distribution (raw)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Respondents")
    plt.tight_layout()
    plt.show()

# %%
if "gender" in df_raw.columns:
    gender_counts = _clean_str(df_raw["gender"]).replace("", "__MISSING__").value_counts().head(25).rename_axis("gender_raw").reset_index(name="n")
    show_df(gender_counts, "Top 25 raw gender strings (high-cardinality field)")

# %%
if "work_position" in df_raw.columns:
    pos_counts = _clean_str(df_raw["work_position"]).replace("", "__MISSING__").value_counts().head(25).rename_axis("work_position_raw").reset_index(name="n")
    show_df(pos_counts, "Top 25 raw work_position strings (multi-select text)")

# %% [markdown]
# ## 2) Cleaning + feature engineering (HR levers + interpretable scores)
#
# `build_df_feat()` constructs the modeling dataset. It is the “ground truth” transformation for later notebooks.
#
# What it does (high-level):
# - filters to employees in tech-oriented contexts,
# - performs routing-aware missingness handling for selected blocks (`__not_applicable__`),
# - creates interpretable scored items and indices:
#   - `sc_support__*`, `idx_support`
#   - `sc_safety__*`, `idx_safety`
# - adds response-quality diagnostics:
#   - `qc_unknown_count` (how often respondents answer “I don’t know” on workplace items),
#   - `qc_missing_count` (how many workplace items are missing).
#
# Why both indices and item-level scores exist:
# - indices are interpretable and stable, good for HR dashboards,
# - item-level scores preserve information and support lift/driver interpretation.

# %%
df_feat = build_df_feat()
show_df(pd.DataFrame([{"df_feat_shape": df_feat.shape}]), "Modeling dataset shape")
sets = feature_sets(df_feat)
show_df(pd.DataFrame([{"feature_set": k, "n_cols": len(v)} for k, v in sets.items()]).sort_values("n_cols", ascending=False), "Feature sets (column counts)")

# %% [markdown]
# ### 2.1 Population flow: raw → employees → tech-oriented → final df_feat
#
# This documents the preprocessing funnel and makes it explicit that “the model sees a subset”.

# %%
employees_n = int((df_raw["self_employed"] == 0).sum()) if "self_employed" in df_raw.columns else int(len(df_raw))
non_tech_removed = int(employees_n - len(df_feat))
show_df(
    pd.DataFrame(
        [
            {"stage": "raw_total", "n": int(len(df_raw))},
            {"stage": "employees_only", "n": employees_n},
            {"stage": "removed (non-tech-oriented / routing constraints)", "n": non_tech_removed},
            {"stage": "final_df_feat", "n": int(len(df_feat))},
        ]
    ),
    "Population flow counts",
)

# %% [markdown]
# ### 2.2 Index distributions (compact, interpretable axes)
#
# These distributions show:
# - whether indices have usable variation (not all 0/1),
# - whether values are concentrated (which can limit discriminative power),
# - and whether extreme missingness exists (coverage).

# %%
idx_cols = [c for c in ["idx_support", "idx_safety", "idx_support_n", "idx_safety_n"] if c in df_feat.columns]
show_df(df_feat[idx_cols].describe().T, "Index summaries")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df_feat["idx_support"].dropna(), bins=25, ax=axes[0], color=PALETTE[0])
axes[0].set_title("idx_support")
sns.histplot(df_feat["idx_safety"].dropna(), bins=25, ax=axes[1], color=PALETTE[0])
axes[1].set_title("idx_safety")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.2.1 The “benefits visibility gap” KPI
#
# The engineered KPI `gap__benefits_yes_but_not_yes_options` is a dominant binary axis in many survey analyses:
#
# - it captures a concrete HR-lever problem: “benefits exist” but “options are not discoverable/known”,
# - it often separates “well-supported” vs “confused/unaware” subpopulations.
#
# We show its distribution early because it can dominate clustering if included naïvely.

# %%
gap = "gap__benefits_yes_but_not_yes_options"
if gap in df_feat.columns:
    show_df(df_feat[gap].value_counts(dropna=False).rename_axis(gap).to_frame("n"), "Gap KPI distribution (counts)")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x=gap, data=df_feat, color=PALETTE[0], ax=ax)
    ax.set_title("Gap KPI: benefits yes, options not yes")
    ax.set_xlabel("")
    ax.set_ylabel("Respondents")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 2.2.2 Correlation among engineered numeric features (Spearman)
#
# We use Spearman correlation because several engineered variables are:
# - ordinal / bounded (0–1 scores),
# - counts with skew,
# - and monotonic relationships are more plausible than linear ones.
#
# This plot is a *diagnostic*:
# - strong correlations indicate redundancy (useful for interpretation, may affect coefficients),
# - it also checks that “QC” variables are not perfectly entangled with indices.

# %%
num_cols = [
    "idx_support",
    "idx_safety",
    "qc_unknown_count",
    "qc_missing_count",
    "role__count",
    "role__multi_role",
    "age_clean",
    "cond__any_reported",
    gap,
]
num_cols = [c for c in num_cols if c in df_feat.columns]
if num_cols:
    corr = df_feat[num_cols].corr(method="spearman")
    show_df(corr, "Engineered numeric Spearman correlation")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Engineered numeric features correlation (Spearman)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 2.3 Outcome base rates: original vs filtered
#
# Filtering changes population composition. Any “predictive performance” has to be interpreted relative to these base rates.
# We also track “not applicable” coverage for interference outcomes.

# %%
def _summarize_binary(series: pd.Series, positive_values: set[str], not_applicable_value: str | None = None):
    s = _clean_str(series)
    missing = s.eq("")
    not_applicable = s.eq(not_applicable_value) if not_applicable_value is not None else pd.Series(False, index=s.index)
    applicable = ~missing & ~not_applicable
    pos = s.isin(list(positive_values)) & applicable
    n_total = int(len(s))
    n_applicable = int(applicable.sum())
    return {
        "n_total": n_total,
        "n_applicable": n_applicable,
        "n_positive": int(pos.sum()),
        "positive_rate_applicable": float(pos.sum() / n_applicable) if n_applicable > 0 else float("nan"),
        "missing_rate": float(missing.mean()),
        "not_applicable_rate": float(not_applicable.mean()),
    }


outcome_defs = [
    {"outcome": "current_disorder_yes", "col": "current_disorder", "pos": {"Yes"}, "na": None},
    {"outcome": "treatment_yes", "col": "treatment_sought", "pos": {"Yes", "1", "1.0"}, "na": None},
    {"outcome": "work_interference_untreated_often", "col": "work_interference_untreated", "pos": {"Often"}, "na": "Not applicable to me"},
]

rows = []
for dset_name, frame in [("original", df_raw), ("filtered_df_feat", df_feat)]:
    for od in outcome_defs:
        if od["col"] not in frame.columns:
            continue
        rows.append({"dataset": dset_name, "outcome": od["outcome"], **_summarize_binary(frame[od["col"]], od["pos"], od["na"])})

mh_rates = pd.DataFrame(rows).sort_values(["outcome", "dataset"]).reset_index(drop=True)
show_df(mh_rates, "Outcome base rates and coverage")

# %%
fig, ax = plt.subplots(figsize=(10, 4))
plot_df = mh_rates.copy()
plot_df = plot_df[plot_df.groupby("outcome")["n_applicable"].transform("min") > 0].copy()
plot_df["positive_rate_pct_applicable"] = plot_df["positive_rate_applicable"] * 100.0
sns.barplot(data=plot_df, x="outcome", y="positive_rate_pct_applicable", hue="dataset", ax=ax)
ax.set_title("Outcome base rates: original vs filtered (among applicable)")
ax.set_xlabel("")
ax.set_ylabel("Positive rate (%)")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3) Model comparison (KMeans/GMM/HDBSCAN, SVD space vs UMAP space)
#
# We compare:
# - **KMeans/GMM** as baseline convex clusterers in SVD space,
# - **HDBSCAN** in SVD space (density-based on linear geometry),
# - **UMAP + HDBSCAN** as exploratory (nonlinear embedding + density clustering).
#
# We do not “optimize silhouette” as the objective. For HR segmentation, we want:
# - stable clusters,
# - interpretable levers,
# - and segment sizes usable for interventions.
#
# ### Metrics (what to trust, what not to overread)
#
# - **Silhouette** can be low even for meaningful segments when clusters overlap (common in survey data). Treat as a diagnostic, not a goal.
# - **Inertia** always decreases with k; use it only as an elbow heuristic.
# - **GMM BIC/AIC** are likelihood criteria under Gaussian assumptions; for one-hot survey data, treat as heuristics.
# - **HDBSCAN noise fraction** matters operationally: too much noise means many respondents are not assigned to a segment.
# - **Decision-tree explainability (macro-F1)** is a sanity check: can we describe clusters with simple rules on interpretable levers?

# %%
compare_sets = {
    "workplace_no_roles": sets["workplace_no_roles"],
    "workplace_no_roles_no_gap": sets["workplace_no_roles_no_gap"],
    "scored_items_no_gap": sets["scored_items_no_gap"],
}
k_range = [2, 3, 4, 5, 6, 7, 8]
explain_X = df_feat[sets["scored_items_no_gap"]].copy()

# %%
rows = []
for set_name, cols in compare_sets.items():
    X = to_object_for_cat(df_feat[cols].copy())
    pre = build_preprocessor(X, min_frequency=10)
    Xt = pre.fit_transform(X)
    _, Z = embed_svd(Xt, n_components=15)

    for k in k_range:
        km = eval_kmeans(Z, k)
        rows.append(
            {
                "feature_set": set_name,
                "model": "kmeans",
                "k": int(k),
                "silhouette_svd": km["silhouette"],
                "inertia": km["inertia"],
                "bic": np.nan,
                "aic": np.nan,
            }
        )

    for cov in ["diag"]:
        for k in k_range:
            gg = eval_gmm(Z, k, covariance_type=cov)
            rows.append(
                {
                    "feature_set": set_name,
                    "model": f"gmm_{cov}",
                    "k": int(k),
                    "silhouette_svd": gg["silhouette"],
                    "inertia": np.nan,
                    "bic": gg["bic"],
                    "aic": gg["aic"],
                }
            )

baseline = pd.DataFrame(rows)
show_df(baseline.head(30), "Baseline (KMeans + GMM) metrics (head)")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
tmp = baseline[baseline["model"].eq("kmeans")].copy()
sns.lineplot(data=tmp, x="k", y="silhouette_svd", hue="feature_set", marker="o", ax=ax)
ax.set_title("KMeans silhouette (SVD space)")
ax.set_xlabel("k")
ax.set_ylabel("silhouette")
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 4))
tmp = baseline[baseline["model"].str.startswith("gmm_")].copy()
sns.lineplot(data=tmp, x="k", y="bic", hue="model", style="feature_set", marker="o", ax=ax)
ax.set_title("GMM BIC across feature sets (lower is better; heuristic)")
ax.set_xlabel("k")
ax.set_ylabel("BIC")
plt.tight_layout()
plt.show()

# %%
rows = []
for set_name, cols in compare_sets.items():
    X = to_object_for_cat(df_feat[cols].copy())
    pre = build_preprocessor(X, min_frequency=10)
    Xt = pre.fit_transform(X)
    _, Z = embed_svd(Xt, n_components=15)
    for metric in ["euclidean", "manhattan"]:
        for min_cluster_size in [25, 50]:
            res = eval_hdbscan(Z, min_cluster_size=min_cluster_size, min_samples=None, metric=metric)
            exp = decision_tree_explainability(explain_X, res["labels"], max_depth=3, n_splits=5, seed=0)
            rows.append({"feature_set": set_name, "metric": metric, "min_cluster_size": int(min_cluster_size), **res, **exp})

hdb_svd = pd.DataFrame(rows).sort_values(["relative_validity", "silhouette_inliers"], ascending=[False, False])
show_df(hdb_svd.head(20), "HDBSCAN on SVD space (top configs)")

# %%
rows = []
for set_name, cols in compare_sets.items():
    X = to_object_for_cat(df_feat[cols].copy())
    pre = build_preprocessor(X, min_frequency=10)
    Xt = pre.fit_transform(X)
    _, Z = embed_svd(Xt, n_components=15)

    configs = [
        {"n_neighbors": 10, "min_dist": 0.1, "min_cluster_size": 50},
        {"n_neighbors": 30, "min_dist": 0.1, "min_cluster_size": 50},
        {"n_neighbors": 10, "min_dist": 0.0, "min_cluster_size": 50},
        {"n_neighbors": 10, "min_dist": 0.1, "min_cluster_size": 25},
    ]
    for cfg in configs:
        res = eval_umap_hdbscan(Z, umap_dim=5, n_neighbors=cfg["n_neighbors"], min_dist=cfg["min_dist"], min_cluster_size=cfg["min_cluster_size"], min_samples=None, seed=0)
        exp = decision_tree_explainability(explain_X, res["labels"], max_depth=3, n_splits=5, seed=0)
        labels_seeds = [
            eval_umap_hdbscan(Z, umap_dim=5, n_neighbors=cfg["n_neighbors"], min_dist=cfg["min_dist"], min_cluster_size=cfg["min_cluster_size"], min_samples=None, seed=s)["labels"]
            for s in [0, 1, 2]
        ]
        rows.append({**cfg, "feature_set": set_name, **{k: res[k] for k in ["n_clusters", "noise_frac", "relative_validity", "silhouette_umap_inliers", "silhouette_svd_inliers"]}, "stability_ari_seeds_0_1_2": stability_ari(labels_seeds), **exp})

umap_hdb = pd.DataFrame(rows).sort_values(["tree_macro_f1", "relative_validity", "silhouette_umap_inliers"], ascending=[False, False, False])
show_df(umap_hdb.head(20), "UMAP + HDBSCAN configs (top)")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
tmp = umap_hdb[umap_hdb["feature_set"].eq("scored_items_no_gap")].copy()
sns.scatterplot(
    data=tmp,
    x="silhouette_svd_inliers",
    y="silhouette_umap_inliers",
    hue="n_clusters",
    size="noise_frac",
    sizes=(20, 200),
    ax=ax,
)
ax.set_title("UMAP+HDBSCAN tradeoff: silhouette in SVD vs UMAP space (scored_items_no_gap)")
ax.set_xlabel("silhouette (SVD, inliers)")
ax.set_ylabel("silhouette (UMAP, inliers)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.1 Candidate selection (what we carry into interpretation)
#
# We prefer configurations with:
# - acceptable noise fraction (too much noise reduces usability),
# - a reasonable number of clusters (neither trivial 2 nor exploding into many tiny groups),
# - stability across UMAP seeds,
# - and shallow-tree explainability (clusters correspond to simple patterns in levers).

# %%
cand = umap_hdb[umap_hdb["feature_set"].eq("scored_items_no_gap")].copy()
cand = cand[cand["noise_frac"] <= 0.2].copy()
cand = cand[cand["n_clusters"].between(3, 10)].copy()
cand = cand.sort_values(["tree_macro_f1", "stability_ari_seeds_0_1_2", "relative_validity"], ascending=[False, False, False]).head(15)
show_df(cand, "Finalist configurations (top)")

# %% [markdown]
# ## 4) Cluster interpretation (profiles, drivers, overlays)
#
# We fix one configuration (as in notebook 04) and interpret clusters through:
# - cluster profiles (indices and key HR items),
# - lift tables (“drivers”),
# - mental-health outcome overlays (post-hoc).
#
# Important: clusters are defined on HR levers; burden outcomes are descriptive overlays.

# %%
config_final = {
    "feature_set": "scored_items_no_gap",
    "min_frequency": 10,
    "svd_dim": 15,
    "umap_dim": 5,
    "n_neighbors": 10,
    "min_dist": 0.1,
    "min_cluster_size": 50,
    "min_samples": None,
    "seed": 0,
}

cols = sets[config_final["feature_set"]]
X = to_object_for_cat(df_feat[cols].copy())
pre = build_preprocessor(X, min_frequency=config_final["min_frequency"])
Xt = pre.fit_transform(X)
_, Z = embed_svd(Xt, n_components=config_final["svd_dim"])
res = eval_umap_hdbscan(
    Z,
    umap_dim=config_final["umap_dim"],
    n_neighbors=config_final["n_neighbors"],
    min_dist=config_final["min_dist"],
    min_cluster_size=config_final["min_cluster_size"],
    min_samples=config_final["min_samples"],
    seed=config_final["seed"],
)
labels = res["labels"]
Zu = res["Zu"]
show_df(pd.Series(labels).value_counts().sort_index().to_frame("n"), "Cluster sizes (including -1 noise)")

# %%
if Zu.shape[1] >= 2:
    emb = pd.DataFrame({"x": Zu[:, 0], "y": Zu[:, 1], "cluster": labels})
    fig, ax = plt.subplots(figsize=(7, 6))
    hue_order = sorted([int(c) for c in emb["cluster"].dropna().unique().tolist()])
    sns.scatterplot(data=emb, x="x", y="y", hue="cluster", hue_order=hue_order, palette=sns.color_palette("deep", n_colors=len(hue_order)), s=18, alpha=0.85, ax=ax)
    ax.set_title("UMAP embedding with HDBSCAN clusters")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

# %%
def pct(series, value):
    s = series.astype(object).fillna("__MISSING__").astype(str).str.strip()
    return float((s == value).mean() * 100.0)


profile_rows = []
dfp = df_feat.copy()
dfp["cluster"] = labels
for cl, g in dfp[dfp["cluster"] != -1].groupby("cluster"):
    r = {"cluster": int(cl), "n": int(len(g))}
    for col in ["idx_support", "idx_safety", "qc_unknown_count", "qc_missing_count"]:
        if col in g.columns:
            r[f"{col}_mean"] = float(g[col].mean())
    if "benefits" in g.columns:
        r["benefits_yes_pct"] = pct(g["benefits"], "Yes")
    if "benefits_options_known" in g.columns:
        r["options_yes_pct"] = pct(g["benefits_options_known"], "Yes")
        r["options_no_pct"] = pct(g["benefits_options_known"], "No")
    if "anonymity_protected" in g.columns:
        r["anonymity_idk_pct"] = pct(g["anonymity_protected"], "I don't know")
    if "resources_available" in g.columns:
        r["resources_no_pct"] = pct(g["resources_available"], "No")
    if "supervisor_comfort" in g.columns:
        r["supervisor_no_pct"] = pct(g["supervisor_comfort"], "No")
    profile_rows.append(r)

profile = pd.DataFrame(profile_rows).sort_values("cluster").reset_index(drop=True)
show_df(profile, "Cluster profiles (selected KPIs)")

# %%
persona_map = {
    0: "Privacy-Seekers",
    1: "The Unaware",
    2: "The At-Risk",
    3: "The Disconnected",
    4: "The Self-Reliant",
    5: "Well-Supported",
}

# %% [markdown]
# ### 4.0.1 Full profile table for HR (counts + levers + burden overlays)
#
# The table below is designed to contain *all information needed* to build an HR-facing summary such as:
#
# - cluster id and persona label,
# - segment size,
# - “defining characteristics” in plain language with **data-verified percentages**,
# - burden overlays (e.g., current disorder rate),
# - support/safety index means.
#
# We compute both:
# - **within-cluster percentages** (what is true inside this persona), and
# - **lift** vs overall (what is unusually high/low compared to the whole modeling population).

# %%
def _pct_of_value(series: pd.Series, value: str) -> float:
    s = series.astype(object).fillna("__MISSING__").astype(str).str.strip()
    return float((s == value).mean() * 100.0)


def _rate_current_disorder(frame: pd.DataFrame) -> float:
    if "current_disorder" not in frame.columns:
        return float("nan")
    s = _clean_str(frame["current_disorder"])
    return float(s.eq("Yes").mean() * 100.0)


def _rate_treatment(frame: pd.DataFrame) -> float:
    if "treatment_sought" not in frame.columns:
        return float("nan")
    s = _clean_str(frame["treatment_sought"])
    return float(s.isin(["Yes", "1", "1.0"]).mean() * 100.0)


df_in = df_feat.copy()
df_in["cluster"] = labels
df_in = df_in[df_in["cluster"] != -1].copy()
df_in["label"] = df_in["cluster"].map(persona_map).fillna("Persona (unlabeled)")

overall = df_in.copy()

cat_specs = [
    ("anonymity_protected", ["No", "I don't know", "Yes"]),
    ("supervisor_comfort", ["No", "Yes"]),
    ("resources_available", ["No", "Yes", "I don't know"]),
    ("benefits", ["No", "Yes"]),
    ("benefits_options_known", ["No", "Yes", "__not_applicable__"]),
    ("employer_serious", ["No", "Yes"]),
]
cat_specs = [(c, vals) for c, vals in cat_specs if c in df_in.columns]

full_rows = []
for cl, g in df_in.groupby(["cluster", "label"]):
    cluster_id, label = int(cl[0]), str(cl[1])
    row = {
        "cluster": cluster_id,
        "label": label,
        "count": int(len(g)),
        "idx_support_mean": float(g["idx_support"].mean()) if "idx_support" in g.columns else float("nan"),
        "idx_safety_mean": float(g["idx_safety"].mean()) if "idx_safety" in g.columns else float("nan"),
        "current_disorder_rate_pct": _rate_current_disorder(g),
        "treatment_sought_rate_pct": _rate_treatment(g),
    }
    for col, values in cat_specs:
        for v in values:
            row[f"{col}__{v}__pct"] = _pct_of_value(g[col], v)
            row[f"{col}__{v}__lift_pp"] = row[f"{col}__{v}__pct"] - _pct_of_value(overall[col], v)
    full_rows.append(row)

cluster_profile_full = pd.DataFrame(full_rows).sort_values("cluster").reset_index(drop=True)
show_df(cluster_profile_full, "Cluster profile (full table: levers + overlays)")

# %%
# Cleaner version: only the metrics referenced in the HR "Corrected defining characteristics" bullets.
keep_map = {
    "cluster": "cluster",
    "label": "label",
    "count": "count",
    "idx_support_mean": "idx_support_mean",
    "idx_safety_mean": "idx_safety_mean",
    "current_disorder_rate_pct": "current_disorder_rate_pct",
    "anonymity_protected__No__pct": "anonymity_no_pct",
    "anonymity_protected__I don't know__pct": "anonymity_dk_pct",
    "supervisor_comfort__No__pct": "supervisor_no_pct",
    "resources_available__No__pct": "resources_no_pct",
    "benefits__No__pct": "benefits_no_pct",
    "benefits__Yes__pct": "benefits_yes_pct",
    "benefits_options_known__No__pct": "options_no_pct",
    "benefits_options_known__Yes__pct": "options_yes_pct",
    "employer_serious__No__pct": "employer_serious_no_pct",
}

cols_present = [c for c in keep_map.keys() if c in cluster_profile_full.columns]
cluster_profile_key = cluster_profile_full[cols_present].rename(columns={c: keep_map[c] for c in cols_present}).copy()

for c in cluster_profile_key.columns:
    if c.endswith("_pct") or c.endswith("_rate_pct"):
        cluster_profile_key[c] = cluster_profile_key[c].astype(float).round(1)
for c in ["idx_support_mean", "idx_safety_mean"]:
    if c in cluster_profile_key.columns:
        cluster_profile_key[c] = cluster_profile_key[c].astype(float).round(2)

show_df(cluster_profile_key, "Cluster profile (key metrics used in defining characteristics)")

# %% [markdown]
# ### 4.0.2 “Corrected defining characteristics” (data-verified bullets)
#
# We generate a concise, HR-readable list of defining characteristics per cluster:
#
# - For categorical levers, we select the strongest differentiators based on absolute lift (percentage point difference vs overall).
# - We also add burden/indices highlights when a cluster has the highest/lowest value across clusters.
#
# This produces a table format that matches an HR summary slide: **Cluster / Label / Count / Defining characteristics**.

# %%
def _format_pct(x: float) -> str:
    return f"{x:.1f}%"


def _pretty_feature(col: str) -> str:
    mapping = {
        "anonymity_protected": "Anonymity protected",
        "supervisor_comfort": "Supervisor comfort",
        "resources_available": "Resources available",
        "benefits": "Benefits",
        "benefits_options_known": "Options known",
        "employer_serious": "Employer seriousness",
    }
    return mapping.get(col, col)


def _pretty_answer(ans: str) -> str:
    if ans == "I don't know":
        return "\"Don't know\""
    if ans == "__not_applicable__":
        return "Not applicable"
    return f"\"{ans}\""


lift_cols = [c for c in cluster_profile_full.columns if c.endswith("__lift_pp")]
lift_long = []
for _, r in cluster_profile_full.iterrows():
    for c in lift_cols:
        pct_col = c.replace("__lift_pp", "__pct")
        base = c.split("__lift_pp")[0]
        feat, ans = base.split("__", 1)
        lift_long.append(
            {
                "cluster": int(r["cluster"]),
                "label": str(r["label"]),
                "feature": feat,
                "answer": ans,
                "pct_in_cluster": float(r[pct_col]),
                "lift_pp": float(r[c]),
                "abs_lift": abs(float(r[c])),
            }
        )

lift_long = pd.DataFrame(lift_long)

max_disorder = float(cluster_profile_full["current_disorder_rate_pct"].max())
min_disorder = float(cluster_profile_full["current_disorder_rate_pct"].min())
max_support = float(cluster_profile_full["idx_support_mean"].max()) if "idx_support_mean" in cluster_profile_full.columns else float("nan")
max_safety = float(cluster_profile_full["idx_safety_mean"].max()) if "idx_safety_mean" in cluster_profile_full.columns else float("nan")

rows = []
for _, r in cluster_profile_full.iterrows():
    cl = int(r["cluster"])
    label = str(r["label"])
    n = int(r["count"])

    dd = lift_long[lift_long["cluster"] == cl].copy()
    # Prefer levers that are most actionable/central for HR; then rank by absolute lift.
    prefer_feats = ["anonymity_protected", "supervisor_comfort", "resources_available", "benefits", "benefits_options_known", "employer_serious"]
    dd["feat_rank"] = dd["feature"].map(lambda f: prefer_feats.index(f) if f in prefer_feats else 999)
    dd = dd.sort_values(["feat_rank", "abs_lift"], ascending=[True, False])
    dd = dd.groupby("feature", as_index=False).head(1).sort_values("abs_lift", ascending=False).head(4)

    bullets = []
    for _, d in dd.iterrows():
        bullets.append(f"• {_format_pct(d['pct_in_cluster'])} {_pretty_answer(d['answer'])} to {_pretty_feature(d['feature'])}")

    disorder = float(r["current_disorder_rate_pct"])
    if np.isfinite(disorder):
        if disorder == max_disorder:
            bullets.insert(0, f"• {_format_pct(disorder)} current disorder rate (highest)")
        elif disorder == min_disorder:
            bullets.insert(0, f"• {_format_pct(disorder)} current disorder rate (lowest)")

    if "idx_support_mean" in r and np.isfinite(float(r["idx_support_mean"])) and float(r["idx_support_mean"]) == max_support:
        bullets.append(f"• Highest support index (mean={float(r['idx_support_mean']):.2f})")
    if "idx_safety_mean" in r and np.isfinite(float(r["idx_safety_mean"])) and float(r["idx_safety_mean"]) == max_safety:
        bullets.append(f"• Highest safety index (mean={float(r['idx_safety_mean']):.2f})")

    rows.append(
        {
            "cluster": cl,
            "label": label,
            "count": n,
            "corrected_defining_characteristics_data_verified": "\n".join(bullets),
        }
    )

cluster_hr_summary = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
show_df(cluster_hr_summary, "HR summary table: Cluster / Label / Count / Corrected defining characteristics")

# %%
heat_cols = [c for c in profile.columns if c.endswith("_mean")]
if heat_cols:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    Zp = profile.set_index("cluster")[heat_cols]
    Zz = (Zp - Zp.mean()) / (Zp.std(ddof=0) + 1e-9)
    sns.heatmap(Zz, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "z-score"})
    ax.set_title("Cluster profile (z-scored means)")
    plt.tight_layout()
    plt.show()

# %%
driver_feats = [
    "benefits",
    "benefits_options_known",
    "resources_available",
    "anonymity_protected",
    "supervisor_comfort",
    "coworker_comfort",
    "mental_health_consequences",
    "career_harm",
    "leave_ease",
    "employer_serious",
    "remote_work",
    "company_size",
]
drivers = top_lift_drivers(df_feat, labels, driver_feats, top_n=12)
show_df(drivers.head(40), "Top lift drivers (head)")

# %% [markdown]
# ### 4.1 Persona labels and “traffic light” summary
#
# Numeric cluster IDs are convenient for modeling but not for HR communication.
# We map clusters to persona-style labels based on the dominant differentiators and KPI patterns.
# The “traffic light” table is a compact HR dashboard view across key levers (support, safety, anonymity clarity, benefits navigation).

# %%
df_seg = df_feat.copy()
df_seg["cluster"] = labels
df_seg = df_seg[df_seg["cluster"] != -1].copy()
df_seg["persona"] = df_seg["cluster"].map(persona_map).fillna("Persona (unlabeled)")


def traffic_light(value, q_low, q_high, higher_is_better=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if higher_is_better:
        if value <= q_low:
            return "RED"
        if value >= q_high:
            return "GREEN"
        return "YELLOW"
    if value >= q_high:
        return "RED"
    if value <= q_low:
        return "GREEN"
    return "YELLOW"


traffic = profile[["cluster", "n"]].copy()
traffic["persona"] = traffic["cluster"].map(persona_map).fillna("Persona (unlabeled)")

q_support = df_feat["idx_support"].quantile([0.25, 0.75]).to_dict() if "idx_support" in df_feat.columns else {}
q_safety = df_feat["idx_safety"].quantile([0.25, 0.75]).to_dict() if "idx_safety" in df_feat.columns else {}

if "idx_support_mean" in profile.columns:
    traffic["support_level"] = profile["idx_support_mean"].map(lambda v: traffic_light(v, q_support.get(0.25, np.nan), q_support.get(0.75, np.nan), True))
if "idx_safety_mean" in profile.columns:
    traffic["safety_level"] = profile["idx_safety_mean"].map(lambda v: traffic_light(v, q_safety.get(0.25, np.nan), q_safety.get(0.75, np.nan), True))
if "anonymity_idk_pct" in profile.columns:
    q_idk = profile["anonymity_idk_pct"].quantile([0.25, 0.75]).to_dict()
    traffic["anonymity_clarity"] = profile["anonymity_idk_pct"].map(lambda v: traffic_light(v, q_idk.get(0.25, np.nan), q_idk.get(0.75, np.nan), higher_is_better=False))
if "options_yes_pct" in profile.columns:
    q_opt = profile["options_yes_pct"].quantile([0.25, 0.75]).to_dict()
    traffic["benefits_navigation"] = profile["options_yes_pct"].map(lambda v: traffic_light(v, q_opt.get(0.25, np.nan), q_opt.get(0.75, np.nan), True))

show_df(traffic, "Traffic light table (per cluster/persona)")

# %%
fig, ax = plt.subplots(figsize=(10, 2.6))
cols = [c for c in ["support_level", "safety_level", "anonymity_clarity", "benefits_navigation"] if c in traffic.columns]
map_color = {"RED": 0, "YELLOW": 1, "GREEN": 2, "NA": np.nan}
M = traffic.set_index("persona")[cols].replace(map_color).astype(float)
sns.heatmap(M, cmap="RdYlGn", vmin=0, vmax=2, cbar=False, ax=ax)
ax.set_title("Traffic light summary (personas)")
ax.set_xlabel("KPI category")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Burden overlays by persona (post-hoc)
#
# These overlays help size interventions:
# - current disorder rate/cases (self-reported),
# - treatment sought rate/cases.
#
# They are descriptive, not causal.

# %%
def _as_clean_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def _is_yes_like(x):
    if isinstance(x, (int, np.integer)) and x == 1:
        return True
    if isinstance(x, (float, np.floating)) and np.isfinite(x) and float(x) == 1.0:
        return True
    s = _as_clean_str(x)
    if s is None:
        return False
    return s in {"Yes", "1", "1.0", "True", "true"}


def _is_current_disorder_yes(x):
    s = _as_clean_str(x)
    return s == "Yes"


rows = []
for persona, g in df_seg.groupby("persona"):
    n = int(len(g))
    disorders = int(g["current_disorder"].map(_is_current_disorder_yes).sum()) if "current_disorder" in g.columns else np.nan
    treated = int(g["treatment_sought"].map(_is_yes_like).sum()) if "treatment_sought" in g.columns else np.nan
    rows.append(
        {
            "persona": persona,
            "n": n,
            "current_disorder_cases": disorders,
            "current_disorder_rate": float(disorders / n) if n > 0 else np.nan,
            "treatment_sought_cases": treated,
            "treatment_sought_rate": float(treated / n) if n > 0 else np.nan,
        }
    )

persona_mh = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
show_df(persona_mh, "Burden overlays by persona")

# %%
fig, ax = plt.subplots(figsize=(11, 4))
tmp = persona_mh.sort_values("persona")
idx = np.arange(len(tmp))
w = 0.35
ax.bar(idx - w / 2, tmp["current_disorder_cases"], width=w, label="Current disorder (cases)", color=PALETTE[0])
ax.bar(idx + w / 2, tmp["treatment_sought_cases"], width=w, label="Treatment sought (cases)", color=PALETTE[1])
ax.set_xticks(idx)
ax.set_xticklabels(tmp["persona"], rotation=20, ha="right")
ax.set_title("Burden overlay by persona (estimated cases)")
ax.set_ylabel("Cases (count)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(idx - w / 2, 100 * tmp["current_disorder_rate"], width=w, label="Current disorder (rate)", color=PALETTE[0])
ax.bar(idx + w / 2, 100 * tmp["treatment_sought_rate"], width=w, label="Treatment sought (rate)", color=PALETTE[1])
ax.set_xticks(idx)
ax.set_xticklabels(tmp["persona"], rotation=20, ha="right")
ax.set_title("Burden overlay by persona (rates)")
ax.set_ylabel("Rate (%)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 Explainability sanity check (shallow tree)
#
# We fit a shallow decision tree to predict cluster labels from interpretable levers.
# If the tree has reasonable macro-F1, it supports the interpretation that clusters correspond to simple patterns in levers
# (as opposed to being purely embedding artifacts).

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

explain_feats = [
    "benefits",
    "benefits_options_known",
    "resources_available",
    "anonymity_protected",
    "leave_ease",
    "employer_serious",
    "supervisor_comfort",
    "coworker_comfort",
    "mental_health_consequences",
    "career_harm",
    "team_views_negative",
    "remote_work",
    "company_size",
    "idx_support",
    "idx_safety",
    "qc_unknown_count",
]
explain_feats = [c for c in explain_feats if c in df_feat.columns]

mask = labels != -1
Xexp = df_feat.loc[mask, explain_feats].copy()
yexp = pd.Series(labels[mask], name="cluster")

for c in Xexp.columns:
    if not pd.api.types.is_numeric_dtype(Xexp[c].dtype):
        Xexp[c] = Xexp[c].astype("string").fillna("__MISSING__")

Xenc = pd.get_dummies(Xexp, drop_first=False)
clf = DecisionTreeClassifier(max_depth=3, random_state=0, min_samples_leaf=15)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

pred = np.zeros(len(yexp), dtype=int)
for tr, te in cv.split(Xenc, yexp):
    clf.fit(Xenc.iloc[tr], yexp.iloc[tr])
    pred[te] = clf.predict(Xenc.iloc[te])

cm = confusion_matrix(yexp, pred, labels=sorted(yexp.unique()))
cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in sorted(yexp.unique())], columns=[f"pred_{c}" for c in sorted(yexp.unique())])
f1_macro = float(f1_score(yexp, pred, average="macro"))
show_df(pd.DataFrame([{"cv_macro_f1": f1_macro, "n_inliers": int(mask.sum())}]), "Shallow-tree explainability (summary)")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Shallow tree explainability (CV confusion matrix)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5) Predictive checks (association, not causation)
#
# We quantify out-of-sample signal of HR-lever feature sets for burden proxies via cross-validated logistic regression.
# A dummy-prior baseline is included to show “how much better than predicting the prior” we are.
#
# Interpretation:
# - AUC is a ranking metric: 0.5 is random.
# - Average precision’s naive baseline is approximately the positive rate.
#
# Even a “decent” AUC does **not** establish causality; this is prioritization evidence only.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def build_preprocessor_for_predict(X):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), cat),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def cv_scores(X, y, pipe, n_splits=5, seed=0):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, aps = [], []
    for tr, te in cv.split(X, y):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        aps.append(average_precision_score(y.iloc[te], p))
    aucs = np.asarray(aucs, dtype=float)
    aps = np.asarray(aps, dtype=float)
    return {"roc_auc_mean": float(aucs.mean()), "roc_auc_std": float(aucs.std(ddof=1)), "avg_precision_mean": float(aps.mean()), "avg_precision_std": float(aps.std(ddof=1))}


def y_treatment(frame):
    s = _clean_str(frame["treatment_sought"])
    mask = ~s.eq("")
    y = s.isin(["1", "1.0", "Yes"]).astype(int)
    return y.where(mask)


def y_current_disorder(frame):
    s = _clean_str(frame["current_disorder"])
    mask = ~s.eq("")
    y = s.eq("Yes").astype(int)
    return y.where(mask)


def y_interfere_untreated_often_applicable(frame):
    s = _clean_str(frame["work_interference_untreated"])
    mask = ~s.eq("Not applicable to me")
    y = s.eq("Often").astype(int)
    return y.where(mask)


outcomes = {
    "treatment_yes": y_treatment,
    "current_disorder_yes": y_current_disorder,
    "interfere_untreated_often_applicable": y_interfere_untreated_often_applicable,
}

pred_sets = {
    "indices_only": sets["indices_only"],
    "workplace_no_roles": sets["workplace_no_roles"],
    "scored_items_no_gap": sets["scored_items_no_gap"],
}

rows = []
for out_name, fn in outcomes.items():
    y = fn(df_feat)
    mask = y.notna()
    y = y.loc[mask].astype(int)
    for set_name, cols in pred_sets.items():
        X = to_object_for_cat(df_feat.loc[mask, cols].copy())
        pre = build_preprocessor_for_predict(X)
        pos_rate = float(y.mean())

        logreg = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0))])
        dummy = Pipeline([("pre", pre), ("clf", DummyClassifier(strategy="prior"))])

        rows.append({"outcome": out_name, "feature_set": set_name, "model": "logreg", "n": int(len(X)), "pos_rate": pos_rate, **cv_scores(X, y, logreg)})
        rows.append({"outcome": out_name, "feature_set": set_name, "model": "dummy_prior", "n": int(len(X)), "pos_rate": pos_rate, **cv_scores(X, y, dummy)})

metrics = pd.DataFrame(rows).sort_values(["outcome", "feature_set", "model"]).reset_index(drop=True)
show_df(metrics, "Predictive CV metrics (logreg vs dummy)")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
tmp = metrics[metrics["model"].eq("logreg")].copy()
sns.barplot(data=tmp, x="outcome", y="roc_auc_mean", hue="feature_set", ax=ax)
ax.set_title("Predictive signal (ROC-AUC, CV mean) — logistic regression")
ax.set_xlabel("")
ax.set_ylabel("ROC-AUC")
plt.tight_layout()
plt.show()

# %%
wide_auc = metrics.pivot_table(index=["outcome", "feature_set"], columns="model", values="roc_auc_mean", aggfunc="first").reset_index()
wide_auc["auc_uplift_vs_dummy"] = wide_auc["logreg"] - wide_auc["dummy_prior"]
show_df(wide_auc.sort_values(["outcome", "auc_uplift_vs_dummy"], ascending=[True, False]), "AUC uplift vs dummy baseline")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=wide_auc, x="outcome", y="auc_uplift_vs_dummy", hue="feature_set", ax=ax)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("ΔAUC vs dummy baseline (logreg − dummy_prior)")
ax.set_xlabel("")
ax.set_ylabel("AUC uplift")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.1 “Which levers matter?” (coefficient summaries; interpretation-only)
#
# We fit logistic regression on the full dataset (for each outcome) and summarize coefficients.
# Caveats:
# - coefficients are conditional associations and can be unstable under collinearity,
# - the primary claim remains cross-validated performance above.
#
# We:
# - list top encoded coefficients,
# - then aggregate absolute coefficient mass back to the base feature level.

# %%
def fit_full_and_extract(pipe, X, y):
    pipe.fit(X, y)
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    coefs = clf.coef_.ravel()
    out = pd.DataFrame({"encoded_feature": names, "coef": coefs})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False)


def infer_base_feature(encoded_feature, base_cols):
    if encoded_feature.startswith("num__"):
        return encoded_feature.replace("num__", "")
    if encoded_feature.startswith("cat__"):
        rest = encoded_feature.replace("cat__", "")
        candidates = sorted(base_cols, key=len, reverse=True)
        for c in candidates:
            if rest == c or rest.startswith(c + "_"):
                return c
        return rest.split("_")[0]
    return encoded_feature


coef_rows = []
for out_name, fn in outcomes.items():
    y = fn(df_feat)
    mask = y.notna()
    y = y.loc[mask].astype(int)
    cols = pred_sets["workplace_no_roles"]
    X = to_object_for_cat(df_feat.loc[mask, cols].copy())
    pre = build_preprocessor_for_predict(X)
    clf = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    coef_df = fit_full_and_extract(pipe, X, y)
    coef_df["outcome"] = out_name
    coef_df["base_feature"] = coef_df["encoded_feature"].map(lambda s: infer_base_feature(s, cols))
    coef_rows.append(coef_df.head(30)[["outcome", "encoded_feature", "coef", "base_feature"]])

top_encoded_all = pd.concat(coef_rows, ignore_index=True)
show_df(top_encoded_all, "Top encoded coefficients (|coef|), interpretation-only")

# %%
base_importance = top_encoded_all.copy()
base_importance["abs_coef"] = base_importance["coef"].abs()
base_importance = base_importance.groupby(["outcome", "base_feature"])["abs_coef"].sum().sort_values(ascending=False).reset_index()
show_df(base_importance.groupby("outcome").head(15), "Base feature importance (sum of |coef| over top encoded terms)")

# %%
fig, ax = plt.subplots(figsize=(10, 4))
tmp = base_importance.groupby("outcome").head(10)
sns.barplot(data=tmp, y="base_feature", x="abs_coef", hue="outcome", ax=ax)
ax.set_title("Top base features by |coef| (interpretation-only)")
ax.set_xlabel("Sum of absolute coefficients (top encoded terms)")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusions (what is robust vs what not to overclaim)
#
# Robust:
# - Routing and structural missingness are real and must be handled; otherwise clusters can reflect eligibility rather than HR levers.
# - Workplace levers admit interpretable segments with consistent “lift” differentiators.
#
# Expected limitations:
# - Predictive signal for burden proxies is at best moderate. Burden depends on many non-workplace factors and the outcomes are noisy.
#
# Avoid overclaiming:
# - This is cross-sectional observational survey data. Cluster differences and predictive associations are not causal effects.
