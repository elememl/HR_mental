#%% [markdown]
# ### 0. Setup
# Project imports, plotting conventions, and file-system paths used throughout the analysis are specified.

#%% 00 Setup

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import hashlib
import itertools
import json
import re
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import HTML, Image, display
from matplotlib import MatplotlibDeprecationWarning
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn_extra.cluster import KMedoids
from umap import UMAP

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "patch.force_edgecolor": False,
    "patch.edgecolor": "none",
    "patch.linewidth": 0.0,
})

#%% [markdown]
# ### 1. Data Load + Column Rename
# Raw survey data are ingested, long questionnaire fields are mapped to shortened names, and a normalized column set is written.

#%% 01 Data Load + Column Rename

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "survey.csv"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "survey_renamed.csv"

df = pd.read_csv(RAW_PATH)

rename_map = {
    "Are you self-employed?": "self_employed",
    "How many employees does your company or organization have?": "company_size",
    "Is your employer primarily a tech company/organization?": "tech_company",
    "Is your primary role within your company related to tech/IT?": "tech_role",
    "Does your employer provide mental health benefits as part of healthcare coverage?": "benefits",
    "Do you know the options for mental health care available under your employer-provided coverage?": "mh_options_known",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?": "boss_mh_discuss",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?": "resources",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?": "anonymity_protected",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:": "leave_easy",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?": "bad_conseq_mh_boss",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?": "bad_conseq_ph_boss",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?": "mh_comfort_coworkers",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?": "mh_comfort_supervisor",
    "Do you feel that your employer takes mental health as seriously as physical health?": "mh_ph_boss_serious",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?": "observed_mhdcoworker_bad_conseq",
    "Do you have medical coverage (private insurance or state-provided) which includes treatment of \xa0mental health issues?": "coverage_mh",
    "Do you know local or online resources to seek help for a mental health disorder?": "know_resources_mh_help",
    "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?": "if_mhd_reveal_clients",
    "If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?": "if_mhd_reveal_clients_neg",
    "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?": "if_mhd_reveal_coworkers",
    "If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?": "if_mhd_reveal_coworkers_neg",
    "Do you believe your productivity is ever affected by a mental health issue?": "productivity_mh",
    "If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?": "percent_work_time_mh_affect",
    "Do you have previous employers?": "prev_boss",
    "Have your previous employers provided mental health benefits?": "prev_benefits",
    "Were you aware of the options for mental health care provided by your previous employers?": "prev_mh_options_known",
    "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?": "prev_boss_mh_discuss",
    "Did your previous employers provide resources to learn more about mental health issues and how to seek help?": "prev_resources",
    "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?": "prev_anonymity_protected",
    "Do you think that discussing a mental health disorder with previous employers would have negative consequences?": "bad_conseq_mh_prev_boss",
    "Do you think that discussing a physical health issue with previous employers would have negative consequences?": "bad_conseq_ph_prev_boss",
    "Would you have been willing to discuss a mental health issue with your previous co-workers?": "mh_comfort_prev_coworkers",
    "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?": "mh_comfort_prev_supervisor",
    "Did you feel that your previous employers took mental health as seriously as physical health?": "mh_ph_prev_boss_serious",
    "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?": "prev_observed_bad_conseq_mh",
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?": "ph_interview",
    "Why or why not?": "why",
    "Would you bring up a mental health issue with a potential employer in an interview?": "mh_interview",
    "Why or why not?.1": "why_1",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?": "mhd_hurt_career",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?": "coworkers_view_neg_mhd",
    "How willing would you be to share with friends and family that you have a mental illness?": "friends_family_mhd_comfort",
    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?": "ever_observed_mhd_bad_response",
    "Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?": "mhdcoworker_you_not_reveal",
    "Do you have a family history of mental illness?": "mh_family_history",
    "Have you had a mental health disorder in the past?": "mhd_past",
    "Do you currently have a mental health disorder?": "current_mhd",
    "If yes, what condition(s) have you been diagnosed with?": "mhd_diagnosed_condition",
    "If maybe, what condition(s) do you believe you have?": "mhd_believe_condition",
    "Have you been diagnosed with a mental health condition by a medical professional?": "mhd_by_med_pro",
    "If so, what condition(s) were you diagnosed with?": "med_pro_condition",
    "Have you ever sought treatment for a mental health issue from a mental health professional?": "pro_treatment",
    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?": "treat_mhd_bad_work",
    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?": "no_treat_mhd_bad_work",
    "What is your age?": "age",
    "What is your gender?": "gender",
    "What country do you live in?": "country_live",
    "What US state or territory do you live in?": "US_state",
    "What country do you work in?": "country_work",
    "What US state or territory do you work in?": "US_work",
    "Which of the following best describes your work position?": "work_position",
    "Do you work remotely?": "remote_work",
}


df_renamed = df.rename(columns=rename_map)
df_renamed.to_csv(OUT_PATH, index=False)

#%% [markdown]
# ### 2. Population Filtering
# The target population is defined and filtering criteria for analytical inclusion/exclusion are applied.

#%% 02 Cohort Definition - Helpers

def _log_change(step, before, after):
    """
    Print a single summary line: rows removed, cols removed, and shapes.
    """
    rows_removed = before.shape[0] - after.shape[0]
    cols_removed = before.shape[1] - after.shape[1]
    print(f"{step}: removed {rows_removed} rows, {cols_removed} cols "
          f"(shape {before.shape} -> {after.shape})")


def filter_not_self_employed(df):
    """
    Step 1:
    - Remove respondents who ARE self-employed (self_employed == 1)
    - Keep only self_employed == 0
    - Drop the 'self_employed' column afterward because it's constant (all remaining are 0).
   - (remove self-employed): removed 287 rows, 1 cols (shape (1433, 63) -> (1146, 62))
    """
    before = df

    # Keep only not-self-employed (0). This drops NaNs in self_employed automatically.
    df2 = df[df["self_employed"] == 0].copy()

    # Drop the defining column (everyone left is not self-employed)
    df2 = df2.drop(columns=["self_employed"], errors="ignore")

    _log_change("remove self-employed", before, df2)
    return df2


def filter_tech_role(df):
    """
    Step 2:
    - Remove respondents whose role is explicitly NOT tech-related.
    - Keep respondents who answered Yes AND respondents with missing values (NaN).
    - Remove only explicit No.

    This function supports two common encodings:
      A) numeric: 1/0 or 1.0/0.0 (with NaN)
      B) text: "Yes"/"No" (with NaN)
    After filtering, we drop 'tech_role' because it's defining the population and not informative.
    (remove explicit non-tech roles): removed 15 rows, 1 cols (shape (1146, 62) -> (1131, 61))
    """
    # The respondents who explicitly answered “no” to having a technology/IT-related role (15 rows) were removed from the dataset.
    # Among the remaining samples, 1,170 responses were missing and 248 indicated “yes.”
    # Although it cannot be confirmed that all missing responses correspond to technology-related roles, these samples were retained for simplicity and to preserve dataset size.
    # The column was subsequently removed as a feature due to its high proportion of missing values and limited variability among observed responses, which reduces its usefulness for modeling and may introduce bias if included.

    before = df
    col = df["tech_role"]

    # Case A: numeric 0/1 (or floats). Example values: NaN, 1.0, 0.0
    if pd.api.types.is_numeric_dtype(col):
        # Keep: (col == 1) and NaN. Remove: (col == 0)
        df2 = df[(col.isna()) | (col != 0)].copy()

    else:
        # Case B: text Yes/No. Normalize to compare reliably.
        # Keep: Yes + NaN. Remove: No.
        normalized = col.astype(str).str.strip().str.lower()
        df2 = df[(col.isna()) | (normalized != "no")].copy()

    # Drop the defining column
    df2 = df2.drop(columns=["tech_role"], errors="ignore")

    _log_change("remove explicit non-tech roles", before, df2)
    return df2


NA_TOKEN = "Not applicable"

_MISSING_STRINGS = {"nan", "na", "n/a", "null", "none", "<na>", "<nan>", ""}

def canonicalize_true_missing(df):
    """
    Convert textual/blank missing values to real np.nan, without touching NA_TOKEN.
    """
    df2 = df.copy()

    obj_cols = df2.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        s = df2[c].astype("string")

        # preserve structural category
        is_na_token = s == NA_TOKEN

        s2 = s.str.strip()
        missing_mask = s2.str.lower().isin(_MISSING_STRINGS)

        df2.loc[~is_na_token & missing_mask, c] = np.nan
        df2.loc[is_na_token, c] = NA_TOKEN

    return df2


def apply_driver_skip_logic(df):
    """
    Fix structural missingness (skip-logic) by filling NAs with 'Not applicable'
    when the parent condition indicates the question did not apply.
    """
    df2 = canonicalize_true_missing(df)
    # -----------------------------
    # Rule A: prev_* block depends on prev_boss
    # If prev_boss == 0 -> prev_* questions should be Not applicable
    # -----------------------------
    prev_children = [
        "prev_benefits",
        "prev_mh_options_known",
        "prev_boss_mh_discuss",
        "prev_resources",
        "prev_anonymity_protected",
        "bad_conseq_mh_prev_boss",
        "bad_conseq_ph_prev_boss",
        "mh_comfort_prev_coworkers",
        "mh_comfort_prev_supervisor",
        "mh_ph_prev_boss_serious",
        "prev_observed_bad_conseq_mh",
    ]

    if "prev_boss" in df2.columns:
        # treat 0 or 0.0 as "No previous employers"
        mask_no_prev = df2["prev_boss"].astype(str).str.strip().isin(["0", "0.0"])
        cols = [c for c in prev_children if c in df2.columns]
        for c in cols:
            df2.loc[mask_no_prev & df2[c].isna(), c] = NA_TOKEN

    # -----------------------------
    # Rule B:
    # If ever_observed_mhd_bad_response is "No" OR "nan" (string) -> child is Not applicable
    # (No other cases)
    # -----------------------------
    if "ever_observed_mhd_bad_response" in df2.columns and "mhdcoworker_you_not_reveal" in df2.columns:
        parent_str = df2["ever_observed_mhd_bad_response"].astype(str).str.strip().str.lower()
        mask_not_applicable = parent_str.isin(["no"])

        child = "mhdcoworker_you_not_reveal"
        df2.loc[mask_not_applicable & df2[child].isna(), child] = NA_TOKEN

    # -----------------------------
    # Rule C:
    # If benefits is "No" OR "Not eligible for coverage / N/A"
    # -> mh_options_known is Not applicable
    # -----------------------------
    if "benefits" in df2.columns and "mh_options_known" in df2.columns:
        parent = df2["benefits"].astype(str).str.strip().str.lower()
        mask_not_applicable = parent.isin(["no", "not eligible for coverage / n/a"])

        child = "mh_options_known"
        # Overwrite even if the respondent filled something (noncompliance / logically impossible)
        df2.loc[mask_not_applicable, child] = NA_TOKEN
    df2 = standardize_binary_drivers(df2)
    return df2

def standardize_binary_drivers(df):
    """
    Standardize known binary driver features to consistent 0/1 numeric encoding.

    - tech_company: expected values {0,1} (sometimes float); enforce int 0/1 where possible
    - prev_boss: expected values {0,1}; enforce int 0/1 where possible
    - observed_mhdcoworker_bad_conseq: expected {"Yes","No"}; map to {1,0}

    Leaves non-binary tokens (e.g., NA_TOKEN, "No response") unchanged if they appear.
    """
    df2 = df.copy()

    # 1) Numeric-coded binaries (may be float strings etc.)
    for col in ["tech_company", "prev_boss"]:
        if col in df2.columns:
            s = df2[col]

            # Coerce to numeric when possible; keep non-numeric as-is
            s_num = pd.to_numeric(s, errors="coerce")

            # Only overwrite where conversion succeeded (not NaN)
            mask = s_num.notna()
            df2.loc[mask, col] = (s_num.loc[mask].astype(float) != 0).astype(int)

    # 2) Yes/No-coded binary
    col = "observed_mhdcoworker_bad_conseq"
    if col in df2.columns:
        # Use a safe mapping that won't fail on pandas StringDtype
        s = df2[col].astype(str).str.strip().str.lower()
        mapped = s.map({"yes": 1, "no": 0})
        # Only overwrite when mapping is known; otherwise keep original
        df2[col] = df2[col].astype(object).where(mapped.isna(), mapped)
        num = pd.to_numeric(df2[col], errors="coerce")
        df2[col] = df2[col].where(num.isna(), num)

    return df2

# Columns used ONLY for interpretation after clustering
OVERLAY_COLS = [
    "respondent_id",
    "pro_treatment",
    "age",
    "gender",
    "country_live",
    "US_state",
    "country_work",
    "US_work",
    "work_position",
    "mhd_past",
    "current_mhd",
    "mhd_diagnosed_condition",
    "mhd_believe_condition",
    "mhd_by_med_pro",
    "med_pro_condition",
    "mhd_hurt_career",
    "coworkers_view_neg_mhd",
    "treat_mhd_bad_work",
    "no_treat_mhd_bad_work",
]

def show_table(df, title, max_rows=25, figsize=(12, 6), dpi=150):
    if df is None or df.empty:
        print(f"[show_table] '{title}': nothing to show (empty).")
        return

    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    table = ax.table(
        cellText=df2.values,
        colLabels=df2.columns,
        rowLabels=df2.index.astype(str),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.show()


IN_PATH = PROJECT_ROOT / "data" / "raw" / "survey_renamed.csv"


#%% 02A Cohort Definition - Apply Filters
# 1) Load renamed dataset
df = pd.read_csv(IN_PATH)

# ---------------------------------------------------------------------
# Consistency check (population filters) WITHOUT touching preprocessing.py
# Step 1: keep only not self-employed (0)
# Step 2: remove explicit non-tech roles, keep yes + missing
# ---------------------------------------------------------------------
df_step1 = df[df["self_employed"] == 0].copy()

col = df_step1["tech_role"]
if pd.api.types.is_numeric_dtype(col):
    df_step2_keep_role = df_step1[(col.isna()) | (col != 0)].copy()
else:
    norm = col.astype(str).str.strip().str.lower()
    df_step2_keep_role = df_step1[(col.isna()) | (norm != "no")].copy()

subset = df_step2_keep_role[df_step2_keep_role["tech_company"] == 0]
n_non_tech_company = (df_step2_keep_role["tech_company"] == 0).sum()
n_tech_role_yes = (subset["tech_role"] == 1).sum()
n_subset = len(subset)

print("\nCheck consistency:")
print("Rows where tech_company == 0:", n_non_tech_company)
print("Rows where tech_company == 0 AND tech_role == 1:", n_tech_role_yes)

# ---------------------------------------------------------------------
# Main cleaning pipeline (drops self_employed and tech_role)
# ---------------------------------------------------------------------
df_clean = filter_not_self_employed(df)
df_clean = filter_tech_role(df_clean)

# -----------------------------
# Add respondent ID (stable row-based)
# -----------------------------

df_clean = df_clean.reset_index(drop=True)
df_clean.insert(0, "respondent_id", range(1, len(df_clean) + 1))


#%% [markdown]
# ### 3. Data Cleaning + Missingness
# Applying cleaning and reviewing missingness patterns before constructing modeling features.

#%% 03A Data Cleaning + Missingness Summary
# ---------------------------------------------------------------------
# Row missingness: summary + histogram
# ---------------------------------------------------------------------
row_miss = df_clean.isna().mean(axis=1)

summary = (row_miss.describe()[["min", "max"]] * 100).round(1)
summary.index = ["min", "max"]
display(summary.to_frame(name="Row missingness (%)"))

#%% 03B Row Missingness Histogram
plt.figure(figsize=(8, 5), dpi=150)
plt.hist(row_miss, bins=20)
plt.xlabel("Fraction missing per row")
plt.ylabel("Number of rows")
plt.title("Row missingness distribution")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Column missingness: ONE combined top-15 table + plots
# ---------------------------------------------------------------------
n_rows = len(df_clean)
miss_count = df_clean.isna().sum()
miss_pct = (miss_count / n_rows) * 100

miss_table = (
    pd.DataFrame({"missing_count": miss_count, "missing_percent": miss_pct})
    .query("missing_count > 0")
    .sort_values("missing_percent", ascending=False)
    .rename_axis("feature")
    .reset_index()
)

top15 = miss_table.head(15).copy()
# Format % nicely for printing
top15["missing_percent"] = top15["missing_percent"].map(lambda x: f"{x:.2f}%")

#%% 03C Column Missingness Table
display(HTML(top15.to_html(index=False)))

#%% 03D Column Missingness Histogram
# Histogram of column missingness (fractions)
col_miss = df_clean.isna().mean()
plt.figure(figsize=(8, 5), dpi=150)
plt.hist(col_miss, bins=20)
plt.xlabel("Fraction missing per column")
plt.ylabel("Number of columns")
plt.title("Column missingness distribution")
plt.tight_layout()
plt.show()

#%% 03E Column Missingness Feature Bars
# Bar plot: missingness per feature (sorted) — only features with missingness
col_miss_sorted = col_miss[col_miss > 0].sort_values(ascending=False)
plt.figure(figsize=(10, 6), dpi=150)
plt.bar(col_miss_sorted.index, col_miss_sorted.values)
plt.xticks(rotation=90)
plt.ylabel("Fraction missing")
plt.title("Missingness per feature (sorted)")
plt.tight_layout()
plt.show()

# -----------------------------
# Drop columns with 100% missing
# -----------------------------

rows_before, cols_before = df_clean.shape
fully_missing_cols = df_clean.columns[df_clean.isna().all()]
df_clean = df_clean.drop(columns=fully_missing_cols)
rows_after, cols_after = df_clean.shape

removed_cols = cols_before - cols_after
print(
    f"\nRemoved {removed_cols} (100% missingness) columns: "
    f"(shape {rows_before, cols_before} -> {rows_after, cols_after})"
)

# ---------------------------------------------------------------------
# Extract overlays (now guaranteed to include respondent_id)
# ---------------------------------------------------------------------
overlay_cols = [c for c in OVERLAY_COLS if c in df_clean.columns]
df_overlays = df_clean[overlay_cols].copy()

# Force respondent_id to exist in overlays even if something changes later
if "respondent_id" not in df_overlays.columns:
    df_overlays.insert(0, "respondent_id", df_clean["respondent_id"].values)

overlay_path = PROJECT_ROOT / "data" / "out" / "overlays_clean.csv"
overlay_path.parent.mkdir(parents=True, exist_ok=True)
df_overlays.to_csv(overlay_path, index=False)

# ---------------------------------------------------------------------
# Remove overlay columns from drivers BEFORE saving drivers
# Keep respondent_id in drivers.
# ---------------------------------------------------------------------

overlay_cols_present = [c for c in OVERLAY_COLS if c in df_clean.columns and c != "respondent_id"]
df_drivers = df_clean.drop(columns=overlay_cols_present)
print(f"Dropped overlay columns from drivers (kept respondent_id): {len(overlay_cols_present)}")

# -----------------------------
# Drop open-ended text features (permanently) FROM DRIVERS
# -----------------------------
drop_cols = ["why", "why_1"]
drop_present = [c for c in drop_cols if c in df_drivers.columns]
drop_stats = []
n_rows_drivers = len(df_drivers)
for c in drop_present:
    s = df_drivers[c]
    observed = s.dropna()
    n_unique = int(observed.nunique())
    pct_missing = (float(s.isna().sum()) / n_rows_drivers * 100) if n_rows_drivers > 0 else float("nan")
    if len(observed) > 0:
        dominant_share = float(observed.value_counts(normalize=True).iloc[0] * 100)
        dominant_share_txt = f"{dominant_share:.2f}%"
    else:
        dominant_share_txt = "NA"
    pct_missing_txt = f"{pct_missing:.2f}%" if pd.notna(pct_missing) else "NA"
    drop_stats.append(
        f"{c}(dominant_share={dominant_share_txt}, n_unique={n_unique}, missing={pct_missing_txt})"
    )
df_drivers = df_drivers.drop(columns=drop_present)

stats_suffix = f" | stats: {'; '.join(drop_stats)}" if drop_stats else ""
print(f"Dropped open-ended features: {drop_present}{stats_suffix}")
print("Drivers shape now:", df_drivers.shape)

#%% [markdown]
# ### 4. Driver/Overlay Split
# Clustering driver variables are separated from interpretive overlay variables, followed by an audit of feature quality.

#%% 04A Driver/Overlay Split - Driver Audit
# ---------------------------------------------------------------------
# DRIVER AUDIT TABLE (structure + missingness)
# ---------------------------------------------------------------------
rows = []
for c in df_drivers.columns:
    if c == "respondent_id":
        continue

    s = df_drivers[c]
    n = len(s)

    n_missing = s.isna().sum()
    pct_missing = (n_missing / n) * 100
    pct_observed = 100 - pct_missing

    observed = s.dropna()
    n_unique = int(observed.nunique())

    if len(observed) > 0:
        dominant_share = (observed.value_counts().iloc[0] / len(observed)) * 100
    else:
        dominant_share = float("nan")

    rows.append({
        "feature": c,
        "% observed": round(pct_observed, 2),
        "% missing": round(pct_missing, 2),
        "n_unique": n_unique,
        "dominant_share_%": round(dominant_share, 2) if pd.notna(dominant_share) else None,
    })

audit_df = pd.DataFrame(rows).sort_values("% missing", ascending=False).reset_index(drop=True)

audit_missing = audit_df[audit_df["% missing"] > 0].reset_index(drop=True)
display(HTML(audit_missing.to_html(index=False)))

# ---------------------------------------------------------------------
# MISSINGNESS CORRELATION (pairs) - also show as table image
# ---------------------------------------------------------------------
miss = df_drivers.drop(columns=["respondent_id"], errors="ignore").isna().astype(int)
corr = miss.corr()

pairs = []
cols = list(corr.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        val = corr.iloc[i, j]
        if pd.notna(val) and abs(val) >= 0.95:
            pairs.append({"feature_1": cols[i], "feature_2": cols[j], "corr": round(val, 3)})

corr_pairs_df = pd.DataFrame(pairs).sort_values("corr", ascending=False).reset_index(drop=True)


# Contingency table for benefits and mh_options_known

# take columns
b = df_drivers["benefits"]
c = df_drivers["mh_options_known"]

# make copies as strings so we can label missing explicitly
b2 = b.copy()
c2 = c.copy()

# label true NaN clearly
b2 = b2.astype(object)
c2 = c2.astype(object)

b2[b2.isna()] = "NaN"
c2[c2.isna()] = "NaN"

# optional: clean string "nan" if exists
b2 = b2.astype(str).str.strip()
c2 = c2.astype(str).str.strip()

# build full contingency table
full_ct = pd.crosstab(
    b2,
    c2,
    dropna=False
)

#%% 04B Driver/Overlay Split - Skip-Logic Checks
display(HTML("<b>Contingency table (raw)</b>" + full_ct.to_html()))

# apply parent-child question changes
df_drivers = apply_driver_skip_logic(df_drivers)


# contigency table after applying rule C and enforcing 'non applicable'
ct_after = pd.crosstab(
    df_drivers["benefits"],
    df_drivers["mh_options_known"],
    dropna=False
)

display(HTML("<b>Contingency table (after Rule C)</b>" + ct_after.to_html()))


# ============================================================
# PROOF TABLES (post skip-logic)  <-- place AFTER apply_driver_skip_logic
# ============================================================
NA_TOKEN = "Not applicable"

# ---------- Helper to compute proof metrics ----------
def _rule_metrics(rule_name, parent_mask_false, child):
    n_false = int(parent_mask_false.sum())
    na = int((parent_mask_false & (child == NA_TOKEN)).sum())
    nan = int((parent_mask_false & child.isna()).sum())
    other = int((parent_mask_false & (~child.isna()) & (child != NA_TOKEN)).sum())

    # PASS condition: in "does-not-apply" rows -> all NA_TOKEN, no NaN, no other
    pass_fill = (n_false == 0) or (na == n_false and nan == 0 and other == 0)

    return {
        "rule": rule_name,
        "rows_where_not_applicable": n_false,
        "NA_token_count": na,
        "NaN_count": nan,
        "Other_value_count": other,
        "PASS": bool(pass_fill),
    }

proof_metrics = []
proof_failures = []

# ---------- Rule A: prev_boss -> prev_* ----------
prev_children = [
    "prev_benefits",
    "prev_mh_options_known",
    "prev_boss_mh_discuss",
    "prev_resources",
    "prev_anonymity_protected",
    "bad_conseq_mh_prev_boss",
    "bad_conseq_ph_prev_boss",
    "mh_comfort_prev_coworkers",
    "mh_comfort_prev_supervisor",
    "mh_ph_prev_boss_serious",
    "prev_observed_bad_conseq_mh",
]

if "prev_boss" in df_drivers.columns:
    p = df_drivers["prev_boss"].astype(str).str.strip()
    mask_no_prev = p.isin(["0", "0.0"])

    for c in [x for x in prev_children if x in df_drivers.columns]:
        m = _rule_metrics("prev_boss==0 -> prev_* = NA_TOKEN", mask_no_prev, df_drivers[c])
        m["child"] = c
        proof_metrics.append(m)
        if not m["PASS"]:
            proof_failures.append(m)

# ---------- Rule B: ever_observed_mhd_bad_response -> mhdcoworker_you_not_reveal ----------
# Your CURRENT rule: ONLY "No" makes child Not applicable (not "nan")
if "ever_observed_mhd_bad_response" in df_drivers.columns and "mhdcoworker_you_not_reveal" in df_drivers.columns:
    p = df_drivers["ever_observed_mhd_bad_response"].astype(str).str.strip()
    mask_not_app = p.eq("No")

    m = _rule_metrics('ever_observed_mhd_bad_response=="No" -> child = NA_TOKEN',
                      mask_not_app,
                      df_drivers["mhdcoworker_you_not_reveal"])
    m["child"] = "mhdcoworker_you_not_reveal"
    proof_metrics.append(m)
    if not m["PASS"]:
        proof_failures.append(m)

# ---------- Rule C: benefits -> mh_options_known ----------
# If benefits in {"No", "Not eligible for coverage / N/A"} -> mh_options_known = NA_TOKEN
if "benefits" in df_drivers.columns and "mh_options_known" in df_drivers.columns:
    p = df_drivers["benefits"].astype(str).str.strip()
    mask_not_app = p.isin(["No", "Not eligible for coverage / N/A"])

    m = _rule_metrics('benefits in {"No","Not eligible..."} -> mh_options_known = NA_TOKEN',
                      mask_not_app,
                      df_drivers["mh_options_known"])
    m["child"] = "mh_options_known"
    proof_metrics.append(m)
    if not m["PASS"]:
        proof_failures.append(m)

# ---------- TABLE 1A: proof metrics (always show; report-friendly) ----------
proof_metrics_df = pd.DataFrame(proof_metrics)

if proof_metrics_df.empty:
    print("\nStructural missingness proof — INFO")
    print("No skip-logic rules matched columns in df_drivers (nothing to prove).")

# ---------- TABLE 1B: failures only (only if needed) ----------
fail_df = pd.DataFrame(proof_failures)

# ---------- TABLE 2: TRUE missingness only (NaN), no redundancy ----------
drivers = df_drivers.drop(columns=["respondent_id"], errors="ignore")
n = len(drivers)

miss_nan = drivers.isna().sum()
miss_nan = miss_nan[miss_nan > 0].sort_values(ascending=False)

if miss_nan.empty:
    show_table(
        pd.DataFrame([{
            "result": "PASS",
            "message": "No TRUE NaN missingness remains in drivers after skip-logic."
        }]),
        "True missingness (NaN only) — PASS",
        max_rows=5,
        figsize=(12, 2.5)
        )
else:
    true_miss_df = pd.DataFrame({
        "feature": miss_nan.index,
        "missing_count": miss_nan.values,
        "missing_%": (miss_nan.values / n * 100).round(2),
    }).reset_index(drop=True)

    display(HTML(true_miss_df.to_html(index=False)))

    # Dependency situation (NO threshold): show top 20 |corr| among NaN indicators
    miss_bin = drivers[miss_nan.index].isna().astype(int)

    if miss_bin.shape[1] < 2:
        show_table(
            pd.DataFrame([{
                "result": "INFO",
                "message": "Only one truly-missing feature -> no dependency pairs exist."
            }]),
            "True-missingness dependencies",
            max_rows=5,
            figsize=(12, 2.5)
        )

#%% 04C Driver/Overlay Split - Feature Type Audit
# ============================================================
# Handle TRUE missingness after structural skip-logic
# ============================================================

# 1) Recode true missingness as explicit "No response"
# for sensitive categorical features

true_missing_to_no_response = [
    "ever_observed_mhd_bad_response",
    "mhdcoworker_you_not_reveal",
]

for col in true_missing_to_no_response:
    if col in df_drivers.columns:
        df_drivers[col] = df_drivers[col].fillna("No response")

# ------------------------------------------------------------
# 2) imputation for mh_options_known (very low missingness)
#    Only within applicable population (benefits = I dont know)
# ------------------------------------------------------------
# Fix remaining true missingness in mh_options_known
# (these occur only when benefits == "I don't know")

if "mh_options_known" in df_drivers.columns:
    df_drivers["mh_options_known"] = df_drivers["mh_options_known"].fillna("No")

# ============================================================
# Standardize binary drivers to strict 0/1 integers
# ============================================================

binary_cols = ["tech_company", "prev_boss", "observed_mhdcoworker_bad_conseq"]

for col in binary_cols:
    if col not in df_drivers.columns:
        continue

    # If already numeric 0/1 (possibly floats), just cast safely
    if pd.api.types.is_numeric_dtype(df_drivers[col]):
        df_drivers[col] = df_drivers[col].astype(int)

    else:
        # If text Yes/No appears, map it
        s = df_drivers[col].astype(str).str.strip().str.lower()
        mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
        df_drivers[col] = s.map(mapping)

        df_drivers[col] = df_drivers[col].astype(int)

# AFTER: confirm binaries are now int 0/1

# ============================================================
# POST-FIX VALIDATION CHECKS
# ============================================================


# feature type audit
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n=== FEATURE TYPE AUDIT ===")

audit_rows = []

for col in df_drivers.columns:
    if col == "respondent_id":
        continue

    s = df_drivers[col]
    observed = s.dropna()
    examples = s.value_counts(dropna=False).head(5).index.tolist()
    if len(observed) > 0:
        dominant_share = float(observed.value_counts(normalize=True).iloc[0] * 100)
    else:
        dominant_share = float("nan")

    audit_rows.append({
        "feature": col,
        "dominant_share_%": round(dominant_share, 2) if pd.notna(dominant_share) else None,
        "example_values": examples,
    })

audit_table = pd.DataFrame(audit_rows)

display(HTML(audit_table.to_html(index=False)))

# ---------------------------------------------------------------------
# Save drivers
# ---------------------------------------------------------------------
out_path = PROJECT_ROOT / "data" / "out" / "survey_drivers.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df_drivers.to_csv(out_path, index=False)



#%% [markdown]
# ### 5. Encoding
# Cleaned survey responses are encoded into numeric feature representations suitable for distance-based clustering.

#%% 05 Encoding

IN_PATH = PROJECT_ROOT / "data" / "out" / "survey_drivers.csv"
OUT_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"

_ = IN_PATH.exists()

# -------------------
# Helpers / constants
# -------------------
NA_TOKEN = "Not applicable"
IDK_TOKEN = "I don't know"


def norm_str(x):
    if pd.isna(x):
        return pd.NA
    return str(x).strip()


def encode_ordinal(series, mapping, name):
    """
    Encode an ordinal feature using an explicit mapping (text -> number).
    Preserves NaN. Raises if any non-NaN category is unmapped.
    Returns float to allow NaNs.
    """
    s = series.map(norm_str).astype("string")
    enc = s.map(mapping)

    return enc.astype("float")


# -------------------
# ORDINAL MAPS (single source of truth)
# -------------------
ORDINAL_MAPS = {
    "company_size": {
        "1-5": 1,
        "6-25": 2,
        "26-100": 3,
        "100-500": 4,
        "500-1000": 5,
        "More than 1000": 6,
    },
    "remote_work": {"Never": 1, "Sometimes": 2, "Always": 3},
    "ph_interview": {"No": 1, "Maybe": 2, "Yes": 3},
    "mh_interview": {"No": 1, "Maybe": 2, "Yes": 3},
    "bad_conseq_mh_boss": {"No": 1, "Maybe": 2, "Yes": 3},
    "bad_conseq_ph_boss": {"No": 1, "Maybe": 2, "Yes": 3},
    "mh_comfort_coworkers": {"No": 1, "Maybe": 2, "Yes": 3},
    "mh_comfort_supervisor": {"No": 1, "Maybe": 2, "Yes": 3},
}


def encode_company_size(series):
    """
    Robust company_size encoding:
    - If numeric-coded (1..6), validate and return.
    - Else map text bins using ORDINAL_MAPS['company_size'].
    """
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        return s_num.astype("float")

    return encode_ordinal(series, ORDINAL_MAPS["company_size"], "company_size")


# ==========================================================
# ==========================================================
def encode_mixed_ord_plus_flags(df, feature, ord_mapping, na_tokens=None, idk_tokens=None, collapse_map=None, drop_all_zero_flags=True):
    """
    Mixed (v1) = ordinal column + optional indicator columns for special categories.

    Produces:
      - <feature>_ord  (float, NaN for NA/IDK)
      - <feature>__na  (0/1) if na_tokens provided AND not all zeros (optional drop)
      - <feature>__idk (0/1) if idk_tokens provided AND not all zeros (optional drop)

    collapse_map is applied BEFORE encoding.
    """
    if feature not in df.columns:
        return pd.DataFrame(index=df.index)

    na_tokens = na_tokens or set()
    idk_tokens = idk_tokens or set()
    collapse_map = collapse_map or {}

    raw = df[feature].map(norm_str).astype("string")
    collapsed = raw.replace(collapse_map)

    out = pd.DataFrame(index=df.index)

    if na_tokens:
        col_na = f"{feature}__na"
        out[col_na] = collapsed.isin(list(na_tokens)).astype(int)
        if drop_all_zero_flags and out[col_na].sum() == 0:
            out = out.drop(columns=[col_na])

    if idk_tokens:
        col_idk = f"{feature}__idk"
        out[col_idk] = collapsed.isin(list(idk_tokens)).astype(int)
        if drop_all_zero_flags and out[col_idk].sum() == 0:
            out = out.drop(columns=[col_idk])

    # Ordinal: set NA/IDK tokens to missing, then encode
    ord_input = collapsed.copy()
    mask_special = pd.Series(False, index=df.index)

    if na_tokens:
        mask_special |= ord_input.isin(list(na_tokens))
    if idk_tokens:
        mask_special |= ord_input.isin(list(idk_tokens))

    ord_input = ord_input.mask(mask_special, pd.NA)
    out[f"{feature}_ord"] = encode_ordinal(ord_input, ord_mapping, feature)

    return out


# -------------------
# -------------------
MIXED_SPECS = {
    "friends_family_mhd_comfort": {
        "ord_mapping": {
            "Not open at all": 1,
            "Somewhat not open": 2,
            "Neutral": 3,
            "Somewhat open": 4,
            "Very open": 5,
        },
        "na_tokens": {"Not applicable to me (I do not have a mental illness)", NA_TOKEN},
        "idk_tokens": set(),
        "collapse_map": {},
    },
    "prev_observed_bad_conseq_mh": {
        "ord_mapping": {
            "None of them": 1,
            "Some of them": 2,
            "Yes, all of them": 3,
            "Yes, all of them ": 3,  # harmless whitespace variant
        },
        "na_tokens": {NA_TOKEN},
        "idk_tokens": set(),
        "collapse_map": {},
    },
    "bad_conseq_ph_prev_boss": {
        "ord_mapping": {"None of them": 1, "Some of them": 2, "Yes, all of them": 3},
        "na_tokens": {NA_TOKEN},
        "idk_tokens": set(),
        "collapse_map": {},
    },
    "prev_resources": {
        "ord_mapping": {"None did": 1, "Some did": 2, "Yes, they all did": 3},
        "na_tokens": {NA_TOKEN},
        "idk_tokens": set(),
        "collapse_map": {},
    },
    "mh_comfort_prev_coworkers": {
        "ord_mapping": {
            "No, at none of my previous employers": 1,
            "Some of my previous employers": 2,
            "Yes, at all of my previous employers": 3,
        },
        "na_tokens": {NA_TOKEN},
        "idk_tokens": set(),
        "collapse_map": {},
    },
    "leave_easy": {
        "ord_mapping": {
            "Very difficult": 1,
            "Somewhat difficult": 2,
            "Neither easy nor difficult": 3,
            "Somewhat easy": 4,
            "Very easy": 5,
        },
        "na_tokens": {NA_TOKEN, "Not applicable"},  # auto-drops if unused
        "idk_tokens": {IDK_TOKEN},                  # keep if present
        "collapse_map": {},
    },
    "prev_mh_options_known": {
        "ord_mapping": {
            "No": 1,
            "I was aware of some": 2,
            "Yes, I was aware of all of them": 3,
        },
        "na_tokens": {NA_TOKEN, "Not applicable"},
        "idk_tokens": set(),
        "collapse_map": {
            "N/A (not currently aware)": "No",
            "No, I only became aware later": "No",
        },
    },
}


# ==========================================================
# ==========================================================
def encode_mixed_ord_plus_flags_v2(
    df,
    feature,
    ord_mapping,
    na_tokens=None,
    idk_tokens=None,
    extra_flag_tokens=None,
    collapse_map=None,
    drop_all_zero_flags=True,
):
    """
    Mixed (v2) = ordinal column + multiple binary flag columns (NA / IDK / other tokens).

    Produces:
      - <feature>_ord  (float; NaN for any special token)
      - <feature>__na  (0/1) if na_tokens provided (optional drop if all zeros)
      - <feature>__idk (0/1) if idk_tokens provided (optional drop if all zeros)
      - <feature><suffix> (0/1) for each extra_flag_tokens entry (optional drop if all zeros)

    All special tokens MUST be declared (na_tokens/idk_tokens/extra_flag_tokens),
    otherwise encode_ordinal will raise on unmapped categories.
    """
    if feature not in df.columns:
        return pd.DataFrame(index=df.index)

    na_tokens = na_tokens or set()
    idk_tokens = idk_tokens or set()
    extra_flag_tokens = extra_flag_tokens or {}
    collapse_map = collapse_map or {}

    raw = df[feature].map(norm_str).astype("string")
    collapsed = raw.replace(collapse_map)

    out = pd.DataFrame(index=df.index)

    # NA flag
    if na_tokens:
        col_na = f"{feature}__na"
        out[col_na] = collapsed.isin(list(na_tokens)).astype(int)
        if drop_all_zero_flags and out[col_na].sum() == 0:
            out = out.drop(columns=[col_na])

    # IDK/uncertainty flag
    if idk_tokens:
        col_idk = f"{feature}__idk"
        out[col_idk] = collapsed.isin(list(idk_tokens)).astype(int)
        if drop_all_zero_flags and out[col_idk].sum() == 0:
            out = out.drop(columns=[col_idk])

    # Extra flags (e.g., No response)
    for token, suffix in extra_flag_tokens.items():
        col = f"{feature}{suffix}"
        out[col] = (collapsed == token).astype(int)
        if drop_all_zero_flags and out[col].sum() == 0:
            out = out.drop(columns=[col])

    # Ordinal: mask all special tokens -> NaN, then encode
    ord_input = collapsed.copy()
    mask_special = pd.Series(False, index=df.index)

    if na_tokens:
        mask_special |= ord_input.isin(list(na_tokens))
    if idk_tokens:
        mask_special |= ord_input.isin(list(idk_tokens))
    if extra_flag_tokens:
        mask_special |= ord_input.isin(list(extra_flag_tokens.keys()))

    ord_input = ord_input.mask(mask_special, pd.NA)
    out[f"{feature}_ord"] = encode_ordinal(ord_input, ord_mapping, feature)

    return out


# -------------------
# -------------------
MIXED_SPECS_V2 = {
    # ordinal severity + uncertainty flag + nonresponse flag (2 nominal flags)
    "ever_observed_mhd_bad_response": {
        "ord_mapping": {
            "No": 0,
            "Yes, I observed": 1,
            "Yes, I experienced": 2,
        },
        "na_tokens": set(),
        "idk_tokens": {"Maybe/Not sure", IDK_TOKEN},
        "extra_flag_tokens": {"No response": "__no_response"},
        "collapse_map": {},
    },

    # previous employment variables: ordinal + NA flag + IDK flag (2 nominal flags)
    "prev_anonymity_protected": {
        "ord_mapping": {"No": 0, "Sometimes": 1, "Yes, always": 2},
        "na_tokens": {NA_TOKEN},
        "idk_tokens": {IDK_TOKEN},
        "extra_flag_tokens": {},
        "collapse_map": {},
    },
    "bad_conseq_mh_prev_boss": {
        "ord_mapping": {"None of them": 0, "Some of them": 1, "Yes, all of them": 2},
        "na_tokens": {NA_TOKEN},
        "idk_tokens": {IDK_TOKEN},
        "extra_flag_tokens": {},
        "collapse_map": {},
    },
    "mh_ph_prev_boss_serious": {
        "ord_mapping": {"None did": 0, "Some did": 1, "Yes, they all did": 2},
        "na_tokens": {NA_TOKEN},
        "idk_tokens": {IDK_TOKEN},
        "extra_flag_tokens": {},
        "collapse_map": {},
    },
    "prev_boss_mh_discuss": {
        "ord_mapping": {"None did": 0, "Some did": 1, "Yes, they all did": 2},
        "na_tokens": {NA_TOKEN},
        "idk_tokens": {IDK_TOKEN},
        "extra_flag_tokens": {},
        "collapse_map": {},
    },
    "mh_comfort_prev_supervisor": {
        "ord_mapping": {
            "No, at none of my previous employers": 0,
            "Some of my previous employers": 1,
            "Yes, at all of my previous employers": 2,
        },
        "na_tokens": {NA_TOKEN},
        "idk_tokens": {IDK_TOKEN},
        "extra_flag_tokens": {},
        "collapse_map": {},
    },
    "prev_benefits": {
        # corrected category name
        "ord_mapping": {"No, none did": 0, "Some did": 1, "Yes, they all did": 2},
        "na_tokens": {NA_TOKEN},
        "idk_tokens": {IDK_TOKEN},
        "extra_flag_tokens": {},
        "collapse_map": {},
    },
}


df = pd.read_csv(IN_PATH)


# -------------------
# 1) Binary
# -------------------
binary_cols = ["tech_company", "prev_boss", "observed_mhdcoworker_bad_conseq"]
binary_cols = [c for c in binary_cols if c in df.columns]

for c in binary_cols:
    df[c] = pd.to_numeric(df[c], errors="raise")

out = pd.DataFrame({"respondent_id": df["respondent_id"]})
for c in binary_cols:
    out[c] = df[c].astype(int)


# -------------------
# 2) Nominal (one-hot)
# -------------------
nominal_cols = [
    "anonymity_protected",
    "mh_family_history",
    "mh_ph_boss_serious",
    "boss_mh_discuss",
    "resources",
    "benefits",
    "mh_options_known",
    "mhdcoworker_you_not_reveal",
]
nominal_cols = [c for c in nominal_cols if c in df.columns]


X_nom = df[nominal_cols].copy()
X_nom = X_nom.fillna("NaN")
for c in nominal_cols:
    X_nom[c] = X_nom[c].astype(str).str.strip()

nom_dum = pd.get_dummies(X_nom, prefix=nominal_cols, prefix_sep="=")


cols_before = out.shape[1]
out = out.join(nom_dum)

# -------------------
# 3) Ordinal
# -------------------
ordinal_cols = [
    "company_size",
    "remote_work",
    "ph_interview",
    "mh_interview",
    "bad_conseq_mh_boss",
    "bad_conseq_ph_boss",
    "mh_comfort_coworkers",
    "mh_comfort_supervisor",
]
ordinal_cols = [c for c in ordinal_cols if c in df.columns]


ord_out = pd.DataFrame(index=df.index)

for c in ordinal_cols:
    if c == "company_size":
        raw_vals = sorted(df["company_size"].dropna().astype(str).map(norm_str).unique().tolist())
        ord_out["company_size_ord"] = encode_company_size(df["company_size"])
    else:
        mapping = ORDINAL_MAPS[c]
        s = df[c].map(norm_str).astype("string")
        ord_out[f"{c}_ord"] = encode_ordinal(df[c], mapping, c)

cols_before = out.shape[1]
out = out.join(ord_out)


# -------------------
# 4) Mixed (v1): ordinal + (NA/IDK) flags
# -------------------

mixed_cols = [c for c in MIXED_SPECS.keys() if c in df.columns]

mixed_out = pd.DataFrame(index=df.index)

for c in mixed_cols:
    spec = MIXED_SPECS[c]
    raw = df[c].map(norm_str).astype("string")
    obs = sorted(raw.dropna().unique().tolist())

    enc_block = encode_mixed_ord_plus_flags(
        df=df,
        feature=c,
        ord_mapping=spec["ord_mapping"],
        na_tokens=spec["na_tokens"],
        idk_tokens=spec["idk_tokens"],
        collapse_map=spec["collapse_map"],
        drop_all_zero_flags=True,
    )


    mixed_out = mixed_out.join(enc_block)

cols_before = out.shape[1]
out = out.join(mixed_out)


# -------------------
# 5) Mixed (v2): ordinal + TWO nominal flags (plus optional extra flags)
# -------------------

mixed_v2_cols = [c for c in MIXED_SPECS_V2.keys() if c in df.columns]

mixed_v2_out = pd.DataFrame(index=df.index)

for c in mixed_v2_cols:
    spec = MIXED_SPECS_V2[c]
    raw = df[c].map(norm_str).astype("string")
    obs = sorted(raw.dropna().unique().tolist())

    enc_block = encode_mixed_ord_plus_flags_v2(
        df=df,
        feature=c,
        ord_mapping=spec["ord_mapping"],
        na_tokens=spec.get("na_tokens"),
        idk_tokens=spec.get("idk_tokens"),
        extra_flag_tokens=spec.get("extra_flag_tokens"),
        collapse_map=spec.get("collapse_map"),
        drop_all_zero_flags=True,
    )


    mixed_v2_out = mixed_v2_out.join(enc_block)

cols_before = out.shape[1]
out = out.join(mixed_v2_out)


# ----------------------------------------------------------
# DROP LOW-INFORMATION FLAG COLUMNS
# ----------------------------------------------------------
drop_cols = [
    "ever_observed_mhd_bad_response__no_response",
]

drop_present = [c for c in drop_cols if c in out.columns]
if drop_present:
    out = out.drop(columns=drop_present)
# ==========================================================
# DROP redundant previous-employment __na flags (explicit list)
# KEEP prev_boss as the single applicability indicator
# ==========================================================

# Explicit list of structurally identical NA flags tied to prev_boss
prev_employment_na_cols = [
    "prev_observed_bad_conseq_mh__na",
    "prev_resources__na",
    "prev_mh_options_known__na",
    "prev_anonymity_protected__na",
    "prev_boss_mh_discuss__na",
    "prev_benefits__na",
    "mh_comfort_prev_supervisor__na",
    "mh_comfort_prev_coworkers__na",
    "mh_ph_prev_boss_serious__na",
    "bad_conseq_mh_prev_boss__na",
    "bad_conseq_ph_prev_boss__na",
]

# Drop only those that actually exist (safe against refactors)
drop_cols = [c for c in prev_employment_na_cols if c in out.columns]
out = out.drop(columns=drop_cols)

# ==========================================================
# NORMALIZE BINARY FEATURES TO 0/1 INTEGERS
# (eliminate True/False everywhere)
# ==========================================================
# identify boolean columns
bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()

if bool_cols:
    out[bool_cols] = out[bool_cols].astype(int)



OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)




#%% [markdown]
# ### 6. Gower Distance
# A missingness-aware pairwise distance matrix is computed for subsequent PAM clustering.

#%% 06 Gower Distance


IN_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"
OUT_NPY = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
OUT_META = PROJECT_ROOT / "data" / "out" / "drivers_gower_meta.json"


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_gower_numeric_with_missing(X, feature_ranges, block_size=256):
    """
    Numeric-only Gower distance WITH missing values (NaN) supported.

    For each pair (i, j), compute:
        d(i,j) = mean_k( |xik - xjk| / range_k ) over features k where:
                 - range_k > 0
                 - both xik and xjk are observed (not NaN)

    If a pair has zero comparable features (should be rare), distance is set to 1.0.

    This is the correct Gower-style handling of missingness: do NOT impute, instead
    compute distances using available comparable dimensions.
    """
    n, p = X.shape

    # Only non-constant, well-defined ranges contribute
    valid_feat = np.isfinite(feature_ranges) & (feature_ranges > 0)

    Xv = X[:, valid_feat].astype(np.float64, copy=False)
    rv = feature_ranges[valid_feat].astype(np.float64, copy=False)

    n2, p2 = Xv.shape
    assert n2 == n

    D = np.zeros((n, n), dtype=np.float64)

    # Precompute finite mask for Xv to avoid repeated np.isfinite calls
    finite_mask = np.isfinite(Xv)  # (n, p2)

    for i0 in range(0, n, block_size):
        i1 = min(n, i0 + block_size)
        A = Xv[i0:i1, :]                 # (b, p2)
        A_fin = finite_mask[i0:i1, :]    # (b, p2)

        # Compute normalized absolute differences for all pairs block vs all rows
        # diff: (b, n, p2)
        diff = np.abs(A[:, None, :] - Xv[None, :, :]) / rv[None, None, :]

        # comparable mask: both finite
        comp = A_fin[:, None, :] & finite_mask[None, :, :]  # (b, n, p2)

        # zero-out non-comparable contributions
        diff = np.where(comp, diff, 0.0)

        # count comparable features per pair
        denom = comp.sum(axis=2).astype(np.float64)  # (b, n)

        # sum of contributions
        numer = diff.sum(axis=2)  # (b, n)

        # average over comparable features
        block = np.empty((i1 - i0, n), dtype=np.float64)

        # where denom > 0, compute mean; else set to 1.0 (max dissimilarity)
        good = denom > 0
        block[good] = numer[good] / denom[good]
        block[~good] = 1.0

        D[i0:i1, :] = block

    # Force symmetry and diagonal zero (important for scipy squareform/linkage)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)

    return D


force = False
block_size = 256
in_hash = sha256_file(IN_PATH)

skip_gower_compute = False
if OUT_NPY.exists() and OUT_META.exists() and not force:
    meta = json.loads(OUT_META.read_text(encoding="utf-8"))
    if meta.get("input_sha256") == in_hash:
        skip_gower_compute = True

if not skip_gower_compute:
    df = pd.read_csv(IN_PATH)
    if "respondent_id" in df.columns:
        df = df.drop(columns=["respondent_id"])

    # Require numeric dtypes (NaNs allowed)
    X = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)

    # Compute ranges ignoring NaNs (required for ordinal columns with NaN by design)
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    ranges = (col_max - col_min).astype(np.float64)

    n, p = X.shape
    print(f"Computing Gower (numeric-only, missing-aware) for n={n}, p={p}")
    print(f"Output distance matrix shape: ({n}, {n})")
    print(f"Block size: {block_size}")

    D64 = compute_gower_numeric_with_missing(X=X, feature_ranges=ranges, block_size=block_size)

    # Diagnostics
    n_nan = int(np.isnan(D64).sum())
    max_asym = float(np.max(np.abs(D64 - D64.T)))
    diag_max = float(np.max(np.abs(np.diag(D64))))
    dmin = float(np.min(D64))
    dmax = float(np.max(D64))

    print(f"Max |D-D.T| after symmetrize: {max_asym:.3e}")
    print(f"Max |diag(D)|: {diag_max:.3e}")
    print(f"Distance range: min={dmin:.6f}, max={dmax:.6f}")

    # Store compactly
    D = D64.astype(np.float32)

    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, D)

    meta = {
        "input_path": str(IN_PATH),
        "input_sha256": in_hash,
        "n_rows": int(n),
        "n_features": int(p),
        "dtype_saved": "float32",
        "dtype_computed": "float64",
        "block_size": int(block_size),
        "missingness_handling": (
            "Pairwise mean over comparable (non-NaN) features only; "
            "features with zero range excluded; if a pair has 0 comparable features, distance=1.0."
        ),
        "note": (
            "Gower-style distance computed as mean(|xi-xj|/range) over non-constant features, "
            "ignoring NaNs pairwise (no imputation). Encoded 0/1 flags + ordinals treated as numeric."
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved metadata:", OUT_META)




#%% [markdown]
# ### 7. PAM + k Selection
# k-medoids (PAM) solutions are fit across candidate k values, and silhouette and cluster-size diagnostics are reviewed.

#%% 07 PAM + k Selection

# ==========================================================
# PAM (K-MEDOIDS) ON PRECOMPUTED GOWER DISTANCES
# ==========================================================

GOWER_PATH = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Primary candidate k values + secondary checks
K_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
RANDOM_STATE = 42


def validate_distance_matrix(D):
    # Gower is typically in [0, 1], but do not hard-fail if scaling differs slightly.
    if np.max(D) > 1.0 + 1e-6:
        print(f"WARNING: max(D)={np.max(D):.6f} > 1.0. If you expected Gower, verify scaling.")


def run_pam(D, k):
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

def _plot_silhouette_vs_k(df, out_path):
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


def _plot_cluster_sizes_by_k(df, out_path):
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


D = np.load(GOWER_PATH)
validate_distance_matrix(D)

n = D.shape[0]
print(
    f"Gower matrix shape={D.shape}; stats min={np.min(D):.6f}, max={np.max(D):.6f}, mean={np.mean(D):.6f}"
)

rows = []
for k in K_LIST:
    res = run_pam(D, k)

    # Save labels per k
    np.save(OUT_DIR / f"pam_labels_k{k}.npy", res["labels"])

    rows.append({kk: vv for kk, vv in res.items() if kk not in ("labels",)})

# Summary table (in-memory only; no CSV output)
df = pd.DataFrame(rows).sort_values("k")
df_table = df[["k", "silhouette_avg", "silhouette_q25", "silhouette_q50", "silhouette_q75", "cluster_sizes"]].copy()
df_table = df_table.rename(columns={
    "silhouette_avg": "sil_avg",
    "silhouette_q25": "sil_q25",
    "silhouette_q50": "sil_q50",
    "silhouette_q75": "sil_q75",
})
df_table[["sil_avg", "sil_q25", "sil_q50", "sil_q75"]] = df_table[["sil_avg", "sil_q25", "sil_q50", "sil_q75"]].round(4)

#%% 07A PAM Summary Table
display(HTML(df_table.to_html(index=False)))

# ----------------------------
# ----------------------------
sil_plot_path = OUT_DIR / "silhouette_vs_k.png"
_plot_silhouette_vs_k(df, sil_plot_path)

#%% 07B Silhouette Plot
display(Image(filename=str(sil_plot_path)))

size_plot_path = OUT_DIR / "cluster_sizes_by_k.png"
_plot_cluster_sizes_by_k(df, size_plot_path)

#%% 07C Cluster Size Plot
display(Image(filename=str(size_plot_path)))




#%% [markdown]
# ### 8. Embeddings (PCoA/UMAP)
# The clustered distance structure is summarized using low-dimensional embeddings for visualization.

#%% 08A Embeddings - PCoA



PAM_DIR = PROJECT_ROOT / "data" / "out" / "pam"
OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam_post"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOWER_PATH = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"

K = 6
LABELS_PATH = PAM_DIR / f"pam_labels_k{K}.npy"


def pcoa(D, n_components=3):
    """
    Principal Coordinates Analysis (PCoA) / Classical MDS.

    Steps:
      1) Square distances: D^2
      2) Double-center: B = -0.5 * J D^2 J, where J = I - 11^T/n
      3) Eigendecompose B
      4) Coordinates = V * sqrt(Lambda) for positive eigenvalues only

    Returns:
      coords = min(n_components, #positive eigenvalues)
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

    m = min(n_components, evals_pos.size)
    evals_pos = evals_pos[:m]
    evecs_pos = evecs_pos[:, :m]

    coords = evecs_pos * np.sqrt(evals_pos)

    explained = evals_pos / np.sum(evals_pos) if np.sum(evals_pos) > 0 else np.zeros_like(evals_pos)
    return coords, explained


def plot_pcoa_2d(coords, explained, labels, save_path):
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


def plot_pcoa_3d(coords, explained, labels, save_path):
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
    ax.legend(
        markerscale=1.5,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    # Increase right margin and avoid tight bbox cropping
    fig.subplots_adjust(left=0.02, right=0.80, top=0.92, bottom=0.06)
    fig.savefig(save_path)
    plt.close(fig)


D = np.load(GOWER_PATH).astype(float)
D = 0.5 * (D + D.T)
np.fill_diagonal(D, 0.0)
labels = np.load(LABELS_PATH).astype(int)

# Need at least 2D for the 2D plot; ask for 3, but handle if only 2 available
coords, explained = pcoa(D, n_components=3)

# 2D
out2d = OUT_DIR / "pcoa_gower_k6_2d.png"
plot_pcoa_2d(coords[:, :2], explained[:2], labels, out2d)

#%% 08A1 PCoA 2D
display(Image(filename=str(out2d)))

#%% 08A2 PCoA 3D
# 3D
if coords.shape[1] >= 3:
    out3d = OUT_DIR / "pcoa_gower_k6_3d.png"
    plot_pcoa_3d(coords[:, :3], explained[:3], labels, out3d)
    display(Image(filename=str(out3d)))
else:
    print("PCoA returned only 2 positive components; 3D plot skipped.")


#%% 08B Embeddings - UMAP



UMAP_PAM_DIR = PROJECT_ROOT / "data" / "out" / "pam"
UMAP_OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam_post"
UMAP_OUT_DIR.mkdir(parents=True, exist_ok=True)

UMAP_GOWER_PATH = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
UMAP_LABELS_PATH = UMAP_PAM_DIR / "pam_labels_k6.npy"


def fit_umap_precomputed_gower(D, n_components=2, n_neighbors=6, min_dist=0.5, random_state=42, spread=2):
    reducer = UMAP(
        n_components=n_components,
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        spread=spread,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*using precomputed metric; inverse_transform will be unavailable.*",
            category=UserWarning,
            module=r"umap\.umap_",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*n_jobs value 1 overridden to 1 by setting random_state.*",
            category=UserWarning,
            module=r"umap\.umap_",
        )
        emb = reducer.fit_transform(D)
    return emb


def plot_umap_2d(embedding, labels, save_path):
    clusts = np.unique(labels)

    plt.figure(figsize=(10, 7), dpi=150)
    for c in clusts:
        mask = labels == c
        plt.scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.8, label=f"Cluster {c}")

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP (precomputed Gower) – k=6 clusters (2D)")
    plt.legend(markerscale=1.5, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_umap_3d(embedding, labels, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    clusts = np.unique(labels)
    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    for c in clusts:
        mask = labels == c
        ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], s=10, alpha=0.8, label=f"Cluster {c}")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    ax.set_title("UMAP (precomputed Gower) – k=6 clusters (3D)")
    ax.legend(
        markerscale=1.5,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.02, right=0.80, top=0.92, bottom=0.06)
    plt.savefig(save_path)
    plt.close(fig)


UMAP_D = np.load(UMAP_GOWER_PATH).astype(float)
UMAP_LABELS = np.load(UMAP_LABELS_PATH).astype(int)
UMAP_EMBEDDING_2D = fit_umap_precomputed_gower(D=UMAP_D, n_components=2)
UMAP_EMBEDDING_3D = fit_umap_precomputed_gower(D=UMAP_D, n_components=3)

umap_out2d = UMAP_OUT_DIR / "umap_gower_k6_2d.png"
umap_out3d = UMAP_OUT_DIR / "umap_gower_k6_3d.png"
plot_umap_2d(UMAP_EMBEDDING_2D, UMAP_LABELS, umap_out2d)
plot_umap_3d(UMAP_EMBEDDING_3D, UMAP_LABELS, umap_out3d)
display(Image(filename=str(umap_out2d)))
display(Image(filename=str(umap_out3d)))

#%% [markdown]
# ### 9. Stability Validation
# Robustness across neighboring k values is assessed using ARI, overlap statistics, and Sankey-style flow summaries.

#%% 09 Stability Validation





# ==========================================================
# POST-CLUSTER VALIDATION (STABILITY):
# - ARI table (printed)
# - Adjacent-transition stability metrics table (single)
# - Sankey diagrams: adjacent + global (k4→k5→k6→k7)
# ==========================================================

PAM_DIR = PROJECT_ROOT / "data" / "out" / "pam"
OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam_post"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS = [4, 5, 6, 7]

# For split detection:
SPLIT_THRESHOLD = 0.80

# If you want row-wise metrics weighted by row sizes:
WEIGHTED = True

def contingency_matrix(labels_from, labels_to):
    """
    Counts = clusters at k_from, cols=clusters at k_to.
    """
    return pd.crosstab(
        pd.Series(labels_from, name="from"),
        pd.Series(labels_to, name="to"),
        normalize=False,
        dropna=False,
    )


def stability_metrics_for_transition(k_from, k_to, labels_from, labels_to, split_threshold=0.80, weighted=True):
    ct = contingency_matrix(labels_from, labels_to)
    row_sums = ct.sum(axis=1).replace(0, np.nan)
    row_pct = ct.div(row_sums, axis=0).astype(float).fillna(0.0)

    # per-row: max overlap, entropy, effective #targets (perplexity)
    max_row = row_pct.max(axis=1).to_numpy(dtype=float)

    ent_values = []
    for i in row_pct.index:
        probs = row_pct.loc[i].to_numpy(dtype=float)
        probs = probs[probs > 0]
        if probs.size == 0:
            ent_values.append(0.0)
        else:
            ent_values.append(float(-(probs * np.log(probs)).sum()))
    ent = np.array(ent_values, dtype=float)
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
def make_sankey_global(ks, label_map, out_html):
    """
    Global Sankey with layers: k4 -> k5 -> k6 -> k7 (adjacent links only).
    Nodes labeled "k4_c0", etc.
    """
    # Build node labels in layer order
    node_labels = []
    node_index = {}

    def add_node(name):
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

    # Always show an inline interactive version in notebook output
    display(HTML(fig.to_html(include_plotlyjs='cdn')))


warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# -------------------------
# Load labels
# -------------------------
label_map = {k: np.load(PAM_DIR / f"pam_labels_k{k}.npy").astype(int) for k in KS}
n = len(next(iter(label_map.values())))

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
#%% 09A ARI Table
display(HTML(ari_df.to_html(index=False)))

# Optional: keep a single CSV artifact (not many files)
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
show = stability_df.copy()
for c in ["avg_max_row_overlap", "median_max_row_overlap", "median_row_entropy", "effective_num_targets_median"]:
    show[c] = show[c].round(3)
show["pct_clusters_split"] = show["pct_clusters_split"].round(1)
#%% 09B Stability Table
display(HTML(show.to_html(index=False)))


# -------------------------
# Global Sankey only
# -------------------------
global_html = OUT_DIR / "sankey_global_k4_k5_k6_k7.html"
#%% 09C Sankey
make_sankey_global(KS, label_map, global_html)


#%% [markdown]
# ### 10. Cluster Profiles
# Cluster-level patterns and pairwise separations are examined for the selected k=6 solution.

#%% 10A Cluster Profiles - K6 Inspect
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


FWD_MAP = {}
for feat, mapping in ORDINAL_MAPS.items():
    FWD_MAP[f"{feat}_ord"] = {str(k): float(v) for k, v in mapping.items()}
for feat, spec in MIXED_SPECS.items():
    FWD_MAP[f"{feat}_ord"] = {str(k): float(v) for k, v in spec["ord_mapping"].items()}
for feat, spec in MIXED_SPECS_V2.items():
    FWD_MAP[f"{feat}_ord"] = {str(k): float(v) for k, v in spec["ord_mapping"].items()}


def _get_code_for_label(feature_col, label):
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


# ============================================================
# ============================================================

def plot_heatmap(data, title, path, vmax=None, max_rows=30):

    if max_rows is not None and data.shape[0] > max_rows:
        data = data.loc[data.max(axis=1).sort_values(ascending=False).head(max_rows).index]

    plt.figure(figsize=(10, max(6, len(data) * 0.35)))
    sns.heatmap(data, cmap="viridis", vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _build_raw_pairwise_diff_matrix(df_raw, labels_aligned, drop_missing=False):
    """
    Build pairwise absolute differences of answer proportions for raw features.
    Rows = answer, Columns: C{i}-C{j} for all cluster pairs.
    """
    features = [c for c in df_raw.columns if c != "respondent_id"]
    clusters = sorted(np.unique(labels_aligned).tolist())
    pairs = list(itertools.combinations(clusters, 2))

    # normalize to string with NA tokens
    df_norm = df_raw[features].copy()
    for col in features:
        s = df_norm[col].astype(object)
        s[pd.isna(s)] = "NA"
        df_norm[col] = s.astype(str).str.strip().replace({"": "NA"})

    rows = []
    for feat in features:
        s_all = df_norm[feat]
        if drop_missing:
            mask = ~s_all.isin(["NA", "Not applicable"])
            s_all = s_all[mask]
            labels_use = labels_aligned[mask.to_numpy()]
        else:
            labels_use = labels_aligned
        cats = s_all.value_counts(dropna=False).index.tolist()

        for cat in cats:
            row = {"feature": f"{feat}={cat}"}
            for c1, c2 in pairs:
                s1 = s_all[labels_use == c1]
                s2 = s_all[labels_use == c2]
                denom1 = float(len(s1)) if len(s1) else 1.0
                denom2 = float(len(s2)) if len(s2) else 1.0
                p1 = float((s1 == cat).sum()) / denom1
                p2 = float((s2 == cat).sum()) / denom2
                row[f"C{c1}-C{c2}"] = abs(p1 - p2)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("feature")
    return df


# ============================================================



df = pd.read_csv(ENCODED_PATH)
labels = np.load(LABELS_PATH).astype(int)

feature = next((c for c in FEATURE_CANDIDATES if c in df.columns), FEATURE_CANDIDATES[0])

values = df[feature].values[labels == CLUSTER_ID]

code_some = _get_code_for_label(feature, "Somewhat open") or 4
code_very = _get_code_for_label(feature, "Very open") or 5

pct_some, cnt_some, total = _pct_and_count_over_full(values, code_some)
pct_very, cnt_very, _ = _pct_and_count_over_full(values, code_very)




#%% 10B Cluster Profiles - Detailed


# Import ordinal maps directly from encoding.py

PAM_DIR = PROJECT_ROOT / "data" / "out" / "pam"
ENCODED_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"
DRIVERS_RAW_PATH = PROJECT_ROOT / "data" / "out" / "survey_drivers.csv"  # before encoding, after preprocessing

K = 6
LABELS_PATH = PAM_DIR / f"pam_labels_k{K}.npy"

RAW_PLOTS_DIR = PROJECT_ROOT / "data" / "out" / "pam_post" / "raw_driver_distributions_k6"
RAW_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def align_labels_to_raw(df_raw, labels):
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
            return mapped.astype(int)

    # fallback: positional alignment
    return labels.astype(int)


# ------------------------------------------------------------
# Build reverse ordinal mapping
# ------------------------------------------------------------

def build_reverse_ordinal_map():
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


def ordinal_label(feat, value):
    if not np.isfinite(value):
        return "NA"
    mapping = REVERSE_ORDINAL_MAP.get(feat, {})
    return mapping.get(float(value), str(value))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def fmt_pct_or_na(x):
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x*100:.1f}%"


def safe_median_and_pct_at_median(values):
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


def pct_at_value_over_full(values, value):
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
# ------------------------------------------------------------



def plot_raw_feature_panel(df_raw, labels_aligned, features, out_path, suptitle, ncols=3, color_map=None, recode_map=None, legend_mode="shared", legend_order=None):
    n = len(features)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.6 * nrows), dpi=150)
    axes = np.asarray(axes).ravel()

    clusters = sorted(np.unique(labels_aligned).tolist())
    x = np.arange(len(clusters))

    # Build a global category list across all features to ensure legend completeness (shared legend mode).
    all_categories = []
    if legend_mode == "shared":
        for feat in features:
            series = df_raw[feat].copy()
            if recode_map is not None and feat in recode_map:
                series = recode_map[feat](series)
            series = series.astype(object)
            series[pd.isna(series)] = "NA"
            series = series.astype(str).str.strip().replace({"": "NA"})
            cats = series.value_counts(dropna=False).index.tolist()
            for c in cats:
                if c not in all_categories:
                    all_categories.append(c)

    # Assign colors deterministically for any categories not explicitly mapped.
    color_for_cat = {}
    if color_map:
        color_for_cat.update(color_map)
    palette = plt.get_cmap("tab20").colors
    idx = 0
    seed_cats = all_categories if all_categories else []
    for c in seed_cats:
        if c in color_for_cat:
            continue
        color_for_cat[c] = palette[idx % len(palette)]
        idx += 1

    for i, feat in enumerate(features):
        ax = axes[i]
        series = df_raw[feat].copy()
        if recode_map is not None and feat in recode_map:
            series = recode_map[feat](series)
        series = series.astype(object)
        series[pd.isna(series)] = "NA"
        series = series.astype(str).str.strip().replace({"": "NA"})

        overall_counts = series.value_counts(dropna=False)
        categories = overall_counts.index.tolist()

        mat = []
        for c in clusters:
            s_c = series[labels_aligned == c]
            vc = s_c.value_counts(dropna=False)
            total = len(s_c)
            props = [(float(vc.get(cat, 0)) / total) if total > 0 else 0.0 for cat in categories]
            mat.append(props)
        mat = np.array(mat, dtype=float)

        bottom = np.zeros(len(clusters), dtype=float)
        for j, cat in enumerate(categories):
            color = color_for_cat.get(cat) if color_for_cat else None
            ax.bar(
                x,
                mat[:, j],
                bottom=bottom,
                label=str(cat),
                color=color,
                edgecolor="none",
                linewidth=0.0,
                antialiased=False,
            )
            bottom += mat[:, j]

        ax.set_xticks(x, [str(c) for c in clusters])
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("")
        ax.set_title(feat)
        if legend_mode == "per_subplot":
            ax.legend(title="Answer", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, y=0.99, fontsize=14)
    fig.text(0.02, 0.5, "Proportion within cluster", va="center", rotation="vertical")
    fig.supxlabel("Cluster", y=0.01)

    legend_handles = [mpatches.Patch(color=color_for_cat[c], label=c) for c in all_categories]
    if legend_mode == "shared":
        if legend_order:
            ordered = [c for c in legend_order if c in all_categories]
            extras = [c for c in all_categories if c not in ordered]
            legend_cats = ordered + extras
        else:
            legend_cats = all_categories
        legend_handles = [mpatches.Patch(color=color_for_cat[c], label=c) for c in legend_cats]
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                title="Answer",
                loc="upper left",
                bbox_to_anchor=(0.885, 0.98),
                fontsize=8,
            )
        plt.tight_layout(rect=[0.04, 0.03, 0.86, 0.96])
    else:
        plt.tight_layout(rect=[0.04, 0.03, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)




def make_all_raw_feature_plots(df_raw, labels_aligned):
    """
    Creates one PNG per raw feature in df_raw (excluding respondent_id).
    """

    # Weak separators panel (restore)
    panel_features = [
        "remote_work",
        "tech_company",
        "mh_interview",
        "company_size",
        "prev_boss",
        "ph_interview",
        "bad_conseq_ph_prev_boss",
        "observed_mhdcoworker_bad_conseq",
        "bad_conseq_ph_boss",
    ]
    available = [f for f in panel_features if f in df_raw.columns]
    if len(available) == len(panel_features):
        panel_path = RAW_PLOTS_DIR / "raw_dist_k6_panel_weak_separators.png"
        global_color_map = {
            "Yes": "#2ca02c",
            "No": "#d62728",
            "Maybe": "#1f77b4",
            "1": "#2ca02c",
            "0": "#d62728",
        }
        plot_raw_feature_panel(
            df_raw=df_raw,
            labels_aligned=labels_aligned,
            features=available,
            out_path=panel_path,
            suptitle="Feature distributions across 6 clusters (weak separators — mostly uniform distributions)",
            ncols=3,
            color_map=global_color_map,
            legend_mode="per_subplot",
        )
    else:
        missing = [f for f in panel_features if f not in df_raw.columns]
        print(f"Skipping raw feature panel (missing columns: {missing})")

    # Strong separators panel
    panel_features = [
        "benefits",
        "mh_ph_boss_serious",
        "leave_easy",
        "resources",
        "bad_conseq_mh_boss",
        "mh_comfort_coworkers",
        "mh_options_known",
        "anonymity_protected",
        "mh_comfort_supervisor",
    ]
    available = [f for f in panel_features if f in df_raw.columns]
    if len(available) == len(panel_features):
        panel_path = RAW_PLOTS_DIR / "raw_dist_k6_panel_strong_separators.png"
        global_color_map = {
            "Yes": "#2ca02c",
            "No": "#d62728",
            "Maybe": "#1f77b4",
            "I don't know": "#4aa3df",
            "I am not sure": "#08306b",
            "Neither easy nor difficult": "#9467bd",
            "easy": "#2ca02c",
            "difficult": "#d62728",
            "No or not eligible for coverage": "#d62728",
            "Not applicable": "#c7c7c7",
        }

        def recode_benefits(s):
            s = s.astype(object)
            s = s.where(~pd.isna(s), other="NA")
            s = s.astype(str).str.strip()
            s = s.replace({"": "NA"})
            return s.replace(
                {"Not eligible for coverage / N/A": "No or not eligible for coverage", "No": "No or not eligible for coverage"}
            )

        def recode_leave_easy(s):
            s = s.astype(object)
            s = s.where(~pd.isna(s), other="NA")
            s = s.astype(str).str.strip()
            s = s.replace({"": "NA"})
            return s.replace(
                {
                    "Very difficult": "difficult",
                    "Somewhat difficult": "difficult",
                    "Very easy": "easy",
                    "Somewhat easy": "easy",
                }
            )

        recode_map = {
            "benefits": recode_benefits,
            "leave_easy": recode_leave_easy,
        }
        plot_raw_feature_panel(
            df_raw=df_raw,
            labels_aligned=labels_aligned,
            features=available,
            out_path=panel_path,
            suptitle="Feature distributions across 6 clusters (strong separators)",
            ncols=3,
            color_map=global_color_map,
            recode_map=recode_map,
            legend_mode="shared",
        )
    else:
        missing = [f for f in panel_features if f not in df_raw.columns]
        print(f"Skipping raw feature panel (missing columns: {missing})")

    # Previous employment panel (normalized answers + shared legend)
    panel_features = [
        "prev_benefits",
        "mh_ph_prev_boss_serious",
        "mh_comfort_prev_coworkers",
        "prev_resources",
        "bad_conseq_mh_prev_boss",
        "mh_comfort_prev_supervisor",
        "prev_mh_options_known",
        "prev_boss_mh_discuss",
        "prev_anonymity_protected",
    ]
    available = [f for f in panel_features if f in df_raw.columns]
    if len(available) == len(panel_features):
        panel_path = RAW_PLOTS_DIR / "raw_dist_k6_panel_prev_employment.png"
        global_color_map = {
            "Yes, all": "#2ca02c",
            "Some": "#ff7f0e",
            "No": "#d62728",
            "Only became aware later": "#9467bd",
            "I don't know": "#1f77b4",
            "Not applicable": "#7f7f7f",
        }

        def recode_prev_answers(s):
            s = s.astype(object)
            s = s.where(~pd.isna(s), other="Not applicable")
            s = s.astype(str).str.strip()
            s = s.replace({"": "Not applicable"})
            return s.replace(
                {
                    "No, none did": "No",
                    "None of them": "No",
                    "No, at none of my previous employers": "No",
                    "No": "No",
                    "N/A (not currently aware)": "No",
                    "None did": "No",
                    "Yes, they all did": "Yes, all",
                    "Yes, I was aware of all of them": "Yes, all",
                    "Yes, all of them": "Yes, all",
                    "Yes, at all of my previous employers": "Yes, all",
                    "Yes, always": "Yes, all",
                    "yes, always": "Yes, all",
                    "Some did": "Some",
                    "I was aware of some": "Some",
                    "Some of them": "Some",
                    "Some of my previous employers": "Some",
                    "Sometimes": "Some",
                    "Not applicable": "Not applicable",
                    "I don't know": "I don't know",
                    "No, I only became aware later": "Only became aware later",
                }
            )

        recode_map = {feat: recode_prev_answers for feat in panel_features}
        legend_order = ["Yes, all", "Some", "No", "Only became aware later", "I don't know", "Not applicable"]
        plot_raw_feature_panel(
            df_raw=df_raw,
            labels_aligned=labels_aligned,
            features=available,
            out_path=panel_path,
            suptitle="Previous employment feature distributions across 6 clusters",
            ncols=3,
            color_map=global_color_map,
            recode_map=recode_map,
            legend_mode="shared",
            legend_order=legend_order,
        )
    else:
        missing = [f for f in panel_features if f not in df_raw.columns]
        print(f"Skipping previous employment panel (missing columns: {missing})")

    # Stigma-related indicators panel (per-subplot legends)
    panel_features = [
        "prev_observed_bad_conseq_mh",
        "mh_family_history",
        "mhdcoworker_you_not_reveal",
        "friends_family_mhd_comfort",
        "ever_observed_mhd_bad_response",
        "boss_mh_discuss",
    ]
    available = [f for f in panel_features if f in df_raw.columns]
    if len(available) == len(panel_features):
        panel_path = RAW_PLOTS_DIR / "raw_dist_k6_panel_stigma_indicators.png"
        global_color_map = {
            "Yes": "#2ca02c",
            "No": "#d62728",
            "I don't know": "#1f77b4",
            "Not applicable": "#7f7f7f",
            "Maybe": "#0b3c8c",
            "Maybe/Not sure": "#0b3c8c",
            "No response": "#9467bd",
            "open": "#4daf4a",
            "not open": "#b2182b",
            "Neutral": "#f1c40f",
            "Yes, I observed": "#ff7f0e",
            "Yes, I experienced": "#8c564b",
            "None of them": "#2ca02c",
            "Some of them": "#ff7f0e",
            "Yes, all of them": "#d62728",
        }

        def recode_stigma(s):
            s = s.astype(object)
            s = s.where(~pd.isna(s), other="Not applicable")
            s = s.astype(str).str.strip()
            s = s.replace({"": "Not applicable"})
            s = s.replace({"Not applicable to me (I do not have a mental illness)": "Not applicable"})
            s = s.replace(
                {
                    "Somewhat open": "open",
                    "Very open": "open",
                    "Somewhat not open": "not open",
                    "Not open at all": "not open",
                }
            )
            return s

        recode_map = {feat: recode_stigma for feat in panel_features}
        plot_raw_feature_panel(
            df_raw=df_raw,
            labels_aligned=labels_aligned,
            features=available,
            out_path=panel_path,
            suptitle="Distribution of stigma-related indicators across 6 clusters",
            ncols=3,
            color_map=global_color_map,
            recode_map=recode_map,
            legend_mode="per_subplot",
        )
    else:
        missing = [f for f in panel_features if f not in df_raw.columns]
        print(f"Skipping stigma-related panel (missing columns: {missing})")


def build_raw_cluster_vs_rest_tables(df_raw, labels_aligned, top_n=15, exclude_answer_substrings=None):
    """
    For each cluster, compute top-N raw feature-answer categories by absolute
    deviation in proportion vs the rest of the sample.
    Returns: {cluster_id: DataFrame}
    """
    features = [c for c in df_raw.columns if c != "respondent_id"]
    clusters = sorted(np.unique(labels_aligned).tolist())
    results = {}

    # Pre-normalize to strings and handle missing
    df_norm = df_raw[features].copy()
    for col in features:
        s = df_norm[col].astype(object)
        s[pd.isna(s)] = "NA"
        df_norm[col] = s.astype(str).str.strip().replace({"": "NA"})

    for c in clusters:
        in_cluster = labels_aligned == c
        rows = []
        for feat in features:
            s_all = df_norm[feat]
            cats = s_all.value_counts(dropna=False).index.tolist()

            s_c = s_all[in_cluster]
            s_r = s_all[~in_cluster]
            vc_c = s_c.value_counts(dropna=False)
            vc_r = s_r.value_counts(dropna=False)
            denom_c = float(len(s_c)) if len(s_c) else 1.0
            denom_r = float(len(s_r)) if len(s_r) else 1.0

            for cat in cats:
                p_c = float(vc_c.get(cat, 0)) / denom_c
                p_r = float(vc_r.get(cat, 0)) / denom_r
                rows.append(
                    {
                        "feature": f"{feat}={cat}",
                        "cluster_%": f"{p_c*100:.1f}%",
                        "rest_%": f"{p_r*100:.1f}%",
                        "_diff": abs(p_c - p_r),
                    }
                )

        df_all = pd.DataFrame(rows).sort_values("_diff", ascending=False)
        if exclude_answer_substrings and int(c) in exclude_answer_substrings:
            subs = [s.strip().lower() for s in exclude_answer_substrings[int(c)]]
            def _keep_feature(x):
                lx = x.lower()
                return not any(sub in lx for sub in subs)
            df_all = df_all[df_all["feature"].map(_keep_feature)]
        df = df_all.head(top_n).drop(columns="_diff")
        results[int(c)] = df.reset_index(drop=True)

    return results


def build_raw_cluster_vs_rest_selected(df_raw, labels_aligned, cluster_id, selected_features):
    features = [f for f in df_raw.columns if f != "respondent_id"]
    selected_lc = {x.strip().lower() for x in selected_features}

    df_norm = df_raw[features].copy()
    for col in features:
        s = df_norm[col].astype(object)
        s[pd.isna(s)] = "NA"
        df_norm[col] = s.astype(str).str.strip().replace({"": "NA"})

    in_cluster = labels_aligned == cluster_id
    rows = []
    for feat in features:
        s_all = df_norm[feat]
        cats = s_all.value_counts(dropna=False).index.tolist()

        s_c = s_all[in_cluster]
        s_r = s_all[~in_cluster]
        vc_c = s_c.value_counts(dropna=False)
        vc_r = s_r.value_counts(dropna=False)
        denom_c = float(len(s_c)) if len(s_c) else 1.0
        denom_r = float(len(s_r)) if len(s_r) else 1.0

        for cat in cats:
            key = f"{feat}={cat}"
            if key.strip().lower() not in selected_lc:
                continue
            p_c = float(vc_c.get(cat, 0)) / denom_c
            p_r = float(vc_r.get(cat, 0)) / denom_r
            rows.append(
                {
                    "feature": key,
                    "cluster_%": f"{p_c*100:.1f}%",
                    "rest_%": f"{p_r*100:.1f}%",
                }
            )

    return pd.DataFrame(rows)


def build_raw_pairwise_tables(df_raw, labels_aligned, pairs, top_n=15):
    """
    For each cluster pair, compute top-N raw feature-answer categories by absolute
    deviation in proportion between the two clusters.
    Returns: {(c1, c2): DataFrame}
    """
    features = [c for c in df_raw.columns if c != "respondent_id"]
    results = {}

    df_norm = df_raw[features].copy()
    for col in features:
        s = df_norm[col].astype(object)
        s[pd.isna(s)] = "NA"
        df_norm[col] = s.astype(str).str.strip().replace({"": "NA"})

    for c1, c2 in pairs:
        mask1 = labels_aligned == c1
        mask2 = labels_aligned == c2
        rows = []
        for feat in features:
            s_all = df_norm[feat]
            cats = s_all.value_counts(dropna=False).index.tolist()

            s_1 = s_all[mask1]
            s_2 = s_all[mask2]
            vc1 = s_1.value_counts(dropna=False)
            vc2 = s_2.value_counts(dropna=False)
            denom1 = float(len(s_1)) if len(s_1) else 1.0
            denom2 = float(len(s_2)) if len(s_2) else 1.0

            for cat in cats:
                p1 = float(vc1.get(cat, 0)) / denom1
                p2 = float(vc2.get(cat, 0)) / denom2
                rows.append(
                    {
                        "feature": f"{feat}={cat}",
                        f"cluster_{c1}_%": f"{p1*100:.1f}%",
                        f"cluster_{c2}_%": f"{p2*100:.1f}%",
                        "_diff": abs(p1 - p2),
                    }
                )

        df = pd.DataFrame(rows).sort_values("_diff", ascending=False).head(top_n).drop(columns="_diff")
        results[(int(c1), int(c2))] = df.reset_index(drop=True)

    return results


# ------------------------------------------------------------
# Main (pam_cluster_profiles_k6)
# ------------------------------------------------------------


def pam_cluster_profiles_main():
    global feature_names
    encoded_df = pd.read_csv(ENCODED_PATH)
    if "respondent_id" in encoded_df.columns:
        encoded_df = encoded_df.drop(columns=["respondent_id"])
    X = encoded_df.to_numpy(dtype=float)
    feature_names = encoded_df.columns.tolist()
    labels = np.load(LABELS_PATH).astype(int)

    clusters = sorted(np.unique(labels))

    size_rows = []
    for c in clusters:
        n = int(np.sum(labels == c))
        size_rows.append({
            "cluster": int(c),
            "count": n,
            "percent": round(n / len(labels) * 100, 1),
        })
    size_df = pd.DataFrame(size_rows)
    display(HTML("<b>k=6 Cluster Sizes</b>" + size_df.to_html(index=False)))

    # ------------------------
    # ------------------------
    df_raw = pd.read_csv(DRIVERS_RAW_PATH)
    labels_raw = align_labels_to_raw(df_raw, labels)
    make_all_raw_feature_plots(df_raw, labels_raw)

    # Raw pairwise feature separation heatmap (Top 30)
    raw_all = _build_raw_pairwise_diff_matrix(df_raw, labels_raw, drop_missing=False)
    out_top30 = OUT_DIR / "pairwise_raw_feature_separation_top30_k6.png"
    plot_heatmap(
        raw_all,
        "Raw feature separation (top 30 by max diff) — k=6",
        out_top30,
        vmax=None,
        max_rows=30,
    )

    display(Image(filename=str(out_top30)))

    # Raw feature answers: cluster vs rest (Top 15)
    exclude_subs = {
        1: ["not applicable"],
    }
    raw_tables = build_raw_cluster_vs_rest_tables(
        df_raw,
        labels_raw,
        top_n=15,
        exclude_answer_substrings=exclude_subs,
    )
    for c in sorted(raw_tables.keys()):
        if c == 0:
            display(HTML("<b>CLUSTER 0 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))

            selected = [
                "prev_boss=Yes",
                "bad_conseq_mh_boss=Maybe",
                "anonymity_protected=I don't know",
                "prev_boss_mh_discuss=None did",
                "boss_mh_discuss=No",
            ]
            forced_df = build_raw_cluster_vs_rest_selected(
                df_raw=df_raw,
                labels_aligned=labels_raw,
                cluster_id=0,
                selected_features=selected,
            )
            if not forced_df.empty:
                display(HTML("<b>CLUSTER 0 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
        elif c == 1:
            display(HTML("<b>CLUSTER 1 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))

            selected = [
                "resources=No",
                "mh_options_known=Yes",
                "boss_mh_discuss=None did",
                "mh_ph_boss_serious=I don't know",
                "bad_conseq_ph_boss=No",
                "bad_conseq_mh_boss=Maybe",
                "anonymity_protected=I don't know",
                "mh_interview=No",
                "mh_comfort_coworkers=No",
                "mh_comfort_supervisor=No",
            ]
            # fix common typos
            selected = [s.replace("bad_consew", "bad_conseq") for s in selected]
            forced_df = build_raw_cluster_vs_rest_selected(
                df_raw=df_raw,
                labels_aligned=labels_raw,
                cluster_id=1,
                selected_features=selected,
            )
            if not forced_df.empty:
                display(HTML("<b>CLUSTER 1 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
        elif c == 2:
            display(HTML("<b>CLUSTER 2 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))

            selected = [
                "resources=No",
                "resources=I don't know",
                "prev_resources=Some did",
                "prev_resources=Yes, they all did",
                "mh_ph_boss_serious=I don't know",
                "mh_ph_boss_serious=No",
                "bad_conseq_mh_boss=Maybe",
                "bad_conseq_mh_prev_boss=Some of them",
                "bad_conseq_mh_prev_boss=Yes, all of them",
                "boss_mh_discuss=No",
                "prev_boss_mh_discuss=None did",
                "prev_observed_bad_conseq_mh=Some of them",
                "prev_observed_bad_conseq_mh=Yes, all of them",
            ]
            selected = [s.replace("prev_resouces", "prev_resources") for s in selected]
            forced_df = build_raw_cluster_vs_rest_selected(
                df_raw=df_raw,
                labels_aligned=labels_raw,
                cluster_id=2,
                selected_features=selected,
            )
            if not forced_df.empty:
                display(HTML("<b>CLUSTER 2 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
        elif c == 3:
            display(HTML("<b>CLUSTER 3 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))

            selected = [
                "boss_mh_discuss=No",
                "mh_ph_boss_serious=No",
                "bad_conseq_mh_boss=Yes",
                "anonymity_protected=I don't know",
                "anonymity_protected=No",
                "mh_comfort_supervisor=No",
                "mh_comfort_coworkers=No",
                "mh_interview=No",
                "ph_interview=No",
                "leave_easy=Very difficult",
                "leave_easy=Somewhat difficult",
                "mh_family_history=Yes",
                "ever_observed_mhd_bad_response=Yes, I observed",
                "ever_observed_mhd_bad_response=Yes, I experienced",
                "prev_benefits=No, none did",
            ]
            forced_df = build_raw_cluster_vs_rest_selected(
                df_raw=df_raw,
                labels_aligned=labels_raw,
                cluster_id=3,
                selected_features=selected,
            )
            if not forced_df.empty:
                display(HTML("<b>CLUSTER 3 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
        elif c == 4:
            display(HTML("<b>CLUSTER 4 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))

            selected = [
                "prev_anonymity_protected=I don't know",
                "anonymity_protected=I don't know",
                "boss_mh_discuss=No",
                "bad_conseq_mh_boss=No",
                "bad_conseq_mh_boss=Maybe",
                "prev_observed_bad_conseq_mh=None of them",
                "mh_family_history=No",
            ]
            forced_df = build_raw_cluster_vs_rest_selected(
                df_raw=df_raw,
                labels_aligned=labels_raw,
                cluster_id=4,
                selected_features=selected,
            )
            if not forced_df.empty:
                display(HTML("<b>CLUSTER 4 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
        elif c == 5:
            display(HTML("<b>CLUSTER 5 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))

            selected = [
                "mh_options_known=Yes",
                "leave_easy=Very easy",
                "leave_easy=Somewhat easy",
                "leave_easy=Neither easy nor difficult",
                "bad_conseq_mh_boss=Yes",
                "mh_comfort_coworkers=Maybe",
                "mh_comfort_supervisor=Maybe",
                "friends_family_mhd_comfort=Somewhat open",
                "friends_family_mhd_comfort=Very open",
                "mh_family_history=Yes",
            ]
            forced_df = build_raw_cluster_vs_rest_selected(
                df_raw=df_raw,
                labels_aligned=labels_raw,
                cluster_id=5,
                selected_features=selected,
            )
            if not forced_df.empty:
                display(HTML("<b>CLUSTER 5 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
        else:
            print("\n" + "=" * 100)
            print(f"RAW (pre-encoding) — CLUSTER {c} vs REST (Top 15 by deviation)")
            print("=" * 100)
            print(raw_tables[c].to_string(index=False))

    # Raw feature answers: pairwise cluster comparisons (Top 15)
    pairs = [(0, 1), (2, 5), (3, 4)]
    pair_tables = build_raw_pairwise_tables(df_raw, labels_raw, pairs=pairs, top_n=15)
    for c1, c2 in pairs:
        title = f"CLUSTER {c1} vs CLUSTER {c2} (Top deviations)"
        display(HTML("<b>" + title + "</b>" + pair_tables[(c1, c2)].to_html(index=False)))


pam_cluster_profiles_main()

#%% [markdown]
# ### 11. Overlay Profiles
# Clusters are characterized using overlay variables that were excluded from the clustering drivers.

#%% 11 Overlay Profiles





# -----------------------------
# Helpers: string normalization
# -----------------------------

def norm_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("\u00a0", " ")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_missing_like(s):
    s = norm_text(s)
    if s in {"", "na", "n/a", "nan", "none", "null", "no answer"}:
        return True
    if any(k in s for k in ["prefer not", "rather not", "no comment", "none of your business", "dont want", "don't want"]):
        return True
    return False


# -----------------------------
# Recode: gender
# -----------------------------

GENDER_TARGET_ORDER = [
    "Male",
    "Female",
    "Transgender Male",
    "Transgender Female",
    "Nonbinary / Gender Diverse",
    "Other",
    "Prefer Not to Say / Invalid Response",
]

def recode_gender(val):
    s = norm_text(val)

    if is_missing_like(s):
        return "Prefer Not to Say / Invalid Response"

    sp = re.sub(r"[^a-z0-9\s\-\/]", " ", s)
    sp = re.sub(r"\s+", " ", sp).strip()

    # Trans
    if any(k in sp for k in ["ftm", "f2m", "trans man", "transman", "trans male", "transmale", "male (trans", "transmasc", "trans masc"]):
        return "Transgender Male"
    if any(k in sp for k in ["mtf", "m2f", "trans woman", "transwoman", "trans female", "transfemale", "transfeminine", "trans feminine", "other/transfeminine"]):
        return "Transgender Female"

    # Nonbinary / gender diverse
    if any(k in sp for k in [
        "nonbinary", "non-binary", "enby", "genderqueer", "agender", "genderfluid",
        "bigender", "androgynous", "genderflux", "multi-gender", "multigender", "gender diverse"
    ]):
        return "Nonbinary / Gender Diverse"
    toks = sp.split()
    if "nb" in toks or any(t.startswith("nb") for t in toks):
        return "Nonbinary / Gender Diverse"
    if "queer" in toks and ("male" not in toks and "female" not in toks):
        return "Nonbinary / Gender Diverse"

    tokens = set(toks)

    # known typo
    if sp == "mail":
        return "Male"

    # Male / Female
    if ("male" in tokens) or ("man" in tokens) or ("dude" in tokens) or (sp == "m") or ("sex is male" in sp) or ("cis male" in sp) or ("cis man" in sp):
        if "female" not in tokens and "woman" not in tokens:
            return "Male"

    if ("female" in tokens) or ("woman" in tokens) or (sp == "f") or ("cis female" in sp) or ("cis-woman" in sp) or ("cisgender female" in sp):
        if "male" not in tokens and "man" not in tokens:
            return "Female"

    return "Other"


# -----------------------------
# Recode: age bins (custom)
# -----------------------------

AGE_LABELS = [
    "less than 20",
    "from 20 to 25",
    "from 26 to 30",
    "from 31 to 35",
    "from 36 to 40",
    "from 41 to 50",
    "more than 51",
]

def recode_age(val):
    v = pd.to_numeric(val, errors="coerce")
    if not np.isfinite(v):
        return "NA"
    v = float(v)

    # Clip extreme outliers into closest bucket:
    # - below 0 still ends up in "less than 20"
    # - above 120 still ends up in "more than 51"
    # (you can tighten these if desired)
    if v < 20:
        return "less than 20"
    if 20 <= v <= 25:
        return "from 20 to 25"
    if 26 <= v <= 30:
        return "from 26 to 30"
    if 31 <= v <= 35:
        return "from 31 to 35"
    if 36 <= v <= 40:
        return "from 36 to 40"
    if 41 <= v <= 50:
        return "from 41 to 50"
    if v >= 51:
        return "more than 51"
    return "NA"


# -----------------------------
# Recode: country -> region
# -----------------------------

EU_COUNTRIES = {
    "austria","belgium","bulgaria","croatia","cyprus","czech republic","czechia","denmark","estonia","finland",
    "france","germany","greece","hungary","ireland","italy","latvia","lithuania","luxembourg","malta","netherlands",
    "poland","portugal","romania","slovakia","slovenia","spain","sweden"
}
OTHER_EUROPE = {
    "norway","switzerland","iceland","serbia","bosnia and herzegovina","bosnia & herzegovina","bosnia",
    "ukraine","belarus","albania","north macedonia","macedonia","montenegro","moldova","georgia","armenia","azerbaijan"
}
MIDDLE_EAST = {"israel","iran","iraq","jordan","lebanon","saudi arabia","united arab emirates","uae","qatar","kuwait","oman","yemen","syria","turkey"}
ASIA = {"india","pakistan","bangladesh","sri lanka","nepal","bhutan","china","hong kong","taiwan","japan","south korea","korea","vietnam","thailand","malaysia","singapore","indonesia","philippines","myanmar","cambodia","laos","mongolia","afghanistan"}
OCEANIA = {"australia","new zealand"}
AFRICA = {"south africa","nigeria","kenya","ghana","egypt","morocco","algeria","tunisia","ethiopia","uganda","tanzania","zambia","zimbabwe","namibia","botswana"}
LATAM = {"mexico","brazil","argentina","colombia","chile","peru","ecuador","uruguay","paraguay","bolivia","venezuela","guatemala","costa rica","panama","honduras","el salvador","nicaragua","dominican republic","cuba","haiti","jamaica","trinidad and tobago"}

def recode_country_region(val):
    s = norm_text(val)
    if is_missing_like(s):
        return "Other/NA"

    s = s.replace("u.s.", "us").replace("u.s.a.", "usa").replace("united states", "united states of america")
    s = s.replace("england", "united kingdom").replace("scotland", "united kingdom").replace("wales", "united kingdom")
    if s in {"uk", "u.k."}:
        s = "united kingdom"

    if s in {"united states of america", "usa", "us", "united states"}:
        return "US"
    if s == "united kingdom":
        return "UK"
    if s == "canada":
        return "Canada"

    if s in EU_COUNTRIES:
        return "European Union"
    if s in OTHER_EUROPE:
        return "Other Europe"
    if s in MIDDLE_EAST:
        return "Middle East"
    if s in OCEANIA:
        return "Oceania"
    if s in ASIA:
        return "Asia"
    if s in AFRICA:
        return "Africa"
    if s in LATAM:
        return "Latin America & Caribbean"

    if "united states" in s:
        return "US"
    if "kingdom" in s:
        return "UK"

    return "Other/NA"


# -----------------------------
# Recode: med_pro_condition -> simplified
# -----------------------------

MED_GROUPS_ORDER = [
    "No diagnosis/NA",
    "Mood disorder",
    "anxiety disorder",
    "other",
]

def recode_med_condition(val):
    s = norm_text(val)
    if is_missing_like(s):
        return "No diagnosis/NA"

    # treat explicit no-diagnosis signals
    if any(k in s for k in ["no diagnosis", "no dx", "none", "healthy", "no condition", "no mental"]):
        return "No diagnosis/NA"

    if ("mood" in s) and ("disorder" in s):
        return "Mood disorder"
    if ("anxiety" in s) and ("disorder" in s):
        return "anxiety disorder"

    return "other"


# -----------------------------
# Recode: work_position -> hierarchy-based buckets
# -----------------------------

WORK_ORDER = [
    "Sales",
    "Dev Evangelist/Advocate",
    "Leadership (Supervisor/Exec)",
    "DevOps/SysAdmin",
    "Support",
    "Full-stack (FE+BE)",
    "Front-end Developer",
    "Back-end Developer",
    "Designer",
    "other",
    "NA",
]

def _contains_any(s, needles):
    return any(n in s for n in needles)

def recode_work_position(val):
    s = norm_text(val)
    if is_missing_like(s):
        return "NA"

    # normalize separators and small typos
    s = s.replace("decops", "devops")  # user mentioned "DecOps/SysAdmin"
    s = s.replace("front'end", "front-end")  # user mentioned "Front'end Developer"
    s = s.replace("frontend", "front-end")
    s = s.replace("backend", "back-end")
    s = s.replace("sys admin", "sysadmin")
    s = s.replace("sys-admin", "sysadmin")

    # tokens / flags (do not rely on exact pipe formatting)
    has_sales = "sales" in s

    has_evangelist = _contains_any(s, ["dev evangelist", "developer evangelist", "advocat", "advocate"])
    has_leadership = _contains_any(s, ["supervisor/team lead", "team lead", "supervisor", "executive leadership", "executive"])

    has_devops = _contains_any(s, ["devops", "sysadmin", "sys admin", "sys-admin"])
    has_support = "support" in s

    has_fe = _contains_any(s, ["front-end developer", "front end developer", "front-end"])
    has_be = _contains_any(s, ["back-end developer", "back end developer", "back-end"])
    has_designer = "designer" in s

    # Hierarchy (requested): Sales > Evangelist > Leadership > DevOps > Support > FE+BE > FE/BE > other
    if has_sales:
        return "Sales"
    if has_evangelist:
        return "Dev Evangelist/Advocate"
    if has_leadership:
        return "Leadership (Supervisor/Exec)"
    if has_devops:
        return "DevOps/SysAdmin"
    if has_support:
        return "Support"

    # After higher-priority buckets, handle dev roles
    if has_fe and has_be:
        return "Full-stack (FE+BE)"

    # "Front-end Developer", "Designer", "Front-end Developer|Designer" -> group
    # If FE + designer but no BE, keep FE as primary (unless you prefer separate FE+Designer bucket).
    if has_fe and (not has_be):
        return "Front-end Developer"

    if has_be and (not has_fe):
        return "Back-end Developer"

    if has_designer:
        return "Designer"

    return "other"


# -----------------------------
# Alignment: labels <-> respondent_id
# -----------------------------

def align_labels_to_overlays(df_over, labels, encoded_drivers_csv):
    if "respondent_id" in df_over.columns and encoded_drivers_csv.exists():
        df_enc = pd.read_csv(encoded_drivers_csv)
        if "respondent_id" in df_enc.columns and len(df_enc) == len(labels):
            id_to_label = pd.Series(labels, index=df_enc["respondent_id"]).to_dict()
            mapped = df_over["respondent_id"].map(id_to_label).to_numpy()
            return mapped.astype(int)

    return labels.astype(int)


# -----------------------------
# Plotting: stacked proportions
# -----------------------------

def plot_categorical_stacked_by_cluster(series, labels, title, out_path=None, category_order=None, legend_title="Answer", min_prop_to_keep=0.0, max_categories=None, ax=None):
    s = series.copy()
    s = s.astype(object)
    s[pd.isna(s)] = "NA"
    s = s.astype(str).str.strip()
    s = s.replace({"": "NA"})

    overall = s.value_counts(dropna=False)
    total_n = float(len(s)) if len(s) else 1.0
    cats = overall.index.tolist()

    # Optional collapse for high-cardinality fields
    if max_categories is not None and len(cats) > max_categories:
        keep = set(cats[:max_categories])
        s = s.where(s.isin(keep), other="Other")
        overall = s.value_counts(dropna=False)
        cats = overall.index.tolist()

    if min_prop_to_keep > 0.0:
        keep = set(overall[(overall / total_n) >= min_prop_to_keep].index.tolist())
        if len(keep) < len(cats):
            s = s.where(s.isin(keep), other="Other")
            overall = s.value_counts(dropna=False)
            cats = overall.index.tolist()

    if category_order is not None:
        ordered = [c for c in category_order if c in cats]
        extras = [c for c in cats if c not in ordered]
        cats = ordered + extras

    clusters = sorted(np.unique(labels).astype(int).tolist())

    mat = []
    for c in clusters:
        s_c = s[labels == c]
        vc = s_c.value_counts(dropna=False)
        denom = float(len(s_c)) if len(s_c) else 1.0
        mat.append([float(vc.get(cat, 0)) / denom for cat in cats])

    mat = np.asarray(mat, dtype=float)

    if ax is None:
        fig = plt.figure(figsize=(10, 4.5), dpi=150)
        ax = plt.gca()
    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    for j, cat in enumerate(cats):
        ax.bar(
            x,
            mat[:, j],
            bottom=bottom,
            label=str(cat),
            edgecolor="none",
            linewidth=0.0,
            antialiased=False,
        )
        bottom += mat[:, j]

    ax.set_xticks(x, [str(c) for c in clusters])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion within cluster")
    ax.set_title(title)
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)


def dominant_answer_table(df, labels, features):
    clusters = sorted(np.unique(labels).astype(int).tolist())
    rows = []
    for c in clusters:
        row = {"cluster": c}
        for feat in features:
            if feat not in df.columns:
                row[feat] = "NA"
                continue
            s = df[feat].astype(object)
            s[pd.isna(s)] = "NA"
            s = s.astype(str).str.strip().replace({"": "NA"})
            s_c = s[labels == c]
            denom = float(len(s_c)) if len(s_c) else 1.0
            vc = s_c.value_counts(dropna=False)
            top_val = str(vc.index[0])
            pct = (vc.iloc[0] / denom) * 100.0
            row[feat] = f"{top_val} ({pct:.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_answer_pct_table(series, labels):
    s = series.astype(object)
    s[pd.isna(s)] = "NA"
    s = s.astype(str).str.strip().replace({"": "NA"})
    clusters = sorted(np.unique(labels).astype(int).tolist())
    cats = s.value_counts(dropna=False).index.tolist()
    rows = []
    for c in clusters:
        s_c = s[labels == c]
        denom = float(len(s_c)) if len(s_c) else 1.0
        vc = s_c.value_counts(dropna=False)
        row = {"cluster": c}
        for cat in cats:
            row[str(cat)] = f"{(vc.get(cat, 0) / denom) * 100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Main: apply recodes + plot
# -----------------------------

def apply_recodes(df_over):
    df = df_over.copy()

    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(recode_gender)

    if "age" in df.columns:
        df["age"] = df["age"].apply(recode_age)

    if "country_live" in df.columns:
        df["country_live"] = df["country_live"].apply(recode_country_region)

    if "med_pro_condition" in df.columns:
        df["med_pro_condition"] = df["med_pro_condition"].apply(recode_med_condition)

    if "work_position" in df.columns:
        df["work_position"] = df["work_position"].apply(recode_work_position)

    return df


#%% 11A Overlay Config
OVERLAYS_PATH = PROJECT_ROOT / "data" / "out" / "overlays_clean.csv"
OVERLAY_LABELS_PATH = PROJECT_ROOT / "data" / "out" / "pam" / "pam_labels_k6.npy"
OVERLAY_ENCODED_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"
OVERLAY_OUTDIR = PROJECT_ROOT / "data" / "out" / "pam_post" / "overlay_distributions_k6_cleaned"
OVERLAY_K = 6

OVERLAY_OUTDIR.mkdir(parents=True, exist_ok=True)

df_over = pd.read_csv(OVERLAYS_PATH)
labels = np.load(OVERLAY_LABELS_PATH).astype(int)
labels_aligned = align_labels_to_overlays(df_over, labels, OVERLAY_ENCODED_PATH)
df = apply_recodes(df_over)

features = [c for c in df.columns if c != "respondent_id"]
exclude = {
    "mhd_believe_condition",
    "mhd_diagnosed_condition",
    "treat_mhd_bad_work",
    "US_state",
    "US_work",
}
features = [c for c in features if c not in exclude]

#%% 11B Overlay Panel Age/Gender/Country/Position
panel_feats = ["age", "gender", "country_work", "work_position"]
if all(f in df.columns for f in panel_feats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
    axes = axes.ravel()

    plot_categorical_stacked_by_cluster(
        series=df["age"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): age",
        out_path=None,
        category_order=AGE_LABELS + ["NA"],
        legend_title="Answer",
        ax=axes[0],
    )
    plot_categorical_stacked_by_cluster(
        series=df["gender"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): gender",
        out_path=None,
        category_order=GENDER_TARGET_ORDER,
        legend_title="Answer",
        ax=axes[1],
    )
    plot_categorical_stacked_by_cluster(
        series=df["country_work"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): country_work",
        out_path=None,
        legend_title="Answer",
        max_categories=15,
        min_prop_to_keep=0.01,
        ax=axes[2],
    )
    plot_categorical_stacked_by_cluster(
        series=df["work_position"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): work_position",
        out_path=None,
        category_order=WORK_ORDER,
        legend_title="Answer",
        ax=axes[3],
    )

    panel_path = OVERLAY_OUTDIR / f"overlay_k{OVERLAY_K}_panel_age_gender_country_work_position.png"
    plt.tight_layout()
    fig.savefig(panel_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)
else:
    missing = [f for f in panel_feats if f not in df.columns]
    print(f"Skipping overlay panel (missing columns: {missing})")

#%% 11C Overlay Panel MHD Past/Current/Treatment
panel_feats = ["mhd_past", "current_mhd", "pro_treatment", "no_treat_mhd_bad_work"]
if all(f in df.columns for f in panel_feats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
    axes = axes.ravel()

    plot_categorical_stacked_by_cluster(
        series=df["mhd_past"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): mhd_past",
        out_path=None,
        legend_title="Answer",
        ax=axes[0],
    )
    plot_categorical_stacked_by_cluster(
        series=df["current_mhd"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): current_mhd",
        out_path=None,
        legend_title="Answer",
        ax=axes[1],
    )
    plot_categorical_stacked_by_cluster(
        series=df["pro_treatment"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): pro_treatment",
        out_path=None,
        legend_title="Answer",
        ax=axes[2],
    )
    plot_categorical_stacked_by_cluster(
        series=df["no_treat_mhd_bad_work"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): no_treat_mhd_bad_work",
        out_path=None,
        legend_title="Answer",
        ax=axes[3],
    )

    panel_path = OVERLAY_OUTDIR / f"overlay_k{OVERLAY_K}_panel_mhd_past_current_mhd_pro_treatment_no_treat_mhd_bad_work.png"
    plt.tight_layout()
    fig.savefig(panel_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)
else:
    missing = [f for f in panel_feats if f not in df.columns]
    print(f"Skipping overlay panel (missing columns: {missing})")

#%% 11D Overlay Panel Career/Perception/Diagnosis
panel_feats = ["mhd_hurt_career", "coworkers_view_neg_mhd", "med_pro_condition", "mhd_by_med_pro"]
if all(f in df.columns for f in panel_feats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
    axes = axes.ravel()

    plot_categorical_stacked_by_cluster(
        series=df["mhd_hurt_career"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): mhd_hurt_career",
        out_path=None,
        legend_title="Answer",
        ax=axes[0],
    )
    plot_categorical_stacked_by_cluster(
        series=df["coworkers_view_neg_mhd"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): coworkers_view_neg_mhd",
        out_path=None,
        legend_title="Answer",
        ax=axes[1],
    )
    plot_categorical_stacked_by_cluster(
        series=df["med_pro_condition"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): med_pro_condition",
        out_path=None,
        category_order=MED_GROUPS_ORDER,
        legend_title="Answer",
        ax=axes[2],
    )
    plot_categorical_stacked_by_cluster(
        series=df["mhd_by_med_pro"],
        labels=labels_aligned,
        title=f"Overlay distribution by cluster (k={OVERLAY_K}): mhd_by_med_pro",
        out_path=None,
        legend_title="Answer",
        ax=axes[3],
    )

    panel_path = OVERLAY_OUTDIR / f"overlay_k{OVERLAY_K}_panel_mhd_hurt_career_coworkers_view_neg_mhd_med_pro_condition_mhd_by_med_pro.png"
    plt.tight_layout()
    fig.savefig(panel_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)
else:
    missing = [f for f in panel_feats if f not in df.columns]
    print(f"Skipping overlay panel (missing columns: {missing})")

#%% 11E Overlay Tables
dom_feats = ["mhd_by_med_pro", "pro_treatment", "mhd_past", "current_mhd", "med_pro_condition"]
if all(f in df.columns for f in dom_feats):
    dom_df = dominant_answer_table(df, labels_aligned, dom_feats)
    display(HTML("<b>Dominant answers (overlays)</b>" + dom_df.to_html(index=False)))

pct_feats = ["no_treat_mhd_bad_work", "mhd_hurt_career", "coworkers_view_neg_mhd"]
for feat in pct_feats:
    if feat not in df.columns:
        continue
    pct_df = cluster_answer_pct_table(df[feat], labels_aligned)
    display(HTML(f"<b>{feat}</b>" + pct_df.to_html(index=False)))
