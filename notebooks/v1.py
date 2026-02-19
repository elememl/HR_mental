#%% 00 Config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#% 01 Rename Columns
import pandas as pd

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

#% 02 Cleaning + Drivers + Overlays
from typing import Optional, Tuple
import pandas as pd
import numpy as np

def _log_change(step: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    """
    Print a single summary line: rows removed, cols removed, and shapes.
    """
    rows_removed = before.shape[0] - after.shape[0]
    cols_removed = before.shape[1] - after.shape[1]
    print(f"{step}: removed {rows_removed} rows, {cols_removed} cols "
          f"(shape {before.shape} -> {after.shape})")


def filter_not_self_employed(df: pd.DataFrame) -> pd.DataFrame:
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


def filter_tech_role(df: pd.DataFrame) -> pd.DataFrame:
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


def clean_population_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply your two population filters in order:
      1) filter_not_self_employed
      2) filter_tech_role  (keeps Yes + missing, removes only explicit No)
    """
    df1 = filter_not_self_employed(df)
    df2 = filter_tech_role(df1)
    return df2



NA_TOKEN = "Not applicable"

_MISSING_STRINGS = {"nan", "na", "n/a", "null", "none", "<na>", "<nan>", ""}

def canonicalize_true_missing(df: pd.DataFrame) -> pd.DataFrame:
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


def apply_driver_skip_logic(df: pd.DataFrame) -> pd.DataFrame:
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

def standardize_binary_drivers(df: pd.DataFrame) -> pd.DataFrame:
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
import pandas as pd

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

def extract_overlays(df: pd.DataFrame) -> pd.DataFrame:
    # keep only columns that actually exist (safe)
    cols = [c for c in OVERLAY_COLS if c in df.columns]
    return df[cols].copy()
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_table(df: pd.DataFrame, title: str, max_rows: int = 25, figsize=(12, 6), dpi=150) -> None:
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


def main() -> None:
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
    # subset size and boolean summary output removed

    # ---------------------------------------------------------------------
    # Main cleaning pipeline (drops self_employed and tech_role)
    # ---------------------------------------------------------------------
    df_clean = clean_population_filters(df)

    # -----------------------------
    # Add respondent ID (stable row-based)
    # -----------------------------

    df_clean = df_clean.reset_index(drop=True)
    df_clean.insert(0, "respondent_id", range(1, len(df_clean) + 1))


    # ---------------------------------------------------------------------
    # Row missingness: summary + histogram
    # ---------------------------------------------------------------------
    row_miss = df_clean.isna().mean(axis=1)

    summary = (row_miss.describe()[["min", "max"]] * 100).round(1)
    summary.index = ["min", "max"]
    try:
        from IPython.display import display
        display(summary.to_frame(name="Row missingness (%)"))
    except Exception:
        print(summary.to_string())

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

    try:
        from IPython.display import display, HTML
        display(HTML(top15.to_html(index=False)))
    except Exception:
        print(top15)

    # Histogram of column missingness (fractions)
    col_miss = df_clean.isna().mean()
    plt.figure(figsize=(8, 5), dpi=150)
    plt.hist(col_miss, bins=20)
    plt.xlabel("Fraction missing per column")
    plt.ylabel("Number of columns")
    plt.title("Column missingness distribution")
    plt.tight_layout()
    plt.show()

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
    df_overlays = extract_overlays(df_clean)

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

    # Show only features with any missingness (matches requested view) as text table
    audit_missing = audit_df[audit_df["% missing"] > 0].reset_index(drop=True)
    try:
        from IPython.display import display, HTML
        display(HTML(audit_missing.to_html(index=False)))
    except Exception:
        print(audit_missing.to_string(index=False))

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

    # Strong missingness-correlation pairs table removed per request

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

    try:
        from IPython.display import display, HTML
        display(HTML("<b>Contingency table (raw)</b>" + full_ct.to_html()))
    except Exception:
        print(full_ct)

    # apply parent-child question changes
    df_drivers = apply_driver_skip_logic(df_drivers)


    # contigency table after applying rule C and enforcing 'non applicable'
    ct_after = pd.crosstab(
        df_drivers["benefits"],
        df_drivers["mh_options_known"],
        dropna=False
    )

    try:
        from IPython.display import display, HTML
        display(HTML("<b>Contingency table (after Rule C)</b>" + ct_after.to_html()))
    except Exception:
        print(ct_after)


    # ============================================================
    # PROOF TABLES (post skip-logic)  <-- place AFTER apply_driver_skip_logic
    # ============================================================
    NA_TOKEN = "Not applicable"

    # ---------- Helper to compute proof metrics ----------
    def _rule_metrics(rule_name: str, parent_mask_false: pd.Series, child: pd.Series) -> dict:
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
    else:
        # Structural missingness proof table removed per request
        pass

    # ---------- TABLE 1B: failures only (only if needed) ----------
    fail_df = pd.DataFrame(proof_failures)
    if not fail_df.empty:
        # Failures table removed per request
        pass

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

        try:
            from IPython.display import display, HTML
            display(HTML(true_miss_df.to_html(index=False)))
        except Exception:
            print(true_miss_df.to_string(index=False))

        # Dependency situation (NO threshold): show top 20 |corr| among NaN indicators
        miss_bin = drivers[miss_nan.index].isna().astype(int)

        if miss_bin.shape[1] >= 2:
            # dependencies print removed
            pass
        else:
            show_table(
                pd.DataFrame([{
                    "result": "INFO",
                    "message": "Only one truly-missing feature -> no dependency pairs exist."
                }]),
                "True-missingness dependencies",
                max_rows=5,
                figsize=(12, 2.5)
            )

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
            # Optional safety check: ensure only {0,1}
            bad = ~df_drivers[col].isin([0, 1, 0.0, 1.0])
            if bad.any():
                raise ValueError(f"{col} has non-binary numeric values: {df_drivers.loc[bad, col].unique()}")
            df_drivers[col] = df_drivers[col].astype(int)

        else:
            # If text Yes/No appears, map it
            s = df_drivers[col].astype(str).str.strip().str.lower()
            mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
            df_drivers[col] = s.map(mapping)

            if df_drivers[col].isna().any():
                bad_vals = df_drivers.loc[df_drivers[col].isna(), col].unique()
                raise ValueError(f"{col} has unmapped values: {bad_vals}")

            df_drivers[col] = df_drivers[col].astype(int)

    # AFTER: confirm binaries are now int 0/1
    # binary check prints removed

    # ============================================================
    # POST-FIX VALIDATION CHECKS
    # ============================================================

    # POST-FIX VALIDATION prints removed by request

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

    try:
        from IPython.display import display, HTML
        display(HTML(audit_table.to_html(index=False)))
    except Exception:
        print(audit_table.to_string(index=False))

    # ---------------------------------------------------------------------
    # Save drivers
    # ---------------------------------------------------------------------
    out_path = PROJECT_ROOT / "data" / "out" / "survey_drivers.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_drivers.to_csv(out_path, index=False)
    # saved drivers message removed


main()

#%% 03 Encoding
from pathlib import Path
import pandas as pd

IN_PATH = PROJECT_ROOT / "data" / "out" / "survey_drivers.csv"
OUT_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"

_ = IN_PATH.exists()

# -------------------
# Helpers / constants
# -------------------
NA_TOKEN = "Not applicable"
IDK_TOKEN = "I don't know"


def assert_binary_01(s: pd.Series, name: str) -> None:
    bad = ~s.isin([0, 1])
    if bad.any():
        raise ValueError(f"{name} has non 0/1 values: {sorted(s[bad].unique().tolist())}")


def norm_str(x):
    if pd.isna(x):
        return pd.NA
    return str(x).strip()


def encode_ordinal(series: pd.Series, mapping: dict, name: str) -> pd.Series:
    """
    Encode an ordinal feature using an explicit mapping (text -> number).
    Preserves NaN. Raises if any non-NaN category is unmapped.
    Returns float to allow NaNs.
    """
    s = series.map(norm_str).astype("string")
    enc = s.map(mapping)

    bad_mask = s.notna() & enc.isna()
    if bad_mask.any():
        bad_vals = sorted(s[bad_mask].unique().tolist())
        raise ValueError(f"{name}: unmapped categories found: {bad_vals}")

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


def encode_company_size(series: pd.Series) -> pd.Series:
    """
    Robust company_size encoding:
    - If numeric-coded (1..6), validate and return.
    - Else map text bins using ORDINAL_MAPS['company_size'].
    """
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        allowed = set(range(1, 7))
        bad = s_num.notna() & (~s_num.isin(list(allowed)))
        if bad.any():
            bad_vals = sorted(s_num[bad].unique().tolist())
            raise ValueError(f"company_size numeric values outside {sorted(list(allowed))}: {bad_vals}")
        return s_num.astype("float")

    return encode_ordinal(series, ORDINAL_MAPS["company_size"], "company_size")


# ==========================================================
# MIXED ENCODING (v1) — HISTORY (ordinal + NA/IDK flags only)
# ==========================================================
def encode_mixed_ord_plus_flags(
    df: pd.DataFrame,
    feature: str,
    ord_mapping: dict,
    na_tokens: set | None = None,
    idk_tokens: set | None = None,
    collapse_map: dict | None = None,
    drop_all_zero_flags: bool = True,
) -> pd.DataFrame:
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
# MIXED (v1) SPECS (7 features) — already working
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
# MIXED ENCODING (v2) — NEW (ordinal + multiple nominal flags)
# ==========================================================
def encode_mixed_ord_plus_flags_v2(
    df: pd.DataFrame,
    feature: str,
    ord_mapping: dict,
    na_tokens: set | None = None,
    idk_tokens: set | None = None,
    extra_flag_tokens: dict | None = None,  # {"No response": "__no_response", ...}
    collapse_map: dict | None = None,
    drop_all_zero_flags: bool = True,
) -> pd.DataFrame:
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
# NEW MIXED (v2) SPECS — “1 ordinal + 2 nominal” (and one case with 3 flags)
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


def main() -> None:
    df = pd.read_csv(IN_PATH)

    if "respondent_id" not in df.columns:
        raise ValueError("survey_drivers.csv must contain respondent_id")

    # input shape prints removed

    # -------------------
    # 1) Binary
    # -------------------
    binary_cols = ["tech_company", "prev_boss", "observed_mhdcoworker_bad_conseq"]
    binary_cols = [c for c in binary_cols if c in df.columns]

    for c in binary_cols:
        df[c] = pd.to_numeric(df[c], errors="raise")
        assert_binary_01(df[c], c)

    out = pd.DataFrame({"respondent_id": df["respondent_id"]})
    for c in binary_cols:
        out[c] = df[c].astype(int)

    # after-binary prints removed

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

    # nominal input prints removed

    X_nom = df[nominal_cols].copy()
    X_nom = X_nom.fillna("NaN")
    for c in nominal_cols:
        X_nom[c] = X_nom[c].astype(str).str.strip()

    nom_dum = pd.get_dummies(X_nom, prefix=nominal_cols, prefix_sep="=")

    # nominal one-hot detail prints removed

    cols_before = out.shape[1]
    out = out.join(nom_dum)
    # after nominal prints removed

    # -------------------
    # 3) Ordinal
    # -------------------
    # ordinal input prints removed
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

    # ordinal input prints removed

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

    # after ordinal prints removed

    # -------------------
    # 4) Mixed (v1): ordinal + (NA/IDK) flags
    # -------------------
    # mixed v1 input prints removed

    mixed_cols = [c for c in MIXED_SPECS.keys() if c in df.columns]
    # mixed v1 input prints removed

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

        # mixed v1 detail prints removed

        mixed_out = mixed_out.join(enc_block)

    cols_before = out.shape[1]
    out = out.join(mixed_out)

    # after mixed v1 prints removed

    # -------------------
    # 5) Mixed (v2): ordinal + TWO nominal flags (plus optional extra flags)
    #     (inspected separately, as requested)
    # -------------------
    # mixed v2 input prints removed

    mixed_v2_cols = [c for c in MIXED_SPECS_V2.keys() if c in df.columns]
    # mixed v2 input prints removed

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

        # mixed v2 detail prints removed

        mixed_v2_out = mixed_v2_out.join(enc_block)

    cols_before = out.shape[1]
    out = out.join(mixed_v2_out)

    # after mixed v2 prints removed

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



    # DO NOT DELETE BELOW
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)



main()

#% 04 Compute Gower
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


IN_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"
OUT_NPY = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
OUT_META = PROJECT_ROOT / "data" / "out" / "drivers_gower_meta.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_gower_numeric_with_missing(
    X: np.ndarray,
    feature_ranges: np.ndarray,
    block_size: int = 256,
) -> np.ndarray:
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
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}")

    n, p = X.shape
    if feature_ranges.shape != (p,):
        raise ValueError(f"feature_ranges must have shape ({p},); got {feature_ranges.shape}")

    # Only non-constant, well-defined ranges contribute
    valid_feat = np.isfinite(feature_ranges) & (feature_ranges > 0)
    if valid_feat.sum() == 0:
        raise ValueError("No valid features (all ranges are 0 or NaN). Gower undefined.")

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


def main(force: bool = False, block_size: int = 256) -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input encoded drivers: {IN_PATH}")

    in_hash = sha256_file(IN_PATH)

    if OUT_NPY.exists() and OUT_META.exists() and not force:
        try:
            meta = json.loads(OUT_META.read_text(encoding="utf-8"))
            if meta.get("input_sha256") == in_hash:
                return
        except Exception:
            pass

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
    if np.isnan(D64).any():
        n_nan = int(np.isnan(D64).sum())
        raise ValueError(f"Distance matrix contains NaNs after missing-aware Gower: {n_nan} NaNs")

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

    # saved output message removed
    print("Saved metadata:", OUT_META)



main(force=False, block_size=256)

#% 05 PAM
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples

try:
    from sklearn_extra.cluster import KMedoids  # type: ignore
    _HAS_SKLEARN_EXTRA = True
except Exception:
    _HAS_SKLEARN_EXTRA = False

    class KMedoids:  # minimal fallback for precomputed distances
        def __init__(self, n_clusters, metric="precomputed", method="pam", init="k-medoids++", random_state=None, max_iter=100):
            if metric != "precomputed":
                raise ValueError("Fallback KMedoids only supports metric='precomputed'")
            if method != "pam":
                raise ValueError("Fallback KMedoids only supports method='pam'")
            self.n_clusters = n_clusters
            self.metric = metric
            self.method = method
            self.init = init
            self.random_state = random_state
            self.max_iter = max_iter
            self.medoid_indices_ = None

        def _init_kmedoids_pp(self, D, k, rng):
            n = D.shape[0]
            medoids = [rng.randint(0, n)]
            for _ in range(1, k):
                dist_to_nearest = np.min(D[:, medoids], axis=1)
                probs = dist_to_nearest ** 2
                total = probs.sum()
                if total == 0:
                    cand = rng.randint(0, n)
                else:
                    probs = probs / total
                    cand = rng.choice(np.arange(n), p=probs)
                medoids.append(int(cand))
            return np.array(medoids, dtype=int)

        def _update_medoids(self, D, labels, k):
            medoids = []
            for c in range(k):
                idx = np.flatnonzero(labels == c)
                if idx.size == 0:
                    medoids.append(0)
                    continue
                sub = D[np.ix_(idx, idx)]
                medoid_local = int(np.argmin(sub.sum(axis=1)))
                medoids.append(int(idx[medoid_local]))
            return np.array(medoids, dtype=int)

        def fit(self, D):
            D = np.asarray(D)
            n = D.shape[0]
            rng = np.random.RandomState(self.random_state)
            if self.init == "k-medoids++":
                medoids = self._init_kmedoids_pp(D, self.n_clusters, rng)
            else:
                medoids = rng.choice(np.arange(n), size=self.n_clusters, replace=False)

            labels = np.argmin(D[:, medoids], axis=1)
            for _ in range(self.max_iter):
                new_medoids = self._update_medoids(D, labels, self.n_clusters)
                new_labels = np.argmin(D[:, new_medoids], axis=1)
                if np.array_equal(new_medoids, medoids):
                    medoids = new_medoids
                    labels = new_labels
                    break
                medoids = new_medoids
                labels = new_labels

            self.medoid_indices_ = medoids
            self.labels_ = labels
            return self

        def fit_predict(self, D):
            self.fit(D)
            return self.labels_


# ==========================================================
# PAM (K-MEDOIDS) ON PRECOMPUTED GOWER DISTANCES
# ==========================================================

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
    try:
        from IPython.display import display, HTML
        display(HTML(df_table.to_html(index=False)))
    except Exception:
        pass

    # ----------------------------
    # V2 visuals
    # ----------------------------
    sil_plot_path = OUT_DIR / "silhouette_vs_k.png"
    _plot_silhouette_vs_k(df, sil_plot_path)
    try:
        from IPython.display import Image, display
        display(Image(filename=str(sil_plot_path)))
    except Exception:
        pass

    size_plot_path = OUT_DIR / "cluster_sizes_by_k.png"
    _plot_cluster_sizes_by_k(df, size_plot_path)
    try:
        from IPython.display import Image, display
        display(Image(filename=str(size_plot_path)))
    except Exception:
        pass



main()

#% 07 PCoA Visual
# pam_pcoa_k6.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
    ax.legend(
        markerscale=1.5,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    # Increase right margin and avoid tight bbox cropping
    fig.subplots_adjust(left=0.02, right=0.80, top=0.92, bottom=0.06)
    plt.savefig(save_path)
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
    try:
        from IPython.display import Image, display
        display(Image(filename=str(out2d)))
    except Exception:
        pass

    # 3D
    if coords.shape[1] >= 3:
        out3d = OUT_DIR / "pcoa_gower_k6_3d.png"
        plot_pcoa_3d(coords[:, :3], explained[:3], labels, out3d)
        try:
            from IPython.display import Image, display
            display(Image(filename=str(out3d)))
        except Exception:
            pass
    else:
        print("PCoA returned only 2 positive components; 3D plot skipped.")

    # no extra console summary
main()

#% 07B UMAP Visual
# pam_umap_k6.py

from umap import UMAP


UMAP_PAM_DIR = PROJECT_ROOT / "data" / "out" / "pam"
UMAP_OUT_DIR = PROJECT_ROOT / "data" / "out" / "pam_post"
UMAP_OUT_DIR.mkdir(parents=True, exist_ok=True)

UMAP_GOWER_PATH = PROJECT_ROOT / "data" / "out" / "drivers_gower.npy"
UMAP_LABELS_PATH = UMAP_PAM_DIR / "pam_labels_k6.npy"


def load_umap_gower_precomputed() -> np.ndarray:
    if not UMAP_GOWER_PATH.exists():
        raise FileNotFoundError(f"Missing Gower distance matrix: {UMAP_GOWER_PATH}")
    D = np.load(UMAP_GOWER_PATH)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape {D.shape}")
    return D.astype(float)


def load_umap_labels_k6() -> np.ndarray:
    if not UMAP_LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {UMAP_LABELS_PATH}")
    labels = np.load(UMAP_LABELS_PATH).astype(int)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {labels.shape}")
    return labels


def fit_umap_precomputed_gower_2d(
    D: np.ndarray,
    n_neighbors: int = 12,
    min_dist: float = 0.0,
    spread: float = 2.0,
    random_state: int = 42,
) -> np.ndarray:
    reducer = UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
    )
    emb = reducer.fit_transform(D)
    if emb.shape != (D.shape[0], 2):
        raise ValueError(f"Unexpected UMAP embedding shape: {emb.shape}, expected {(D.shape[0], 2)}")
    return emb


def plot_umap_2d(embedding: np.ndarray, labels: np.ndarray, save_path: Path) -> None:
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


def run_umap_k6_2d() -> np.ndarray:
    D = load_umap_gower_precomputed()
    labels = load_umap_labels_k6()

    if D.shape[0] != labels.shape[0]:
        raise ValueError(f"Row mismatch: D is {D.shape[0]}x{D.shape[1]} but labels has {labels.shape[0]}")

    emb = fit_umap_precomputed_gower_2d(
        D=D,
        n_neighbors=12,
        min_dist=0.0,
        spread=2.0,
        random_state=42,
    )

    out2d = UMAP_OUT_DIR / "umap_gower_k6_2d.png"
    plot_umap_2d(emb, labels, out2d)

    try:
        from IPython.display import Image, display
        display(Image(filename=str(out2d)))
    except Exception:
        pass

    return emb


UMAP_EMBEDDING_2D = run_umap_k6_2d()

#% 08 Post-Cluster Validate
# pam_post_validate.py

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
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


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
    if not _HAS_PLOTLY:
        print("Plotly not available; skipping Sankey diagram:", out_html)
        return
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
    if not _HAS_PLOTLY:
        print("Plotly not available; skipping Sankey diagram:", out_html)
        return
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

    # Always show an inline interactive version in notebook output
    try:
        from IPython.display import display, HTML
        display(HTML(fig.to_html(include_plotlyjs='cdn')))
    except Exception:
        pass


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
    try:
        from IPython.display import display, HTML
        display(HTML(ari_df.to_html(index=False)))
    except Exception:
        print(ari_df.to_string(index=False))

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
    try:
        from IPython.display import display, HTML
        display(HTML(show.to_html(index=False)))
    except Exception:
        print(show.to_string(index=False))

    # stability_summary_adjacent.csv output removed by request
    # Note: robustness_ari.csv output removed by request

    # -------------------------
    # Global Sankey only
    # -------------------------
    global_html = OUT_DIR / "sankey_global_k4_k5_k6_k7.html"
    make_sankey_global(KS, label_map, global_html)

    # Feature-importance outputs and distribution panels removed as requested.
main()

#% 09 K6 Inspect
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


from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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

def plot_heatmap(data, title, path, vmax=None, max_rows: Optional[int] = 30):

    if max_rows is not None and data.shape[0] > max_rows:
        data = data.loc[data.max(axis=1).sort_values(ascending=False).head(max_rows).index]

    plt.figure(figsize=(10, max(6, len(data) * 0.35)))
    sns.heatmap(data, cmap="viridis", vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _build_raw_pairwise_diff_matrix(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """
    Build pairwise absolute differences of answer proportions for raw features.
    Rows: feature=answer, Columns: C{i}-C{j} for all cluster pairs.
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


def main():

    df = load_encoded_df()
    labels = load_labels()

    feature = _find_feature_column(df)

    values = df[feature].values[labels == CLUSTER_ID]

    code_some = _get_code_for_label(feature, "Somewhat open") or 4
    code_very = _get_code_for_label(feature, "Very open") or 5

    pct_some, cnt_some, total = _pct_and_count_over_full(values, code_some)
    pct_very, cnt_very, _ = _pct_and_count_over_full(values, code_very)

    # prints removed
    # Pairwise medoid-diff heatmaps removed; keep only raw top-30 heatmap.
    # done message removed



main()

#% 10 PAM Cluster Profiles K6
# pam_cluster_profiles_k6_console.py

from pathlib import Path
import numpy as np
import pandas as pd

# Import ordinal maps directly from encoding.py

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


def plot_raw_feature_panel(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    features: list[str],
    out_path: Path,
    suptitle: str,
    ncols: int = 3,
    color_map: Optional[dict[str, str]] = None,
    recode_map: Optional[dict[str, "Callable[[pd.Series], pd.Series]"]] = None,
    legend_mode: str = "shared",
    legend_order: Optional[list[str]] = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = len(features)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.6 * nrows), dpi=150)
    axes = np.asarray(axes).ravel()

    clusters = sorted(np.unique(labels_aligned).tolist())
    x = np.arange(len(clusters))

    # Build a global category list across all features to ensure legend completeness (shared legend mode).
    all_categories: list[str] = []
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
    color_for_cat: dict[str, str] = {}
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
            ax.bar(x, mat[:, j], bottom=bottom, label=str(cat), color=color)
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
    plt.savefig(out_path, bbox_inches="tight")
    try:
        from IPython.display import Image, display
        # panel display message removed
        display(Image(filename=str(out_path)))
    except Exception:
        pass
    plt.close()




def make_all_raw_feature_plots(df_raw: pd.DataFrame, labels_aligned: np.ndarray) -> None:
    """
    Creates one PNG per raw feature in df_raw (excluding respondent_id).
    """
    # raw plot header messages removed

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

        def recode_benefits(s: pd.Series) -> pd.Series:
            s = s.astype(object)
            s = s.where(~pd.isna(s), other="NA")
            s = s.astype(str).str.strip()
            s = s.replace({"": "NA"})
            return s.replace(
                {"Not eligible for coverage / N/A": "No or not eligible for coverage", "No": "No or not eligible for coverage"}
            )

        def recode_leave_easy(s: pd.Series) -> pd.Series:
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

        def recode_prev_answers(s: pd.Series) -> pd.Series:
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

        def recode_stigma(s: pd.Series) -> pd.Series:
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


def build_raw_cluster_vs_rest_tables(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    top_n: int = 15,
    exclude_answer_substrings: dict[int, list[str]] | None = None,
) -> dict[int, pd.DataFrame]:
    """
    For each cluster, compute top-N raw feature-answer categories by absolute
    deviation in proportion vs the rest of the sample.
    Returns: {cluster_id: DataFrame}
    """
    features = [c for c in df_raw.columns if c != "respondent_id"]
    clusters = sorted(np.unique(labels_aligned).tolist())
    results: dict[int, pd.DataFrame] = {}

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
            def _keep_feature(x: str) -> bool:
                lx = x.lower()
                return not any(sub in lx for sub in subs)
            df_all = df_all[df_all["feature"].map(_keep_feature)]
        df = df_all.head(top_n).drop(columns="_diff")
        results[int(c)] = df.reset_index(drop=True)

    return results


def build_raw_cluster_vs_rest_selected(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    cluster_id: int,
    selected_features: list[str],
) -> pd.DataFrame:
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


def build_raw_pairwise_tables(
    df_raw: pd.DataFrame,
    labels_aligned: np.ndarray,
    pairs: list[tuple[int, int]],
    top_n: int = 15,
) -> dict[tuple[int, int], pd.DataFrame]:
    """
    For each cluster pair, compute top-N raw feature-answer categories by absolute
    deviation in proportion between the two clusters.
    Returns: {(c1, c2): DataFrame}
    """
    features = [c for c in df_raw.columns if c != "respondent_id"]
    results: dict[tuple[int, int], pd.DataFrame] = {}

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


def pam_cluster_profiles_main() -> None:
    global feature_names
    X, feature_names = load_encoded_drivers()
    labels = load_labels()

    binary_feats, ordinal_feats = split_feature_types(X, feature_names)

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
    try:
        from IPython.display import display, HTML
        display(HTML("<b>k=6 Cluster Sizes</b>" + size_df.to_html(index=False)))
    except Exception:
        print(size_df.to_string(index=False))

    # ------------------------
    # NEW: Raw (pre-encoding) plots by cluster
    # ------------------------
    df_raw = load_preprocessed_drivers_raw()
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

    try:
        from IPython.display import Image, display
        # heatmap display message removed
        display(Image(filename=str(out_top30)))
    except Exception:
        pass

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
            try:
                from IPython.display import display, HTML
                display(HTML("<b>CLUSTER 0 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))
            except Exception:
                print(raw_tables[c].to_string(index=False))

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
                try:
                    from IPython.display import display, HTML
                    display(HTML("<b>CLUSTER 0 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
                except Exception:
                    print(forced_df.to_string(index=False))
        elif c == 1:
            try:
                from IPython.display import display, HTML
                display(HTML("<b>CLUSTER 1 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))
            except Exception:
                print(raw_tables[c].to_string(index=False))

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
                try:
                    from IPython.display import display, HTML
                    display(HTML("<b>CLUSTER 1 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
                except Exception:
                    print(forced_df.to_string(index=False))
        elif c == 2:
            try:
                from IPython.display import display, HTML
                display(HTML("<b>CLUSTER 2 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))
            except Exception:
                print(raw_tables[c].to_string(index=False))

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
                try:
                    from IPython.display import display, HTML
                    display(HTML("<b>CLUSTER 2 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
                except Exception:
                    print(forced_df.to_string(index=False))
        elif c == 3:
            try:
                from IPython.display import display, HTML
                display(HTML("<b>CLUSTER 3 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))
            except Exception:
                print(raw_tables[c].to_string(index=False))

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
                try:
                    from IPython.display import display, HTML
                    display(HTML("<b>CLUSTER 3 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
                except Exception:
                    print(forced_df.to_string(index=False))
        elif c == 4:
            try:
                from IPython.display import display, HTML
                display(HTML("<b>CLUSTER 4 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))
            except Exception:
                print(raw_tables[c].to_string(index=False))

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
                try:
                    from IPython.display import display, HTML
                    display(HTML("<b>CLUSTER 4 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
                except Exception:
                    print(forced_df.to_string(index=False))
        elif c == 5:
            try:
                from IPython.display import display, HTML
                display(HTML("<b>CLUSTER 5 vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))
            except Exception:
                print(raw_tables[c].to_string(index=False))

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
                try:
                    from IPython.display import display, HTML
                    display(HTML("<b>CLUSTER 5 vs REST (Selected features)</b>" + forced_df.to_html(index=False)))
                except Exception:
                    print(forced_df.to_string(index=False))
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
        try:
            from IPython.display import display, HTML
            display(HTML("<b>" + title + "</b>" + pair_tables[(c1, c2)].to_html(index=False)))
        except Exception:
            print(pair_tables[(c1, c2)].to_string(index=False))


pam_cluster_profiles_main()

#%% 11 K6 Cluster Overlays
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pam_overlay_profiles_k6_plots_cleaned.py

PAM overlay cluster-profile plots (k=6), with cleanup / recoding for:
- gender -> {Male, Female, Transgender Male, Transgender Female, Nonbinary / Gender Diverse, Other, Prefer Not to Say / Invalid Response}
- age -> custom bins:
    "<20", "20-25", "26-30", "31-35", "36-40", "41-50", "51+"
  (outliers clipped to nearest bin edge)
- country_live -> coarse regions (US, UK, Canada, European Union, Other Europe, Asia, Oceania, Latin America & Caribbean, Africa, Middle East, Other/NA)
- med_pro_condition -> {No diagnosis/NA, Mood disorder, anxiety disorder, other}
  (Mood disorder if both words "mood" and "disorder" appear; anxiety disorder if both "anxiety" and "disorder" appear)
- work_position -> keyword + hierarchy based recode into:
    "Sales", "Dev Evangelist/Advocate", "Leadership (Supervisor/Exec)",
    "DevOps/SysAdmin", "Support", "Full-stack (FE+BE)",
    "Front-end Developer", "Back-end Developer", "Designer", "other", "NA"
  with hierarchy:
    Sales > Dev Evangelist/Advocate > Leadership > DevOps/SysAdmin > Support >
    (FE+BE) > FE-only/BE-only/Designer-only > other

Outputs ONLY plots (stacked proportions by cluster).
No extra CSVs are produced.

Usage:
  python pam_overlay_profiles_k6_plots_cleaned.py

Override paths if needed:
  python pam_overlay_profiles_k6_plots_cleaned.py --overlays ... --labels ... --encoded-drivers ... --outdir ...
"""


import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Helpers: string normalization
# -----------------------------

def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("\u00a0", " ")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_missing_like(s: str) -> bool:
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

def recode_gender(val) -> str:
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

def recode_age(val) -> str:
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

def recode_country_region(val) -> str:
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

def recode_med_condition(val) -> str:
    s = norm_text(val)
    if is_missing_like(s):
        return "No diagnosis/NA"

    # treat explicit no-diagnosis signals
    if any(k in s for k in ["no diagnosis", "no dx", "none", "healthy", "no condition", "no mental"]):
        return "No diagnosis/NA"

    # apply only if BOTH words appear (as requested)
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

def _contains_any(s: str, needles: Iterable[str]) -> bool:
    return any(n in s for n in needles)

def recode_work_position(val) -> str:
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

def align_labels_to_overlays(df_over: pd.DataFrame, labels: np.ndarray, encoded_drivers_csv: Path) -> np.ndarray:
    if "respondent_id" in df_over.columns and encoded_drivers_csv.exists():
        df_enc = pd.read_csv(encoded_drivers_csv)
        if "respondent_id" in df_enc.columns and len(df_enc) == len(labels):
            id_to_label = pd.Series(labels, index=df_enc["respondent_id"]).to_dict()
            mapped = df_over["respondent_id"].map(id_to_label).to_numpy()
            if np.any(pd.isna(mapped)):
                missing = int(pd.isna(mapped).sum())
                missing_ids = df_over.loc[pd.isna(mapped), "respondent_id"].head(10).tolist()
                raise ValueError(f"Could not align {missing} overlay rows via respondent_id. Examples: {missing_ids}")
            return mapped.astype(int)

    if len(df_over) != len(labels):
        raise ValueError(f"Row mismatch overlays={len(df_over)} vs labels={len(labels)}; cannot align without respondent_id mapping.")
    return labels.astype(int)


# -----------------------------
# Plotting: stacked proportions
# -----------------------------

def safe_name(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace("\n", "_")
        .strip()
    )


def plot_categorical_stacked_by_cluster(
    series: pd.Series,
    labels: np.ndarray,
    title: str,
    out_path: Optional[Path] = None,
    category_order: Optional[List[str]] = None,
    legend_title: str = "Answer",
    min_prop_to_keep: float = 0.0,
    max_categories: Optional[int] = None,
    ax: Optional["plt.Axes"] = None,
) -> None:
    import matplotlib.pyplot as plt

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
        ax.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    ax.set_xticks(x, [str(c) for c in clusters])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion within cluster")
    ax.set_title(title)
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)


def dominant_answer_table(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
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


def cluster_answer_pct_table(
    series: pd.Series,
    labels: np.ndarray,
) -> pd.DataFrame:
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

    ax.legend(
        title=legend_title,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        ncol=1,
    )

    if out_path is not None:
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


# -----------------------------
# Main: apply recodes + plot
# -----------------------------

def apply_recodes(df_over: pd.DataFrame) -> pd.DataFrame:
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


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    project_root_guess = here.parents[2] if len(here.parents) >= 3 else Path.cwd()

    default_overlays = project_root_guess / "data" / "out" / "overlays_clean.csv"
    default_labels = project_root_guess / "data" / "out" / "pam" / "pam_labels_k6.npy"
    default_encoded = project_root_guess / "data" / "out" / "drivers_encoded.csv"
    default_outdir = project_root_guess / "data" / "out" / "pam_post" / "overlay_distributions_k6_cleaned"

    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", type=Path, default=default_overlays, help="Path to overlays_clean.csv")
    ap.add_argument("--labels", type=Path, default=default_labels, help="Path to pam_labels_k6.npy")
    ap.add_argument("--encoded-drivers", type=Path, default=default_encoded, help="Path to drivers_encoded.csv (for respondent_id alignment)")
    ap.add_argument("--outdir", type=Path, default=default_outdir, help="Output directory for PNG plots")
    ap.add_argument("--k", type=int, default=6, help="k used for clustering (used in titles/filenames)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.overlays.exists():
        raise FileNotFoundError(f"Missing overlays file: {args.overlays}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Missing labels file: {args.labels}")

    df_over = pd.read_csv(args.overlays)
    if "respondent_id" not in df_over.columns:
        raise ValueError("overlays CSV must have a respondent_id column for safe alignment.")

    labels = np.load(args.labels).astype(int)
    labels_aligned = align_labels_to_overlays(df_over, labels, args.encoded_drivers)

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

    # overlay plots skip message removed

    # Combined 2x2 panel for key overlay features (age, gender, country_work, work_position)
    panel_feats = ["age", "gender", "country_work", "work_position"]
    if all(f in df.columns for f in panel_feats):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
        axes = axes.ravel()

        # age
        plot_categorical_stacked_by_cluster(
            series=df["age"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): age",
            out_path=None,
            category_order=AGE_LABELS + ["NA"],
            legend_title="Answer",
            ax=axes[0],
        )

        # gender
        plot_categorical_stacked_by_cluster(
            series=df["gender"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): gender",
            out_path=None,
            category_order=GENDER_TARGET_ORDER,
            legend_title="Answer",
            ax=axes[1],
        )

        # country_work (use top categories + Other)
        plot_categorical_stacked_by_cluster(
            series=df["country_work"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): country_work",
            out_path=None,
            legend_title="Answer",
            max_categories=15,
            min_prop_to_keep=0.01,
            ax=axes[2],
        )

        # work_position
        plot_categorical_stacked_by_cluster(
            series=df["work_position"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): work_position",
            out_path=None,
            category_order=WORK_ORDER,
            legend_title="Answer",
            ax=axes[3],
        )

        panel_path = args.outdir / f"overlay_k{args.k}_panel_age_gender_country_work_position.png"
        plt.tight_layout()
        plt.savefig(panel_path, bbox_inches="tight")
        try:
            from IPython.display import Image, display
            # overlay panel display message removed
            display(Image(filename=str(panel_path)))
        except Exception:
            pass
        plt.close()
    else:
        missing = [f for f in panel_feats if f not in df.columns]
        print(f"Skipping overlay panel (missing columns: {missing})")

    # Combined 2x2 panel for key overlay features (mhd_past, current_mhd, pro_treatment, no_treat_mhd_bad_work)
    panel_feats = ["mhd_past", "current_mhd", "pro_treatment", "no_treat_mhd_bad_work"]
    if all(f in df.columns for f in panel_feats):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
        axes = axes.ravel()

        # mhd_past
        plot_categorical_stacked_by_cluster(
            series=df["mhd_past"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): mhd_past",
            out_path=None,
            legend_title="Answer",
            ax=axes[0],
        )

        # current_mhd
        plot_categorical_stacked_by_cluster(
            series=df["current_mhd"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): current_mhd",
            out_path=None,
            legend_title="Answer",
            ax=axes[1],
        )

        # pro_treatment
        plot_categorical_stacked_by_cluster(
            series=df["pro_treatment"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): pro_treatment",
            out_path=None,
            legend_title="Answer",
            ax=axes[2],
        )

        # no_treat_mhd_bad_work
        plot_categorical_stacked_by_cluster(
            series=df["no_treat_mhd_bad_work"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): no_treat_mhd_bad_work",
            out_path=None,
            legend_title="Answer",
            ax=axes[3],
        )

        panel_path = args.outdir / f"overlay_k{args.k}_panel_mhd_past_current_mhd_pro_treatment_no_treat_mhd_bad_work.png"
        plt.tight_layout()
        plt.savefig(panel_path, bbox_inches="tight")
        try:
            from IPython.display import Image, display
            # overlay panel display message removed
            display(Image(filename=str(panel_path)))
        except Exception:
            pass
        plt.close()
    else:
        missing = [f for f in panel_feats if f not in df.columns]
        print(f"Skipping overlay panel (missing columns: {missing})")

    # Combined 2x2 panel for key overlay features (mhd_hurt_career, coworkers_view_neg_mhd, med_pro_condition, mhd_by_med_pro)
    panel_feats = ["mhd_hurt_career", "coworkers_view_neg_mhd", "med_pro_condition", "mhd_by_med_pro"]
    if all(f in df.columns for f in panel_feats):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
        axes = axes.ravel()

        # mhd_hurt_career
        plot_categorical_stacked_by_cluster(
            series=df["mhd_hurt_career"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): mhd_hurt_career",
            out_path=None,
            legend_title="Answer",
            ax=axes[0],
        )

        # coworkers_view_neg_mhd
        plot_categorical_stacked_by_cluster(
            series=df["coworkers_view_neg_mhd"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): coworkers_view_neg_mhd",
            out_path=None,
            legend_title="Answer",
            ax=axes[1],
        )

        # med_pro_condition (uses recoded categories)
        plot_categorical_stacked_by_cluster(
            series=df["med_pro_condition"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): med_pro_condition",
            out_path=None,
            category_order=MED_GROUPS_ORDER,
            legend_title="Answer",
            ax=axes[2],
        )

        # mhd_by_med_pro
        plot_categorical_stacked_by_cluster(
            series=df["mhd_by_med_pro"],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): mhd_by_med_pro",
            out_path=None,
            legend_title="Answer",
            ax=axes[3],
        )

        panel_path = args.outdir / f"overlay_k{args.k}_panel_mhd_hurt_career_coworkers_view_neg_mhd_med_pro_condition_mhd_by_med_pro.png"
        plt.tight_layout()
        plt.savefig(panel_path, bbox_inches="tight")
        try:
            from IPython.display import Image, display
            # overlay panel display message removed
            display(Image(filename=str(panel_path)))
        except Exception:
            pass
        plt.close()
    else:
        missing = [f for f in panel_feats if f not in df.columns]
        print(f"Skipping overlay panel (missing columns: {missing})")

    # Dominant answer table for key overlay features (with percentages)
    dom_feats = ["mhd_by_med_pro", "pro_treatment", "mhd_past", "current_mhd", "med_pro_condition"]
    if all(f in df.columns for f in dom_feats):
        dom_df = dominant_answer_table(df, labels_aligned, dom_feats)
        try:
            from IPython.display import display, HTML
            display(HTML("<b>Dominant answers (overlays)</b>" + dom_df.to_html(index=False)))
        except Exception:
            print(dom_df.to_string(index=False))

    # Cluster × Answer percentage tables for selected overlay features
    pct_feats = ["no_treat_mhd_bad_work", "mhd_hurt_career", "coworkers_view_neg_mhd"]
    for feat in pct_feats:
        if feat not in df.columns:
            continue
        pct_df = cluster_answer_pct_table(df[feat], labels_aligned)
        try:
            from IPython.display import display, HTML
            display(HTML(f"<b>{feat}</b>" + pct_df.to_html(index=False)))
        except Exception:
            print(pct_df.to_string(index=False))

    # done message removed


# Override parse_args to avoid notebook argv issues

def parse_args() -> argparse.Namespace:
    project_root_guess = PROJECT_ROOT

    default_overlays = project_root_guess / "data" / "out" / "overlays_clean.csv"
    default_labels = project_root_guess / "data" / "out" / "pam" / "pam_labels_k6.npy"
    default_encoded = project_root_guess / "data" / "out" / "drivers_encoded.csv"
    default_outdir = project_root_guess / "data" / "out" / "pam_post" / "overlay_distributions_k6_cleaned"

    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", type=Path, default=default_overlays, help="Path to overlays_clean.csv")
    ap.add_argument("--labels", type=Path, default=default_labels, help="Path to pam_labels_k6.npy")
    ap.add_argument("--encoded-drivers", type=Path, default=default_encoded, help="Path to drivers_encoded.csv (for respondent_id alignment)")
    ap.add_argument("--outdir", type=Path, default=default_outdir, help="Output directory for PNG plots")
    ap.add_argument("--k", type=int, default=6, help="k used for clustering (used in titles/filenames)")
    return ap.parse_args(args=[])

main()
