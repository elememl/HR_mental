#%% 00 Config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#% 01 Rename Columns
import pandas as pd
from pathlib import Path

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

print("Renamed file saved to:", OUT_PATH)
print("\nNew columns:")
print(list(df_renamed.columns))

#% 02 Cleaning + Drivers + Overlays
from typing import Tuple
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

    _log_change("Step 1 (remove self-employed)", before, df2)
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

    _log_change("Step 2 (remove explicit non-tech roles)", before, df2)
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
    print("Subset size:", n_subset)
    print("Are ALL non-tech-company rows tech-role workers?", (n_subset == n_tech_role_yes))

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

    print("\nRow missingness summary (fraction missing):")
    print(row_miss.describe()[["min", "25%", "50%", "75%", "max"]].round(3))

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
    )

    top15 = miss_table.head(15).copy()
    # Format % nicely for printing
    top15["missing_percent"] = top15["missing_percent"].map(lambda x: f"{x:.2f}%")

    print("\nTop 15 columns by missingness:")
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

    # Bar plot: missingness per feature (sorted)
    col_miss_sorted = col_miss.sort_values(ascending=False)
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

    print("\nDropped columns with 100% missingness:")
    print(list(fully_missing_cols))
    print(
        f"\nRemoved {cols_before - cols_after} columns and {rows_before - rows_after} rows "
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
    print("Saved overlays:", overlay_path)

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
    df_drivers = df_drivers.drop(columns=drop_present)

    print("Dropped open-ended features:", drop_present)
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

    # Show as image-table (like your plots)
    show_table(audit_df, "Driver feature audit: missingness / unique / dominant share", max_rows=30, figsize=(12, 8))

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

    # Show as image-table too
    show_table(corr_pairs_df, "Strong missingness-correlation pairs (|corr| >= 0.95)", max_rows=30, figsize=(12, 8))

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

    print("\nFULL contingency table (including NaN & Not applicable):")
    print(full_ct)

    # apply parent-child question changes
    df_drivers = apply_driver_skip_logic(df_drivers)


    # contigency table after applying rule C and enforcing 'non applicable'
    print("\nFULL contingency table AFTER Rule C (including NaN & Not applicable):")

    ct_after = pd.crosstab(
        df_drivers["benefits"],
        df_drivers["mh_options_known"],
        dropna=False
    )

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
        show_table(
            pd.DataFrame([{
                "result": "INFO",
                "message": "No skip-logic rules matched columns in df_drivers (nothing to prove)."
            }]),
            "Structural missingness proof — INFO",
            max_rows=5,
            figsize=(12, 2.5)
        )
    else:
        # compact and readable
        cols = ["rule", "child", "rows_where_not_applicable", "NA_token_count", "NaN_count", "Other_value_count",
                "PASS"]
        show_table(
            proof_metrics_df[cols].sort_values(["PASS", "rows_where_not_applicable"],
                                               ascending=[True, False]).reset_index(drop=True),
            "Structural missingness proof — metrics per rule/child",
            max_rows=50,
            figsize=(16, 6)
        )

    # ---------- TABLE 1B: failures only (only if needed) ----------
    fail_df = pd.DataFrame(proof_failures)
    if not fail_df.empty:
        show_table(
            fail_df[["rule", "child", "rows_where_not_applicable", "NA_token_count", "NaN_count", "Other_value_count",
                     "PASS"]]
            .reset_index(drop=True),
            "Structural missingness proof — FAILURES (investigate)",
            max_rows=50,
            figsize=(16, 5)
        )

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

        show_table(true_miss_df, "True missingness (NaN only) — nonzero features", max_rows=50, figsize=(14, 6))

        # Dependency situation (NO threshold): show top 20 |corr| among NaN indicators
        miss_bin = drivers[miss_nan.index].isna().astype(int)

        if miss_bin.shape[1] >= 2:
            corr = miss_bin.corr()
            pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index()
            )
            pairs.columns = ["feature_1", "feature_2", "corr"]
            pairs["abs_corr"] = pairs["corr"].abs()
            pairs = pairs.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"]).reset_index(drop=True)

            show_table(pairs.head(20), "True-missingness dependencies — top 20 |corr| (no threshold)", max_rows=20,
                       figsize=(14, 6))
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
    print("\n=== BINARY CHECK (AFTER standardization) ===")
    for col in binary_cols:
        if col in df_drivers.columns:
            print(col)
            print(df_drivers[col].value_counts(dropna=False))
            print("dtype:", df_drivers[col].dtype)

    # ============================================================
    # POST-FIX VALIDATION CHECKS
    # ============================================================

    print("\n=== POST-FIX VALIDATION ===")

    # ---------- 1) Confirm no true NaN remains (except respondent_id) ----------

    nan_counts = df_drivers.drop(columns=["respondent_id"], errors="ignore").isna().sum()
    nan_remaining = nan_counts[nan_counts > 0]

    print("\nRemaining NaN counts per feature:")
    print(nan_remaining if not nan_remaining.empty else "NONE (all true missingness handled)")

    # ---------- 2) Updated contingency table (benefits vs mh_options_known) ----------

    print("\nContingency table AFTER full missingness handling:")

    ct_final = pd.crosstab(
        df_drivers["benefits"],
        df_drivers["mh_options_known"],
        dropna=False
    )

    print(ct_final)

    # ---------- 3) Value distributions for previously missing features ----------

    for col in ["ever_observed_mhd_bad_response", "mhdcoworker_you_not_reveal", "mh_options_known"]:
        if col in df_drivers.columns:
            print(f"\nValue counts for {col}:")
            print(df_drivers[col].value_counts(dropna=False))

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

        # get example values (up to 5 most common)
        examples = s.value_counts(dropna=False).head(5).index.tolist()

        # number of unique observed values
        n_unique = s.nunique(dropna=True)

        # dominant share
        dominant_share = round((s.value_counts().iloc[0] / len(s)) * 100, 2)

        audit_rows.append({
            "feature": col,
            "example_values": examples,
            "n_unique": n_unique,
            "dominant_share_%": dominant_share,
        })

    audit_table = pd.DataFrame(audit_rows)

    print(audit_table)

    # ---------------------------------------------------------------------
    # Save drivers
    # ---------------------------------------------------------------------
    out_path = PROJECT_ROOT / "data" / "out" / "survey_drivers.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_drivers.to_csv(out_path, index=False)
    print("Saved drivers:", out_path)


main()

#%% 03 Encoding
from pathlib import Path
import pandas as pd

IN_PATH = PROJECT_ROOT / "data" / "out" / "survey_drivers.csv"
OUT_PATH = PROJECT_ROOT / "data" / "out" / "drivers_encoded.csv"

print("IN_PATH:", IN_PATH)
print("Exists?", IN_PATH.exists())

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

    # ===== RECORD 0: INPUT SHAPE =====
    print("\n=== INPUT: survey_drivers.csv ===")
    print("rows, cols:", df.shape)
    print("driver features (excluding respondent_id):", df.shape[1] - 1)

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

    print("\n=== AFTER BINARY ===")
    print("binary_cols kept:", binary_cols)
    print("out rows, cols:", out.shape)

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

    print("\n=== NOMINAL INPUT FEATURES ===")
    print("nominal_cols used:", nominal_cols)
    print("count nominal original features:", len(nominal_cols))

    X_nom = df[nominal_cols].copy()
    X_nom = X_nom.fillna("NaN")
    for c in nominal_cols:
        X_nom[c] = X_nom[c].astype(str).str.strip()

    nom_dum = pd.get_dummies(X_nom, prefix=nominal_cols, prefix_sep="=")

    print("\n=== NOMINAL ONE-HOT OUTPUT ===")
    print("total one-hot columns created from nominal:", nom_dum.shape[1])

    print("\nNominal feature expansion details:")
    for c in nominal_cols:
        created_cols = [col for col in nom_dum.columns if col.startswith(f"{c}=")]
        categories = [col.split("=", 1)[1] for col in created_cols]
        print(f"- {c}: {len(created_cols)} columns -> {categories}")

    cols_before = out.shape[1]
    out = out.join(nom_dum)
    print("\n=== AFTER NOMINAL ===")
    print("columns added:", out.shape[1] - cols_before)
    print("out rows, cols:", out.shape)

    # -------------------
    # 3) Ordinal
    # -------------------
    print("\n=== ORDINAL INPUT FEATURES ===")
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

    print("ordinal_cols used:", ordinal_cols)
    print("count ordinal original features:", len(ordinal_cols))

    ord_out = pd.DataFrame(index=df.index)

    for c in ordinal_cols:
        if c == "company_size":
            print("- company_size -> company_size_ord (robust)")
            raw_vals = sorted(df["company_size"].dropna().astype(str).map(norm_str).unique().tolist())
            print("  observed categories (raw):", raw_vals)
            ord_out["company_size_ord"] = encode_company_size(df["company_size"])
            print("  mapping used (text bins):", ORDINAL_MAPS["company_size"])
        else:
            mapping = ORDINAL_MAPS[c]
            s = df[c].map(norm_str).astype("string")
            print(f"- {c} -> {c}_ord")
            print("  observed categories:", sorted(s.dropna().unique().tolist()))
            print("  mapping used:", mapping)
            ord_out[f"{c}_ord"] = encode_ordinal(df[c], mapping, c)

    cols_before = out.shape[1]
    out = out.join(ord_out)

    print("\n=== AFTER ORDINAL ENCODING ===")
    print("ordinal encoded columns created:", ord_out.shape[1])
    print("columns added:", out.shape[1] - cols_before)
    print("out rows, cols:", out.shape)

    # -------------------
    # 4) Mixed (v1): ordinal + (NA/IDK) flags
    # -------------------
    print("\n=== MIXED v1 (ORDINAL + FLAGS) INPUT FEATURES ===")

    mixed_cols = [c for c in MIXED_SPECS.keys() if c in df.columns]
    print("mixed_cols used:", mixed_cols)
    print("count mixed original features:", len(mixed_cols))

    mixed_out = pd.DataFrame(index=df.index)

    for c in mixed_cols:
        spec = MIXED_SPECS[c]
        print(f"\n- {c}")

        raw = df[c].map(norm_str).astype("string")
        obs = sorted(raw.dropna().unique().tolist())
        print("  observed categories (raw):", obs)

        if spec["collapse_map"]:
            print("  collapse_map used:", spec["collapse_map"])

        print("  ordinal mapping used:", spec["ord_mapping"])
        if spec["na_tokens"]:
            print("  NA tokens:", sorted(list(spec["na_tokens"])))
        if spec["idk_tokens"]:
            print("  IDK tokens:", sorted(list(spec["idk_tokens"])))

        enc_block = encode_mixed_ord_plus_flags(
            df=df,
            feature=c,
            ord_mapping=spec["ord_mapping"],
            na_tokens=spec["na_tokens"],
            idk_tokens=spec["idk_tokens"],
            collapse_map=spec["collapse_map"],
            drop_all_zero_flags=True,
        )

        print("  encoded columns created:", list(enc_block.columns))

        preview = (
            pd.concat([raw.rename(c), enc_block], axis=1)
            .dropna(subset=[c])
            .drop_duplicates()
            .head(10)
        )
        print("  preview (raw -> encoded):")
        print(preview)

        mixed_out = mixed_out.join(enc_block)

    cols_before = out.shape[1]
    out = out.join(mixed_out)

    print("\n=== AFTER MIXED v1 ENCODING ===")
    print("mixed v1 encoded columns created:", mixed_out.shape[1])
    print("columns added:", out.shape[1] - cols_before)
    print("out rows, cols:", out.shape)

    # -------------------
    # 5) Mixed (v2): ordinal + TWO nominal flags (plus optional extra flags)
    #     (inspected separately, as requested)
    # -------------------
    print("\n=== MIXED v2 (ORDINAL + 2 NOMINAL FLAGS) INPUT FEATURES ===")

    mixed_v2_cols = [c for c in MIXED_SPECS_V2.keys() if c in df.columns]
    print("mixed_v2_cols used:", mixed_v2_cols)
    print("count mixed v2 original features:", len(mixed_v2_cols))

    mixed_v2_out = pd.DataFrame(index=df.index)

    for c in mixed_v2_cols:
        spec = MIXED_SPECS_V2[c]
        print(f"\n- {c}")

        raw = df[c].map(norm_str).astype("string")
        obs = sorted(raw.dropna().unique().tolist())
        print("  observed categories (raw):", obs)

        if spec.get("collapse_map"):
            print("  collapse_map used:", spec["collapse_map"])

        print("  ordinal mapping used:", spec["ord_mapping"])
        if spec.get("na_tokens"):
            print("  NA tokens:", sorted(list(spec["na_tokens"])))
        if spec.get("idk_tokens"):
            print("  IDK tokens:", sorted(list(spec["idk_tokens"])))
        if spec.get("extra_flag_tokens"):
            print("  extra flag tokens:", spec["extra_flag_tokens"])

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

        print("  encoded columns created:", list(enc_block.columns))

        preview = (
            pd.concat([raw.rename(c), enc_block], axis=1)
            .dropna(subset=[c])
            .drop_duplicates()
            .head(10)
        )
        print("  preview (raw -> encoded):")
        print(preview)

        mixed_v2_out = mixed_v2_out.join(enc_block)

    cols_before = out.shape[1]
    out = out.join(mixed_v2_out)

    print("\n=== AFTER MIXED v2 ENCODING ===")
    print("mixed v2 encoded columns created:", mixed_v2_out.shape[1])
    print("columns added:", out.shape[1] - cols_before)
    print("out rows, cols:", out.shape)

    # ==========================================================
    # MISSINGNESS SUMMARY (encoded columns)
    # ==========================================================
    print("\n=== MISSINGNESS SUMMARY (ENCODED) ===")

    def prevalence_label(pct: float) -> str:
        # Use simple bins; edit thresholds if you want
        if pct < 5:
            return "rare"
        elif pct < 30:
            return "moderate"
        else:
            return "dominant"

    def informativeness_label(observed_pct: float, dominant_share_pct: float) -> str:
        # observed_pct is % non-NaN for _ord columns
        if observed_pct < 20:
            return "mostly unknown"
        if dominant_share_pct >= 95:
            return "uninformative (almost constant)"
        if dominant_share_pct >= 80:
            return "weak signal (very skewed)"
        return "informative"

    n = len(out)

    # --------------------------
    # 1) _ord columns: %NaN + dominance among observed
    # --------------------------
    ord_cols = [c for c in out.columns if c.endswith("_ord")]
    ord_rows = []

    for c in ord_cols:
        s = out[c]
        pct_nan = (s.isna().sum() / n) * 100
        observed = s.dropna()

        if len(observed) == 0:
            dominant_share = 100.0
            observed_pct = 0.0
        else:
            dominant_share = (observed.value_counts().iloc[0] / len(observed)) * 100
            observed_pct = 100.0 - pct_nan

        ord_rows.append({
            "feature": c,
            "%NaN": round(pct_nan, 2),
            "NaN_level": prevalence_label(pct_nan),
            "dominant_share_observed_%": round(dominant_share, 2),
            "informativeness": informativeness_label(observed_pct, dominant_share),
        })

    ord_summary = pd.DataFrame(ord_rows).sort_values("%NaN", ascending=False)
    if ord_summary.empty:
        print("No _ord columns found.")
    else:
        print("\n--- _ord columns (NaN + informativeness) ---")
        print(ord_summary.to_string(index=False))

    # --------------------------
    # 2) Flag columns: frequency of 1s
    # --------------------------
    flag_cols = [c for c in out.columns if ("__na" in c) or ("__idk" in c) or ("__no_response" in c)]
    flag_rows = []

    for c in flag_cols:
        s = pd.to_numeric(out[c], errors="coerce").fillna(0)
        pct_one = (s.sum() / n) * 100
        flag_rows.append({
            "feature": c,
            "%==1": round(pct_one, 2),
            "flag_level": prevalence_label(pct_one),
        })

    flag_summary = pd.DataFrame(flag_rows).sort_values("%==1", ascending=False)
    if flag_summary.empty:
        print("\nNo flag columns (__na/__idk/__no_response) found.")
    else:
        print("\n--- Flag columns (prevalence of special tokens) ---")
        print(flag_summary.to_string(index=False))
    print("\n=== QUICK CHECK ===")
    ord_cols = [c for c in out.columns if c.endswith("_ord")]
    if ord_cols:
        pct_nan = (out[ord_cols].isna().mean() * 100).round(2).sort_values(ascending=False)
        print("\n% NaN per _ord column:")
        print(pct_nan)

    flag_cols = [c for c in out.columns if ("__na" in c) or ("__idk" in c) or ("__no_response" in c)]
    if flag_cols:
        pct_one = (out[flag_cols].mean() * 100).round(2).sort_values(ascending=False)
        print("\n% == 1 per flag column:")
        print(pct_one)



    # ===== FINAL SHAPE =====
    print("\n=== FINAL ENCODED MATRIX ===")
    print("out rows, cols:", out.shape)
    print("encoded features (excluding respondent_id):", out.shape[1] - 1)

    # ----------------------------------------------------------
    # DROP LOW-INFORMATION FLAG COLUMNS
    # ----------------------------------------------------------
    drop_cols = [
        "ever_observed_mhd_bad_response__no_response",
    ]

    drop_present = [c for c in drop_cols if c in out.columns]
    if drop_present:
        out = out.drop(columns=drop_present)
        print("\nDropped low-information encoded columns:", drop_present)

    # ==========================================================
    # FEATURE DISTRIBUTIONS / VARIANCE CHECKS (pre-correlation)
    # ==========================================================
    print("\n=== FEATURE DISTRIBUTIONS / VARIANCE CHECKS ===")

    n = len(out)

    # --------------------------
    # Helper labels
    # --------------------------
    def rarity_label(pct: float) -> str:
        if pct < 0.5:
            return "extremely rare"
        elif pct < 2:
            return "very rare"
        elif pct < 10:
            return "rare"
        else:
            return "common"

    def near_constant_label(dominant_pct: float) -> str:
        if dominant_pct >= 99:
            return "near-constant (>=99%)"
        elif dominant_pct >= 95:
            return "highly imbalanced (>=95%)"
        elif dominant_pct >= 90:
            return "imbalanced (>=90%)"
        else:
            return "ok"

    # ==========================================================
    # 1) Binary flags (0/1): imbalance check
    #    This includes __na, __idk, __no_response and original binaries
    # ==========================================================
    print("\n--- Binary/flag features: imbalance ---")

    # candidate binary columns: those that look like flags OR are strictly {0,1}
    candidate_bin = [c for c in out.columns if c != "respondent_id"]
    bin_rows = []

    for c in candidate_bin:
        s = out[c]

        # only consider numeric columns for binary test
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.isna().all():
            continue

        vals = set(pd.unique(s_num.dropna()))
        if not vals.issubset({0, 1}):
            continue

        ones = float(s_num.fillna(0).sum())
        pct_one = (ones / n) * 100
        pct_zero = 100 - pct_one
        dominant = max(pct_one, pct_zero)

        bin_rows.append({
            "feature": c,
            "%1": round(pct_one, 2),
            "%0": round(pct_zero, 2),
            "dominant_%": round(dominant, 2),
            "imbalance": near_constant_label(dominant),
        })

    bin_df = pd.DataFrame(bin_rows).sort_values("dominant_%", ascending=False).reset_index(drop=True)
    if bin_df.empty:
        print("No binary/flag columns detected.")
    else:
        print(bin_df.head(50).to_string(index=False))

        # list worst offenders
        worst_bin = bin_df[bin_df["dominant_%"] >= 99]
        if not worst_bin.empty:
            print("\nBinary near-constant (>=99% one value):")
            print(worst_bin[["feature", "%1", "%0", "dominant_%"]].to_string(index=False))

    # ==========================================================
    # 2) Ordinal columns (_ord): spread across allowed range
    # ==========================================================
    print("\n--- Ordinal features (_ord): spread / coverage ---")

    ord_cols = [c for c in out.columns if c.endswith("_ord")]
    ord_rows = []

    for c in ord_cols:
        s = out[c]
        observed = s.dropna()
        if len(observed) == 0:
            ord_rows.append({
                "feature": c,
                "n_obs": 0,
                "n_unique": 0,
                "min": None,
                "max": None,
                "range": None,
                "dominant_share_%": None,
                "spread_flag": "NO DATA",
            })
            continue

        vc = observed.value_counts()
        dominant_share = (vc.iloc[0] / len(observed)) * 100
        vmin = float(observed.min())
        vmax = float(observed.max())
        vrange = vmax - vmin
        n_unique = int(observed.nunique())

        # heuristic: bad if only 1 unique or dominant share huge
        if n_unique <= 1:
            spread_flag = "near-constant"
        elif dominant_share >= 95:
            spread_flag = "very skewed"
        else:
            spread_flag = "ok"

        ord_rows.append({
            "feature": c,
            "n_obs": int(len(observed)),
            "n_unique": n_unique,
            "min": vmin,
            "max": vmax,
            "range": vrange,
            "dominant_share_%": round(dominant_share, 2),
            "spread_flag": spread_flag,
        })

    ord_df = pd.DataFrame(ord_rows).sort_values(
        ["spread_flag", "dominant_share_%"],
        ascending=[True, False]
    ).reset_index(drop=True)

    if ord_df.empty:
        print("No _ord columns found.")
    else:
        # show the most suspicious first
        print(ord_df.sort_values(["spread_flag", "dominant_share_%"], ascending=[True, False]).head(50).to_string(index=False))

        suspicious_ord = ord_df[(ord_df["spread_flag"] != "ok")]
        if not suspicious_ord.empty:
            print("\nOrdinal features flagged as near-constant / very skewed:")
            print(suspicious_ord[["feature", "n_unique", "min", "max", "dominant_share_%", "spread_flag"]].to_string(index=False))

    # ==========================================================
    # 3) One-hot nominal columns: rare category detection
    #    We detect one-hots by: column contains '=' and values are {0,1}
    # ==========================================================
    print("\n--- One-hot nominal features: rare categories ---")

    onehot_cols = [c for c in out.columns if ("=" in c) and (c != "respondent_id")]
    oh_rows = []

    for c in onehot_cols:
        s = pd.to_numeric(out[c], errors="coerce").fillna(0)
        vals = set(pd.unique(s))
        if not vals.issubset({0, 1}):
            continue

        count_one = int(s.sum())
        pct_one = (count_one / n) * 100

        oh_rows.append({
            "feature": c,
            "count_1": count_one,
            "%1": round(pct_one, 3),
            "rarity": rarity_label(pct_one),
        })

    oh_df = pd.DataFrame(oh_rows).sort_values("%1", ascending=True).reset_index(drop=True)
    if oh_df.empty:
        print("No one-hot columns detected (with '=' in name).")
    else:
        print("\nMost rare one-hot categories (top 50 rarest):")
        print(oh_df.head(50).to_string(index=False))

        extreme = oh_df[oh_df["%1"] < 0.5]
        if not extreme.empty:
            print("\nExtremely rare one-hot categories (<0.5%):")
            print(extreme.to_string(index=False))

    # ==========================================================
    # CORRELATION / REDUNDANCY ANALYSIS (STRUCTURED, NO MIXING)
    # ==========================================================
    print("\n=== CORRELATION / REDUNDANCY ANALYSIS ===")

    from scipy.stats import spearmanr

    # ----------------------------------------------------------
    # Helper: report correlations in tiers (more informative than one cutoff)
    # ----------------------------------------------------------
    STRONG_T = 0.8
    MODERATE_T = 0.6

    def report_tiered_corr(df_corr: pd.DataFrame, label: str) -> None:
        rows = []
        cols = list(df_corr.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = df_corr.iloc[i, j]
                if pd.isna(val):
                    continue

                a = abs(val)
                if a >= STRONG_T:
                    level = f"STRONG (>= {STRONG_T})"
                elif a >= MODERATE_T:
                    level = f"MODERATE ({MODERATE_T}–{STRONG_T})"
                else:
                    continue

                rows.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "corr": round(float(val), 3),
                    "level": level,
                })

        if not rows:
            print(f"\n--- {label}: no pairs with |corr| >= {MODERATE_T} ---")
            return

        out_df = pd.DataFrame(rows)

        # Sort: show STRONG first, then by absolute correlation
        out_df["abs_corr"] = out_df["corr"].abs()
        out_df["level_rank"] = out_df["level"].str.startswith("STRONG").map({True: 0, False: 1})
        out_df = out_df.sort_values(["level_rank", "abs_corr"], ascending=[True, False]).drop(
            columns=["abs_corr", "level_rank"]
        )

        print(f"\n--- {label}: |corr| >= {MODERATE_T} (tiered) ---")
        print(out_df.to_string(index=False))

    # ==========================================================
    # 1) ORDINAL ↔ ORDINAL (Spearman)
    # ==========================================================
    print("\n[1] Ordinal ↔ Ordinal (Spearman)")

    ord_cols = [c for c in out.columns if c.endswith("_ord")]
    if len(ord_cols) >= 2:
        ord_df = out[ord_cols]
        corr_ord = ord_df.corr(method="spearman")
        report_tiered_corr(corr_ord, "Ordinal–Ordinal")
    else:
        print("Not enough ordinal columns to compute correlations.")

    # ==========================================================
    # 2) BINARY ↔ BINARY (phi / Pearson on {0,1})
    #    Includes flags (__na, __idk, etc.) and true binaries
    # ==========================================================
    print("\n[2] Binary ↔ Binary (phi / Pearson)")

    bin_cols = []
    for c in out.columns:
        if c == "respondent_id":
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        # require fully observed 0/1 for this audit
        if s.notna().all() and set(s.unique()).issubset({0, 1}):
            bin_cols.append(c)

    if len(bin_cols) >= 2:
        bin_df = out[bin_cols]
        corr_bin = bin_df.corr(method="pearson")
        report_tiered_corr(corr_bin, "Binary–Binary")
    else:
        print("Not enough binary columns to compute correlations.")

    # ==========================================================
    # 3) ORDINAL ↔ BINARY (Spearman)
    #    NOTE: high corr with __na flags is often STRUCTURAL (expected)
    # ==========================================================
    print("\n[3] Ordinal ↔ Binary (Spearman)")

    ord_bin_pairs = []
    for o in ord_cols:
        o_s = out[o]
        for b in bin_cols:
            b_s = out[b]

            # only rows where ordinal is observed
            mask = o_s.notna()
            if int(mask.sum()) < 30:
                continue

            o_vals = o_s[mask]
            b_vals = b_s[mask]

            # skip if either side has no variance (correlation undefined)
            if int(o_vals.nunique()) <= 1:
                continue
            if int(b_vals.nunique()) <= 1:
                continue

            corr, _ = spearmanr(o_vals, b_vals)
            if pd.isna(corr):
                continue

            a = abs(corr)
            if a >= STRONG_T:
                level = f"STRONG (>= {STRONG_T})"
            elif a >= MODERATE_T:
                level = f"MODERATE ({MODERATE_T}–{STRONG_T})"
            else:
                continue

            ord_bin_pairs.append({
                "ordinal": o,
                "binary": b,
                "corr": round(float(corr), 3),
                "level": level,
            })

    if ord_bin_pairs:
        df_pairs = pd.DataFrame(ord_bin_pairs)
        df_pairs["abs_corr"] = df_pairs["corr"].abs()
        df_pairs["level_rank"] = df_pairs["level"].str.startswith("STRONG").map({True: 0, False: 1})
        df_pairs = df_pairs.sort_values(["level_rank", "abs_corr"], ascending=[True, False]).drop(
            columns=["abs_corr", "level_rank"]
        )

        print(f"\n--- Ordinal–Binary pairs with |corr| >= {MODERATE_T} (tiered) ---")
        print(df_pairs.to_string(index=False))
        print("\nNOTE: correlations with __na flags are often STRUCTURAL, not redundancy.")
    else:
        print(f"No ordinal–binary correlations with |corr| >= {MODERATE_T} detected.")

    # ==========================================================
    # 4) ONE-HOT NOMINAL GROUPS (within same original feature)
    #    Correlation within a one-hot group is often structural;
    #    we still report strong/moderate pairs for inspection.
    # ==========================================================
    print("\n[4] One-hot nominal groups (within-feature redundancy)")

    onehot_cols = [c for c in out.columns if "=" in c]
    groups = {}
    for c in onehot_cols:
        base = c.split("=", 1)[0]
        groups.setdefault(base, []).append(c)

    any_group_reported = False

    for base, cols in groups.items():
        if len(cols) < 2:
            continue

        df = out[cols]
        corr = df.corr(method="pearson")

        rows = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if pd.isna(val):
                    continue

                a = abs(val)
                if a >= STRONG_T:
                    level = f"STRONG (>= {STRONG_T})"
                elif a >= MODERATE_T:
                    level = f"MODERATE ({MODERATE_T}–{STRONG_T})"
                else:
                    continue

                rows.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "corr": round(float(val), 3),
                    "level": level,
                })

        if rows:
            any_group_reported = True
            out_df = pd.DataFrame(rows)
            out_df["abs_corr"] = out_df["corr"].abs()
            out_df["level_rank"] = out_df["level"].str.startswith("STRONG").map({True: 0, False: 1})
            out_df = out_df.sort_values(["level_rank", "abs_corr"], ascending=[True, False]).drop(
                columns=["abs_corr", "level_rank"]
            )

            print(f"\nNominal group '{base}': correlations >= {MODERATE_T}")
            print(out_df.to_string(index=False))

    if not any_group_reported:
        print(f"No within-group one-hot correlations with |corr| >= {MODERATE_T} detected.")


    # ==========================================================
    # CORRELATION / REDUNDANCY ANALYSIS (STRUCTURED, NO MIXING)
    # ==========================================================
    print("\n=== CORRELATION / REDUNDANCY ANALYSIS ===")

    from scipy.stats import spearmanr

    # ----------------------------------------------------------
    # Helper: report correlations in tiers (more informative than one cutoff)
    # ----------------------------------------------------------
    STRONG_T = 0.8
    MODERATE_T = 0.6

    def report_tiered_corr(df_corr: pd.DataFrame, label: str) -> None:
        rows = []
        cols = list(df_corr.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = df_corr.iloc[i, j]
                if pd.isna(val):
                    continue

                a = abs(val)
                if a >= STRONG_T:
                    level = f"STRONG (>= {STRONG_T})"
                elif a >= MODERATE_T:
                    level = f"MODERATE ({MODERATE_T}–{STRONG_T})"
                else:
                    continue

                rows.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "corr": round(float(val), 3),
                    "level": level,
                })

        if not rows:
            print(f"\n--- {label}: no pairs with |corr| >= {MODERATE_T} ---")
            return

        out_df = pd.DataFrame(rows)

        # Sort: show STRONG first, then by absolute correlation
        out_df["abs_corr"] = out_df["corr"].abs()
        out_df["level_rank"] = out_df["level"].str.startswith("STRONG").map({True: 0, False: 1})
        out_df = out_df.sort_values(["level_rank", "abs_corr"], ascending=[True, False]).drop(
            columns=["abs_corr", "level_rank"]
        )

        print(f"\n--- {label}: |corr| >= {MODERATE_T} (tiered) ---")
        print(out_df.to_string(index=False))

    # ==========================================================
    # 1) ORDINAL ↔ ORDINAL (Spearman)
    # ==========================================================
    print("\n[1] Ordinal ↔ Ordinal (Spearman)")

    ord_cols = [c for c in out.columns if c.endswith("_ord")]
    if len(ord_cols) >= 2:
        ord_df = out[ord_cols]
        corr_ord = ord_df.corr(method="spearman")
        report_tiered_corr(corr_ord, "Ordinal–Ordinal")
    else:
        print("Not enough ordinal columns to compute correlations.")

    # ==========================================================
    # 2) BINARY ↔ BINARY (phi / Pearson on {0,1})
    #    Includes flags (__na, __idk, etc.) and true binaries
    # ==========================================================
    print("\n[2] Binary ↔ Binary (phi / Pearson)")

    bin_cols = []
    for c in out.columns:
        if c == "respondent_id":
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        # require fully observed 0/1 for this audit
        if s.notna().all() and set(s.unique()).issubset({0, 1}):
            bin_cols.append(c)

    if len(bin_cols) >= 2:
        bin_df = out[bin_cols]
        corr_bin = bin_df.corr(method="pearson")
        report_tiered_corr(corr_bin, "Binary–Binary")
    else:
        print("Not enough binary columns to compute correlations.")

    # ==========================================================
    # 3) ORDINAL ↔ BINARY (Spearman)
    #    NOTE: high corr with __na flags is often STRUCTURAL (expected)
    # ==========================================================
    print("\n[3] Ordinal ↔ Binary (Spearman)")

    ord_bin_pairs = []
    for o in ord_cols:
        o_s = out[o]
        for b in bin_cols:
            b_s = out[b]

            # only rows where ordinal is observed
            mask = o_s.notna()
            if int(mask.sum()) < 30:
                continue

            o_vals = o_s[mask]
            b_vals = b_s[mask]

            # skip if either side has no variance (correlation undefined)
            if int(o_vals.nunique()) <= 1:
                continue
            if int(b_vals.nunique()) <= 1:
                continue

            corr, _ = spearmanr(o_vals, b_vals)
            if pd.isna(corr):
                continue

            a = abs(corr)
            if a >= STRONG_T:
                level = f"STRONG (>= {STRONG_T})"
            elif a >= MODERATE_T:
                level = f"MODERATE ({MODERATE_T}–{STRONG_T})"
            else:
                continue

            ord_bin_pairs.append({
                "ordinal": o,
                "binary": b,
                "corr": round(float(corr), 3),
                "level": level,
            })

    if ord_bin_pairs:
        df_pairs = pd.DataFrame(ord_bin_pairs)
        df_pairs["abs_corr"] = df_pairs["corr"].abs()
        df_pairs["level_rank"] = df_pairs["level"].str.startswith("STRONG").map({True: 0, False: 1})
        df_pairs = df_pairs.sort_values(["level_rank", "abs_corr"], ascending=[True, False]).drop(
            columns=["abs_corr", "level_rank"]
        )

        print(f"\n--- Ordinal–Binary pairs with |corr| >= {MODERATE_T} (tiered) ---")
        print(df_pairs.to_string(index=False))
        print("\nNOTE: correlations with __na flags are often STRUCTURAL, not redundancy.")
    else:
        print(f"No ordinal–binary correlations with |corr| >= {MODERATE_T} detected.")

    # ==========================================================
    # 4) ONE-HOT NOMINAL GROUPS (within same original feature)
    #    Correlation within a one-hot group is often structural;
    #    we still report strong/moderate pairs for inspection.
    # ==========================================================
    print("\n[4] One-hot nominal groups (within-feature redundancy)")

    onehot_cols = [c for c in out.columns if "=" in c]
    groups = {}
    for c in onehot_cols:
        base = c.split("=", 1)[0]
        groups.setdefault(base, []).append(c)

    any_group_reported = False

    for base, cols in groups.items():
        if len(cols) < 2:
            continue

        df = out[cols]
        corr = df.corr(method="pearson")

        rows = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if pd.isna(val):
                    continue

                a = abs(val)
                if a >= STRONG_T:
                    level = f"STRONG (>= {STRONG_T})"
                elif a >= MODERATE_T:
                    level = f"MODERATE ({MODERATE_T}–{STRONG_T})"
                else:
                    continue

                rows.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "corr": round(float(val), 3),
                    "level": level,
                })

        if rows:
            any_group_reported = True
            out_df = pd.DataFrame(rows)
            out_df["abs_corr"] = out_df["corr"].abs()
            out_df["level_rank"] = out_df["level"].str.startswith("STRONG").map({True: 0, False: 1})
            out_df = out_df.sort_values(["level_rank", "abs_corr"], ascending=[True, False]).drop(
                columns=["abs_corr", "level_rank"]
            )

            print(f"\nNominal group '{base}': correlations >= {MODERATE_T}")
            print(out_df.to_string(index=False))

    if not any_group_reported:
        print(f"No within-group one-hot correlations with |corr| >= {MODERATE_T} detected.")

    # ==========================================================
    # DROP redundant previous-employment __na flags (explicit list)
    # KEEP prev_boss as the single applicability indicator
    # ==========================================================
    print("\n=== DROPPING REDUNDANT PREVIOUS-EMPLOYMENT __na FLAGS (EXPLICIT) ===")

    shape_before = out.shape
    n_features_before = out.shape[1] - 1  # exclude respondent_id

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

    shape_after = out.shape
    n_features_after = out.shape[1] - 1

    print(f"Shape: {shape_before} -> {shape_after}")
    print(f"Encoded features (excluding respondent_id): {n_features_before} -> {n_features_after}")
    print(f"Dropped {len(drop_cols)} previous-employment __na flags:")
    for c in drop_cols:
        print(" -", c)

    print(f"Remaining columns: {out.shape[1]} (including respondent_id)")

    # ==========================================================
    # NORMALIZE BINARY FEATURES TO 0/1 INTEGERS
    # (eliminate True/False everywhere)
    # ==========================================================
    print("\n=== NORMALIZING BINARY FEATURES TO 0/1 ===")

    # identify boolean columns
    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()

    if bool_cols:
        out[bool_cols] = out[bool_cols].astype(int)
        print(f"Converted {len(bool_cols)} boolean columns to int (0/1).")
    else:
        print("No boolean columns found.")

    # sanity check: ensure no True/False remain
    remaining_bool = out.select_dtypes(include=["bool"]).columns.tolist()
    if remaining_bool:
        print("WARNING: Boolean columns still present:")
        for c in remaining_bool:
            print(" -", c)


    # ==========================================================
    # FEATURE DOMINANCE TABLE (for conceptual irrelevance review)
    # ==========================================================
    print("\n=== FEATURE DOMINANCE TABLE (ALL REMAINING FEATURES) ===")

    feat_cols = [c for c in out.columns if c != "respondent_id"]

    rows = []
    n = len(out)

    for c in feat_cols:
        s = out[c]

        vc = s.value_counts(dropna=False)

        dominant_value = vc.index[0]
        dominant_count = vc.iloc[0]
        dominant_share = dominant_count / n * 100

        if pd.isna(dominant_value):
            dominant_value_name = "NaN"
        else:
            dominant_value_name = str(dominant_value)

        rows.append({
            "feature": c,
            "dominant_value": dominant_value_name,
            "dominant_share_%": round(dominant_share, 2),
            "n_unique_values": int(s.nunique(dropna=False)),
        })

    dominance_df = (
        pd.DataFrame(rows)
        .sort_values("dominant_share_%", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTotal features audited (excluding respondent_id): {len(dominance_df)}")
    print(dominance_df.to_string(index=False))

    # Save for manual conceptual review
    dominance_path = PROJECT_ROOT / "data" / "out" / "feature_dominance_table.csv"
    dominance_df.to_csv(dominance_path, index=False)
    print("\nSaved feature dominance table to:", dominance_path)



    # DO NOT DELETE BELOW
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print("\nSaved encoded drivers:", OUT_PATH)



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
                print("Gower distance already computed for this exact drivers_encoded.csv.")
                print("Using cached:", OUT_NPY)
                return
            else:
                print("Found cached Gower, but input hash changed -> recomputing.")
        except Exception:
            print("Found cached files but metadata unreadable -> recomputing.")

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

    print("Saved:", OUT_NPY)
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



main()

#% 06 Hierarchical
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram


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



main()

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
# Main (pam_cluster_profiles_k6)
# ------------------------------------------------------------


def pam_cluster_profiles_main() -> None:
    global feature_names
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
    out_path: Path,
    category_order: Optional[List[str]] = None,
    legend_title: str = "Answer",
    min_prop_to_keep: float = 0.0,
    max_categories: Optional[int] = None,
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

    plt.figure(figsize=(10, 4.5), dpi=150)
    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters), dtype=float)

    for j, cat in enumerate(cats):
        plt.bar(x, mat[:, j], bottom=bottom, label=str(cat))
        bottom += mat[:, j]

    plt.xticks(x, [str(c) for c in clusters])
    plt.ylim(0, 1.0)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion within cluster")
    plt.title(title)

    plt.legend(
        title=legend_title,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        ncol=1,
    )

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

    print(f"Plotting {len(features)} overlay features to: {args.outdir}")

    for feat in features:
        s = df[feat]
        out = args.outdir / f"overlay_k{args.k}_{safe_name(feat)}__stacked.png"

        if feat == "gender":
            plot_categorical_stacked_by_cluster(
                series=s, labels=labels_aligned,
                title=f"Overlay distribution by cluster (k={args.k}): {feat}",
                out_path=out, category_order=GENDER_TARGET_ORDER, legend_title="Answer"
            )
            continue

        if feat == "age":
            plot_categorical_stacked_by_cluster(
                series=s, labels=labels_aligned,
                title=f"Overlay distribution by cluster (k={args.k}): {feat}",
                out_path=out, category_order=AGE_LABELS + ["NA"], legend_title="Answer"
            )
            continue

        if feat == "country_live":
            region_order = [
                "US", "Canada", "UK", "European Union", "Other Europe",
                "Asia", "Middle East", "Oceania", "Latin America & Caribbean", "Africa", "Other/NA"
            ]
            plot_categorical_stacked_by_cluster(
                series=s, labels=labels_aligned,
                title=f"Overlay distribution by cluster (k={args.k}): {feat}",
                out_path=out, category_order=region_order, legend_title="Answer"
            )
            continue

        if feat == "med_pro_condition":
            plot_categorical_stacked_by_cluster(
                series=s, labels=labels_aligned,
                title=f"Overlay distribution by cluster (k={args.k}): {feat}",
                out_path=out, category_order=MED_GROUPS_ORDER, legend_title="Answer"
            )
            continue

        if feat == "work_position":
            plot_categorical_stacked_by_cluster(
                series=s, labels=labels_aligned,
                title=f"Overlay distribution by cluster (k={args.k}): {feat}",
                out_path=out, category_order=WORK_ORDER, legend_title="Answer"
            )
            continue

        # Default: high-cardinality features -> keep top categories and group rest into Other
        plot_categorical_stacked_by_cluster(
            series=s, labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={args.k}): {feat}",
            out_path=out, legend_title="Answer",
            max_categories=15, min_prop_to_keep=0.01
        )

    print("Done.")


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
