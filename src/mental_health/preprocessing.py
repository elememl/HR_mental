from __future__ import annotations
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
        s = df2[col].astype(str).str.strip().str.lower()
        # map only known yes/no; leave everything else unchanged
        df2.loc[s == "yes", col] = 1
        df2.loc[s == "no", col] = 0

        # make it numeric where possible
        df2[col] = pd.to_numeric(df2[col], errors="ignore")

    return df2
