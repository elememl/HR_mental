from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mental_health.preprocessing import clean_population_filters, apply_driver_skip_logic
from mental_health.overlays import extract_overlays, OVERLAY_COLS

def show_table(df: pd.DataFrame, title: str, max_rows: int = 25, figsize=(12, 6), dpi=150) -> None:
    """
    Render a pandas DataFrame as a matplotlib table (image popup).
    """
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


PROJECT_ROOT = Path(__file__).resolve().parent
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

if __name__ == "__main__":
    main()
