from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print("\nSaved encoded drivers:", OUT_PATH)


if __name__ == "__main__":
    main()
