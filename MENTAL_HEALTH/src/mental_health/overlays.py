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
