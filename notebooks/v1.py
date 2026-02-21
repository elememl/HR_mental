#%% [markdown]
# ### 0. Setup
# Project imports, plotting conventions, and file-system paths used throughout the analysis are specified.

#%%

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import logging
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import HTML, display
from matplotlib import MatplotlibDeprecationWarning
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn_extra.cluster import KMedoids
from umap import UMAP

logging.basicConfig(level=logging.ERROR, force=True)

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

#%% 

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "survey.csv"

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

#%% [markdown]
# ### 2. Population Filtering
# The target population is defined and filtering criteria for analytical inclusion/exclusion are applied.

#%% 

def _log_shape_change(step, before, after):
    rows_removed = before.shape[0] - after.shape[0]
    cols_removed = before.shape[1] - after.shape[1]
    print(f"{step}: removed {rows_removed} rows, {cols_removed} cols "
          f"(shape {before.shape} -> {after.shape})")


def filter_not_self_employed(df):
    filtered_df = df[df["self_employed"] == 0].drop(columns=["self_employed"]).copy()
    _log_shape_change("remove self-employed", df, filtered_df)
    return filtered_df


def filter_tech_role(df, return_with_role=False):
    col = df["tech_role"]
    normalized = col.astype(str).str.strip().str.lower()
    numeric_values = pd.to_numeric(col, errors="coerce")
    explicit_no = normalized.eq("no") | numeric_values.eq(0)
    kept_role_df = df[(col.isna()) | (~explicit_no)].copy()
    filtered_df = kept_role_df.drop(columns=["tech_role"]).copy()
    _log_shape_change("remove explicit non-tech roles", df, filtered_df)
    return (filtered_df, kept_role_df) if return_with_role else filtered_df


NA_TOKEN = "Not applicable"

_MISSING_STRINGS = {"nan", "na", "n/a", "null", "none", "<na>", "<nan>", ""}

def canonicalize_true_missing(df):
    normalized_df = df.copy()
    for col in normalized_df.select_dtypes(include=["object", "string"]).columns:
        values = normalized_df[col].astype("string")
        is_na_token = values.eq(NA_TOKEN)
        is_missing = values.str.strip().str.lower().isin(_MISSING_STRINGS)
        normalized_df.loc[~is_na_token & is_missing, col] = np.nan
    return normalized_df


def apply_driver_skip_logic(df):
    drivers_df = canonicalize_true_missing(df)
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
    no_prev = drivers_df["prev_boss"].astype(str).str.strip().isin(["0", "0.0"])
    for col in prev_children:
        drivers_df.loc[no_prev & drivers_df[col].isna(), col] = NA_TOKEN

    no_observed_bad_response = drivers_df["ever_observed_mhd_bad_response"].astype(str).str.strip().str.lower().eq("no")
    drivers_df.loc[no_observed_bad_response & drivers_df["mhdcoworker_you_not_reveal"].isna(), "mhdcoworker_you_not_reveal"] = NA_TOKEN

    no_benefits = drivers_df["benefits"].astype(str).str.strip().str.lower().isin(["no", "not eligible for coverage / n/a"])
    drivers_df.loc[no_benefits, "mh_options_known"] = NA_TOKEN
    return standardize_binary_drivers(drivers_df)

def standardize_binary_drivers(df):
    standardized_df = df.copy()
    text_maps = {"tech_company": None, "prev_boss": None, "observed_mhdcoworker_bad_conseq": {"yes": 1, "no": 0}}
    for col, text_map in text_maps.items():
        base = standardized_df[col]
        if text_map:
            mapped = base.astype(str).str.strip().str.lower().map(text_map)
            base = base.where(mapped.isna(), mapped)
        numeric = pd.to_numeric(base, errors="coerce")
        standardized_df[col] = base.where(numeric.isna(), (numeric != 0).astype(int))
    return standardized_df


OVERLAY_COLS = [
    "respondent_id", "pro_treatment", "age", "gender", "country_live", "US_state", "country_work",
    "US_work", "work_position", "mhd_past", "current_mhd", "mhd_diagnosed_condition", "mhd_believe_condition",
    "mhd_by_med_pro", "med_pro_condition", "mhd_hurt_career", "coworkers_view_neg_mhd", "treat_mhd_bad_work", "no_treat_mhd_bad_work",
]


#%% 
df = df_renamed
df_step1 = filter_not_self_employed(df)
df_clean, df_step2_keep_role = filter_tech_role(df_step1, return_with_role=True)

consistency_ct = pd.crosstab(df_step2_keep_role["tech_company"], df_step2_keep_role["tech_role"], dropna=False)
display(HTML("<b>Consistency table: tech_company x tech_role</b>" + consistency_ct.to_html()))


df_clean = df_clean.reset_index(drop=True)
df_clean.insert(0, "respondent_id", range(1, len(df_clean) + 1))


#%% [markdown]
# ### 3. Data Cleaning + Missingness
# Applying cleaning and reviewing missingness patterns before constructing modeling features.

#%% 


row_miss = df_clean.isna().mean(axis=1)
display((row_miss.describe()[["min", "max"]].mul(100).round(1).set_axis(["min", "max"])).to_frame(name="Row missingness (%)"))

#%% 
plt.figure(figsize=(8, 5), dpi=150)
plt.hist(row_miss, bins=20)
plt.xlabel("Fraction missing per row")
plt.ylabel("Number of rows")
plt.title("Row missingness distribution")
plt.tight_layout()
plt.show()


top15 = pd.DataFrame({"missing_count": df_clean.isna().sum(), "missing_percent": df_clean.isna().mean().mul(100)}).query("missing_count > 0").sort_values("missing_percent", ascending=False).rename_axis("feature").reset_index().head(15)
top15["missing_percent"] = top15["missing_percent"].map(lambda x: f"{x:.2f}%")

#%% 
display(HTML(top15.to_html(index=False)))

#%% 

col_miss = df_clean.isna().mean()
plt.figure(figsize=(8, 5), dpi=150)
plt.hist(col_miss, bins=20)
plt.xlabel("Fraction missing per column")
plt.ylabel("Number of columns")
plt.title("Column missingness distribution")
plt.tight_layout()
plt.show()

#%% 

col_miss_sorted = col_miss[col_miss > 0].sort_values(ascending=False)
plt.figure(figsize=(10, 6), dpi=150)
plt.bar(col_miss_sorted.index, col_miss_sorted.values)
plt.xticks(rotation=90)
plt.ylabel("Fraction missing")
plt.title("Missingness per feature (sorted)")
plt.tight_layout()
plt.show()


before = df_clean
df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isna().all()])
_log_shape_change("drop 100% missing columns", before, df_clean)

#%% [markdown]
# ### 4. Driver/Overlay Split
# Clustering driver variables are separated from interpretive overlay variables, followed by an audit of feature quality.

#%% 


overlay_cols_no_id = [c for c in OVERLAY_COLS if c != "respondent_id"]
df_overlays = df_clean[OVERLAY_COLS].copy()

df_drivers = df_clean.drop(columns=overlay_cols_no_id)
_log_shape_change("drop overlay columns from drivers (keep respondent_id)", df_clean, df_drivers)

drop_cols = ["why", "why_1"]
drop_stats = []
for c in drop_cols:
    observed = df_drivers[c].dropna()
    dominant_share = observed.value_counts(normalize=True).mul(100).max()
    drop_stats.append(
        f"{c}(dominant_share={f'{float(dominant_share):.2f}%' if pd.notna(dominant_share) else 'NA'}, "
        f"n_unique={int(observed.nunique())}, missing={df_drivers[c].isna().mean() * 100:.2f}%)"
    )

df_drivers_next = df_drivers.drop(columns=drop_cols)
_log_shape_change("drop open-ended features", df_drivers, df_drivers_next)
df_drivers = df_drivers_next
print(f"Dropped open-ended features: {drop_cols} | stats: {'; '.join(drop_stats)}")


audit_missing = pd.DataFrame([{"feature": c, "% missing": round(df_drivers[c].isna().mean() * 100, 2), "n_unique": int(df_drivers[c].dropna().nunique()), "dominant_share_%": round(float(df_drivers[c].dropna().value_counts(normalize=True).mul(100).max()), 2)} for c in df_drivers.columns if c != "respondent_id"]).sort_values("% missing", ascending=False).query("`% missing` > 0")
display(HTML(audit_missing.to_html(index=False)))

#%% 
display(HTML("<b>Contingency table (raw)</b>" + pd.crosstab(df_drivers["benefits"].fillna("NaN").astype(str).str.strip(), df_drivers["mh_options_known"].fillna("NaN").astype(str).str.strip(), dropna=False).to_html()))
df_drivers = apply_driver_skip_logic(df_drivers)
display(HTML("<b>Contingency table (after Rule C)</b>" + pd.crosstab(df_drivers["benefits"], df_drivers["mh_options_known"], dropna=False).to_html()))
drivers = df_drivers.drop(columns=["respondent_id"])
miss_nan = drivers.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False)
true_miss_df = pd.DataFrame({"feature": miss_nan.index, "missing_count": miss_nan.values, "missing_%": (miss_nan.values / len(drivers) * 100).round(2)})
display(HTML(true_miss_df.to_html(index=False)))

#%% 


df_drivers = df_drivers.fillna({"ever_observed_mhd_bad_response": "No response", "mhdcoworker_you_not_reveal": "No response", "mh_options_known": "No"})

cols = ["tech_company", "prev_boss", "observed_mhdcoworker_bad_conseq"]
df_drivers[cols] = df_drivers[cols].apply(lambda s: pd.to_numeric(s, errors="coerce").where(lambda n: n.notna(), s.astype("string").str.strip().str.lower().map({"yes": 1, "no": 0})).astype(int))

audit_table = pd.DataFrame([{"feature": col, "dominant_share_%": round(float(s.dropna().value_counts(normalize=True).mul(100).max()), 2), "values": s.value_counts(dropna=False).head(5).index.tolist()} for col, s in df_drivers.drop(columns=["respondent_id"]).items()])
display(HTML("<b>FEATURE TYPE AUDIT</b><style>.feature-audit th:nth-child(3),.feature-audit td:nth-child(3){text-align:left!important;}</style>" + audit_table.to_html(index=False, classes="feature-audit")))


#%% [markdown]
# ### Feature Correlations
# Correlations before encoding using Spearman.

#%% 
feature_numeric = df_drivers.drop(columns="respondent_id").apply(lambda s: s.astype("string").str.strip().astype("category").cat.codes.replace(-1, np.nan).astype(float))
corr_matrix = feature_numeric.corr(method="spearman")
spearman_table = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    .stack()
    .rename_axis(["feature_1", "feature_2"])
    .reset_index(name="corr")
    .query("0.3 <= abs(corr) <= 0.99")
    .sort_values("corr", key=lambda s: s.abs(), ascending=False)
    .assign(corr=lambda d: d["corr"].round(3)).reset_index(drop=True)
)
display(HTML("<b>Feature correlations (0.3 <= |corr| <= 0.99) - Spearman</b>" + spearman_table.to_html(index=False)))


for threshold in (0.1, 0.3):
    m = corr_matrix.abs().ge(threshold)
    np.fill_diagonal(m.values, False)
    cols = corr_matrix.columns[m.any(axis=0)]
    plt.figure(figsize=(min(26, max(8, 0.45 * len(cols))), min(24, max(6, 0.40 * len(cols)))), dpi=150)
    sns.heatmap(corr_matrix.loc[cols, cols], cmap="coolwarm", vmin=-1, vmax=1, center=0, linewidths=0.2, cbar_kws={"label": "Spearman correlation"})
    plt.title(f"Spearman heatmap (|corr| >= {threshold})")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()


#%% [markdown]
# ### 5. Encoding
# Cleaned survey responses are encoded into numeric feature representations suitable for distance-based clustering.

#%% 

NA_TOKEN = "Not applicable"
IDK_TOKEN = "I don't know"

NONE_SOME_ALL = {"None of them": 1, "Some of them": 2, "Yes, all of them": 3}
NONE_SOME_ALL_DID = {"None did": 1, "Some did": 2, "Yes, they all did": 3}
PREV_123 = {"No, at none of my previous employers": 1, "Some of my previous employers": 2, "Yes, at all of my previous employers": 3}
MIXED_DEFAULT = {"na_tokens": (NA_TOKEN,), "idk_tokens": (), "collapse_map": {}}
MIXED_V2_DEFAULT = {"na_tokens": (), "idk_tokens": (IDK_TOKEN,), "extra_flag_tokens": {}, "collapse_map": {}}
ORDINAL_MAPS = {"company_size": {"1-5": 1, "6-25": 2, "26-100": 3, "100-500": 4, "500-1000": 5, "More than 1000": 6},
                "remote_work": {"Never": 1, "Sometimes": 2, "Always": 3},
                **{c: {"No": 1, "Maybe": 2, "Yes": 3} for c in ("ph_interview", "mh_interview", "bad_conseq_mh_boss", "bad_conseq_ph_boss", "mh_comfort_coworkers", "mh_comfort_supervisor")}}

def encode_ordinal(series, mapping): return series.astype("string").str.strip().map(mapping).astype("float")


def encode_mixed_ord_plus_flags(df, feature=None, ord_mapping=None, na_tokens=(), idk_tokens=(), collapse_map=None, extra_flag_tokens=None, specs=None):
    if specs is not None:
        return pd.concat((encode_mixed_ord_plus_flags(df=df, feature=f, **spec) for f, spec in specs.items()), axis=1)
    extra_flag_tokens = {} if extra_flag_tokens is None else extra_flag_tokens
    collapse_map = {} if collapse_map is None else collapse_map
    collapsed = df[feature].astype("string").str.strip().replace(collapse_map)
    flags = {
        f"{feature}__na": collapsed.isin(na_tokens).astype(int),
        f"{feature}__idk": collapsed.isin(idk_tokens).astype(int),
        **{f"{feature}{suffix}": (collapsed == token).astype(int) for token, suffix in extra_flag_tokens.items()},
    }
    out = pd.DataFrame(flags).loc[:, lambda d: d.any(axis=0)]
    special = {*na_tokens, *idk_tokens, *extra_flag_tokens}
    out[f"{feature}_ord"] = encode_ordinal(collapsed.mask(collapsed.isin(special), pd.NA), ord_mapping)
    return out


MIXED_SPECS = {
    "friends_family_mhd_comfort": {**MIXED_DEFAULT, "ord_mapping": {"Not open at all": 1, "Somewhat not open": 2, "Neutral": 3, "Somewhat open": 4, "Very open": 5},
                                   "na_tokens": ("Not applicable to me (I do not have a mental illness)", NA_TOKEN)},
    **{f: {**MIXED_DEFAULT, "ord_mapping": NONE_SOME_ALL, "na_tokens": ()} for f in ("prev_observed_bad_conseq_mh", "bad_conseq_ph_prev_boss")},
    "prev_resources": {**MIXED_DEFAULT, "ord_mapping": NONE_SOME_ALL_DID, "na_tokens": ()},
    "mh_comfort_prev_coworkers": {**MIXED_DEFAULT, "ord_mapping": PREV_123, "na_tokens": ()},
    "leave_easy": {**MIXED_DEFAULT, "ord_mapping": {"Very difficult": 1, "Somewhat difficult": 2, "Neither easy nor difficult": 3, "Somewhat easy": 4, "Very easy": 5},
                   "na_tokens": (NA_TOKEN, "Not applicable"), "idk_tokens": (IDK_TOKEN,)},
    "prev_mh_options_known": {**MIXED_DEFAULT, "ord_mapping": {"No": 1, "I was aware of some": 2, "Yes, I was aware of all of them": 3}, "na_tokens": (),
                              "collapse_map": {"N/A (not currently aware)": "No", "No, I only became aware later": "No"}},
}


MIXED_V2_ORD_MAPS = {
    "ever_observed_mhd_bad_response": {"No": 0, "Yes, I observed": 1, "Yes, I experienced": 2},
    "prev_anonymity_protected": {"No": 0, "Sometimes": 1, "Yes, always": 2},
    "bad_conseq_mh_prev_boss": NONE_SOME_ALL,
    **{f: NONE_SOME_ALL_DID for f in ("mh_ph_prev_boss_serious", "prev_boss_mh_discuss")},
    "mh_comfort_prev_supervisor": PREV_123,
    "prev_benefits": {"No, none did": 1, "Some did": 2, "Yes, they all did": 3},
}
MIXED_V2_OVERRIDES = {"ever_observed_mhd_bad_response": {"idk_tokens": ("Maybe/Not sure", IDK_TOKEN)}}
MIXED_SPECS_V2 = {feature: {**MIXED_V2_DEFAULT, "ord_mapping": mapping, **MIXED_V2_OVERRIDES.get(feature, {})} for feature, mapping in MIXED_V2_ORD_MAPS.items()}


df = df_drivers


binary_cols = ["tech_company", "prev_boss", "observed_mhdcoworker_bad_conseq"]
out = df[["respondent_id", *binary_cols]].astype({c: int for c in binary_cols}).copy()


nominal_cols = ["anonymity_protected", "mh_family_history", "mh_ph_boss_serious", "boss_mh_discuss",
                "resources", "benefits", "mh_options_known", "mhdcoworker_you_not_reveal"]


nom = df[nominal_cols].fillna("NaN").astype("string").apply(lambda s: s.str.strip())
out = out.join(pd.get_dummies(nom, prefix=nominal_cols, prefix_sep="="))


ordinal_cols = ["company_size", "remote_work", "ph_interview", "mh_interview",
                "bad_conseq_mh_boss", "bad_conseq_ph_boss", "mh_comfort_coworkers", "mh_comfort_supervisor"]


out = out.join(pd.DataFrame({f"{c}_ord": encode_ordinal(df[c], ORDINAL_MAPS[c]) for c in ordinal_cols}, index=df.index))
out = out.join(encode_mixed_ord_plus_flags(df=df, specs={**MIXED_SPECS, **MIXED_SPECS_V2}))
out = out.astype({c: int for c in out.select_dtypes(include=["bool"]).columns})


#%% [markdown]
# ### 6. Gower Distance
# A missingness-aware pairwise distance matrix is computed for subsequent PAM clustering.

#%% 


X = out.drop(columns=["respondent_id"]).apply(pd.to_numeric, errors="raise").to_numpy(np.float64)
feature_range = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
valid = np.isfinite(feature_range) & (feature_range > 0)
X, feature_range = X[:, valid], feature_range[valid]

n = X.shape[0]
finite = np.isfinite(X)
D = np.zeros((n, n), dtype=np.float64)
for i0 in range(0, n, 256):
    i1 = min(i0 + 256, n)
    comparable = finite[i0:i1, None, :] & finite[None, :, :]
    diff = np.abs(X[i0:i1, None, :] - X[None, :, :]) / feature_range
    numer = np.where(comparable, diff, 0.0).sum(axis=2)
    denom = comparable.sum(axis=2)
    D[i0:i1] = np.divide(numer, denom, out=np.ones_like(numer), where=denom > 0)

D = ((D + D.T) * 0.5).astype(np.float32)
np.fill_diagonal(D, 0.0)


#%% [markdown]
# ### 7. PAM + k Selection
# k-medoids (PAM) solutions are fit across candidate k values, and silhouette and cluster-size diagnostics are reviewed.

#%% 


K_LIST = range(3, 14)
RANDOM_STATE = 42
SIL_QUANTILES = (25, 50, 75)
SIL_COLS = ["sil_avg", *(f"sil_q{q}" for q in SIL_QUANTILES)]


rows = []
pam_labels_by_k = {}
for k in K_LIST:
    model = KMedoids(n_clusters=k, metric="precomputed", method="pam", init="k-medoids++", random_state=RANDOM_STATE)
    labels_k = model.fit_predict(D)
    sil = silhouette_samples(D, labels_k, metric="precomputed")
    sizes = np.bincount(labels_k, minlength=k)
    sil_stats = {"sil_avg": float(np.mean(sil))}
    sil_stats.update({f"sil_q{q}": float(v) for q, v in zip(SIL_QUANTILES, np.percentile(sil, SIL_QUANTILES))})
    pam_labels_by_k[k] = labels_k.astype(np.int32)
    rows.append({"k": k, **sil_stats, "cluster_sizes": sizes.tolist()})


pam_summary = pd.DataFrame(rows).sort_values("k")
df_table = pam_summary[["k", *SIL_COLS, "cluster_sizes"]].round({c: 4 for c in SIL_COLS})

#%% 
display(HTML(df_table.to_html(index=False)))


k_values = pam_summary["k"].to_numpy(dtype=int)
sil = pam_summary["sil_avg"].to_numpy(dtype=float)
cluster_sizes = pam_summary["cluster_sizes"].tolist()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(k_values, sil, marker="o")
ax.set_title("Silhouette vs k")
ax.set_xlabel("k")
ax.set_ylabel("Average silhouette")
ax.set_xticks(k_values)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.boxplot(cluster_sizes, tick_labels=k_values.astype(str), showfliers=True)
ax.set_title("Cluster size distribution by k")
ax.set_xlabel("k")
ax.set_ylabel("Cluster size")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
plt.show()
plt.close(fig)


#%% [markdown]
# ### 8. Embeddings (PCoA/UMAP)
# The clustered distance structure is summarized using low-dimensional embeddings for visualization.

#%% 


K = 6
labels = pam_labels_by_k[K].astype(int)
n_components = 3
n = D.shape[0]
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * (J @ (D ** 2) @ J)
evals, evecs = np.linalg.eigh((B + B.T) * 0.5)
idx = evals.argsort()[::-1]
evals, evecs = evals[idx], evecs[:, idx]
pos = evals > 1e-12
m = min(n_components, int(pos.sum()))
evals_pos, evecs_pos = evals[pos][:m], evecs[:, pos][:, :m]
coords = evecs_pos * np.sqrt(evals_pos)
explained = evals_pos / evals_pos.sum()


def plot_embedding(embedding, labels, title, axis_labels):
    dims = embedding.shape[1]
    is_3d = dims == 3
    fig = plt.figure(figsize=(11, 8) if is_3d else (10, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)

    for c in np.unique(labels):
        pts = embedding[labels == c]
        ax.scatter(*pts[:, :dims].T, s=10, alpha=0.8, label=f"Cluster {c}")

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if is_3d:
        ax.set_zlabel(axis_labels[2])
    ax.set_title(title)

    legend_kws = {"markerscale": 1.5, "frameon": False}
    if is_3d:
        legend_kws.update({"loc": "center left", "bbox_to_anchor": (1.02, 0.5), "borderaxespad": 0.0})
        fig.subplots_adjust(left=0.02, right=0.80, top=0.92, bottom=0.06)
    else:
        fig.tight_layout()
    ax.legend(**legend_kws)

    plt.show()
    plt.close(fig)


plot_embedding(coords[:, :2], labels, f"PCoA (Gower) - k={K} clusters (2D)",
               [f"PCoA1 ({explained[0]*100:.1f}%)", f"PCoA2 ({explained[1]*100:.1f}%)"])
plot_embedding(coords[:, :3], labels, f"PCoA (Gower) - k={K} clusters (3D)",
               [f"PCoA1 ({explained[0]*100:.1f}%)", f"PCoA2 ({explained[1]*100:.1f}%)", f"PCoA3 ({explained[2]*100:.1f}%)"])


#%% 


UMAP_KW = dict(metric="precomputed", n_neighbors=6, min_dist=0.5, random_state=42, spread=2, n_jobs=1)
umap_2d = UMAP(n_components=2, **UMAP_KW).fit_transform(D)
umap_3d = UMAP(n_components=3, **UMAP_KW).fit_transform(D)

plot_embedding(umap_2d, labels, f"UMAP (precomputed Gower) - k={K} clusters (2D)", ["UMAP1", "UMAP2"])
plot_embedding(umap_3d, labels, f"UMAP (precomputed Gower) - k={K} clusters (3D)", ["UMAP1", "UMAP2", "UMAP3"])
#%% [markdown]
# ### 9. Stability Validation
# Robustness across neighboring k values is assessed using ARI, overlap statistics, and Sankey-style flow summaries.

#%% 


KS = [4, 5, 6, 7]
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

label_map = {k: pam_labels_by_k[k].astype(int) for k in KS}
adjacent_transitions = list(zip(KS[:-1], KS[1:]))
ari_df = pd.DataFrame([{"k1": k1, "k2": k2, "ARI": float(adjusted_rand_score(label_map[k1], label_map[k2]))}
                       for i, k1 in enumerate(KS) for k2 in KS[i + 1:]]).sort_values(["k1", "k2"]).reset_index(drop=True)
#%% 
display(HTML(ari_df.to_html(index=False)))


stability_rows, all_flows = [], []
for k_from, k_to in adjacent_transitions:
    labels_from, labels_to = label_map[k_from], label_map[k_to]
    contingency_df = pd.crosstab(pd.Series(labels_from, name="from"), pd.Series(labels_to, name="to"), normalize=False, dropna=False)
    row_pct = contingency_df.div(contingency_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    max_row = row_pct.max(axis=1).to_numpy(dtype=float)
    probs = row_pct.to_numpy(dtype=float)
    row_entropy = -(np.where(probs > 0, probs * np.log(probs), 0.0)).sum(axis=1)
    eff_targets = np.exp(row_entropy)
    weights = contingency_df.sum(axis=1).to_numpy(dtype=float)
    stability_rows.append({
        "transition": f"{k_from}→{k_to}",
        "avg_max_row_overlap": float((max_row * weights).sum() / weights.sum()),
        "median_max_row_overlap": float(np.median(max_row)),
        "median_row_entropy": float(np.median(row_entropy)),
        "pct_clusters_split": float((max_row < 0.80).mean() * 100.0),
        "effective_num_targets_median": float(np.median(eff_targets)),
    })

    all_flows.append(
        contingency_df.stack().rename("value").reset_index().query("value > 0").assign(
            source=lambda d: d["from"].map(lambda c: f"k{k_from}_c{int(c)}"),
            target=lambda d: d["to"].map(lambda c: f"k{k_to}_c{int(c)}"),
        )[["source", "target", "value"]]
    )

stability_table = pd.DataFrame(stability_rows).round(2)
#%% 
display(HTML(stability_table.to_html(index=False)))


#%% 
flow_df = pd.concat(all_flows, ignore_index=True)
node_labels = pd.unique(pd.concat([flow_df["source"], flow_df["target"]])).tolist()
node_index = {name: i for i, name in enumerate(node_labels)}
links = flow_df[["source", "target", "value"]].replace({"source": node_index, "target": node_index}).astype(int)
fig = go.Figure(data=[go.Sankey(node=dict(label=node_labels, pad=15, thickness=14), link=links.to_dict("list"))])
fig.update_layout(title_text=f"Global Sankey: {' → '.join([f'k{k}' for k in KS])} (adjacent transitions)", font_size=11)
display(HTML(fig.to_html(include_plotlyjs="cdn")))


#%% [markdown]
# ### 10. Cluster Profiles
# Cluster-level patterns and pairwise separations are examined for the selected k=6 solution.

def normalize_text(x, na_label=None, lower=False, normalize_quotes=False):
    quote_map = str.maketrans({"’": "'", "“": '"', "”": '"'})

    def clean(v):
        s = na_label if (na_label is not None and pd.isna(v)) else str(v)
        if normalize_quotes:
            s = s.translate(quote_map)
        s = re.sub(r"\s+", " ", s.replace("\u00a0", " ").strip())
        if lower:
            s = s.lower()
        return na_label if (na_label is not None and s == "") else s

    return x.map(clean) if isinstance(x, pd.Series) else clean(x)


def _normalize_categorical_frame(df_raw):
    df_norm = df_raw.drop(columns="respondent_id", errors="ignore").apply(lambda s: normalize_text(s, na_label="NA"))
    features = df_norm.columns.tolist()
    return df_norm, features


def labels_by_id(df_target, labels, respondent_ids_source):
    id_to_label = pd.Series(labels, index=respondent_ids_source).to_dict()
    aligned_labels = df_target["respondent_id"].map(id_to_label).to_numpy()
    return aligned_labels.astype(int)


def plot_raw_feature_panel(df_raw, labels_aligned, features, suptitle, ncols=3, color_map=None, recode_map=None, legend_mode="shared", legend_order=None):
    n = len(features)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.6 * nrows), dpi=150)
    axes = np.asarray(axes).ravel()

    clusters = sorted(np.unique(labels_aligned).tolist())
    x = np.arange(len(clusters))
    cluster_series = pd.Series(labels_aligned, name="cluster")

    recode_map = recode_map or {}
    legend_order = legend_order or []

    series_by_feature = {feat: normalize_text(recode_map[feat](df_raw[feat]) if feat in recode_map else df_raw[feat], na_label="NA")
                         for feat in features}
    all_categories = list(dict.fromkeys(cat for s in series_by_feature.values()
                                        for cat in s.value_counts(dropna=False).index))


    color_for_cat = dict(color_map or {})
    palette = plt.get_cmap("tab20").colors
    missing_cats = [c for c in all_categories if c not in color_for_cat]
    color_for_cat.update({c: palette[i % len(palette)] for i, c in enumerate(missing_cats)})

    for i, feat in enumerate(features):
        ax = axes[i]
        series = series_by_feature[feat]
        categories = series.value_counts(dropna=False).index
        mat = pd.crosstab(cluster_series, series, normalize="index").reindex(index=clusters, columns=categories, fill_value=0.0).to_numpy(float)

        bottom = np.zeros(len(clusters), dtype=float)
        for j, cat in enumerate(categories):
            ax.bar(x, mat[:, j], bottom=bottom, label=str(cat), color=color_for_cat[cat])
            bottom += mat[:, j]

        ax.set(xticks=x, xticklabels=[str(c) for c in clusters], ylim=(0, 1.0), xlabel="", title=feat)
        if legend_mode == "per_subplot":
            ax.legend(title="Answer", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7)

    for ax_extra in axes[n:]:
        ax_extra.axis("off")

    fig.suptitle(suptitle, y=0.99, fontsize=14)
    fig.text(0.02, 0.5, "Proportion within cluster", va="center", rotation="vertical")
    fig.supxlabel("Cluster", y=0.01)

    all_cat_set = set(all_categories)
    ordered = [c for c in legend_order if c in all_cat_set]
    legend_cats = ordered + [c for c in all_categories if c not in set(ordered)]
    legend_handles = [mpatches.Patch(color=color_for_cat[c], label=c) for c in legend_cats]
    if legend_mode == "shared":
        fig.legend(handles=legend_handles, title="Answer", loc="upper left", bbox_to_anchor=(0.885, 0.98), fontsize=8)
    right = 0.86 if legend_mode == "shared" else 1
    plt.tight_layout(rect=[0.04, 0.03, right, 0.96])
    plt.show()
    plt.close(fig)


RAW_GLOBAL_COLOR_PAIRS = [
    ("#2ca02c", ["Yes", "1", "Always", "100-500", "None of them", "Yes, all", "easy", "open"]),
    ("#d62728", ["No", "0", "Never", "6-25", "Yes, all of them", "difficult", "No or not eligible for coverage", "not open"]),
    ("#1f77b4", ["Maybe", "Sometimes", "1-5", "Some of them", "Maybe/Not sure", "I am not sure"]),
    ("#ff7f0e", ["Some", "Yes, I observed"]),
    ("#9467bd", ["Only became aware later", "No response", "Neither easy nor difficult"]),
    ("#4aa3df", ["I don't know"]),
    ("#f1c40f", ["Neutral"]),
    ("#8c564b", ["Yes, I experienced"]),
    ("#7f7f7f", ["Not applicable"]),
]
RAW_GLOBAL_COLOR_MAP = {label: color for color, labels in RAW_GLOBAL_COLOR_PAIRS for label in labels}


def _expand_canonical_map(canonical_map):
    return {raw: canonical for canonical, raws in canonical_map.items() for raw in raws}


PLOT_RECODE_SPECS = {
    "benefits": {
        "na_label": "NA",
        "mapping": {"Not eligible for coverage / N/A": "No or not eligible for coverage", "No": "No or not eligible for coverage"},
    },
    "leave_easy": {
        "na_label": "NA",
        "mapping": {"Very difficult": "difficult", "Somewhat difficult": "difficult", "Very easy": "easy", "Somewhat easy": "easy"},
    },
    "prev": {
        "na_label": "Not applicable",
        "mapping": _expand_canonical_map({
            "No": ["No, none did", "None of them", "No, at none of my previous employers", "No", "N/A (not currently aware)", "None did"],
            "Yes, all": ["Yes, they all did", "Yes, I was aware of all of them", "Yes, all of them", "Yes, at all of my previous employers", "Yes, always", "yes, always"],
            "Some": ["Some did", "I was aware of some", "Some of them", "Some of my previous employers", "Sometimes"],
            "Only became aware later": ["No, I only became aware later"],
        }),
    },
    "stigma": {
        "na_label": "Not applicable",
        "mapping": _expand_canonical_map({
            "Not applicable": ["", "Not applicable to me (I do not have a mental illness)"],
            "open": ["Somewhat open", "Very open"],
            "not open": ["Somewhat not open", "Not open at all"],
        }),
    },
}


def _make_series_recode(mapping, na_label):
    def _recode(series):
        return normalize_text(series, na_label=na_label).replace(mapping)
    return _recode


def build_feature_plots(df_raw, labels_aligned):
    weak_panel_features = [
        "remote_work", "tech_company", "mh_interview", "company_size", "prev_boss",
        "ph_interview", "bad_conseq_ph_prev_boss", "observed_mhdcoworker_bad_conseq", "bad_conseq_ph_boss",
    ]
    strong_panel_features = [
        "benefits", "mh_ph_boss_serious", "leave_easy", "resources", "bad_conseq_mh_boss",
        "mh_comfort_coworkers", "mh_options_known", "anonymity_protected", "mh_comfort_supervisor",
    ]
    prev_panel_features = [
        "prev_benefits", "mh_ph_prev_boss_serious", "mh_comfort_prev_coworkers", "prev_resources",
        "bad_conseq_mh_prev_boss", "mh_comfort_prev_supervisor", "prev_mh_options_known", "prev_boss_mh_discuss", "prev_anonymity_protected",
    ]
    stigma_panel_features = [
        "prev_observed_bad_conseq_mh", "mh_family_history", "mhdcoworker_you_not_reveal",
        "friends_family_mhd_comfort", "ever_observed_mhd_bad_response", "boss_mh_discuss",
    ]
    recode_fns = {name: _make_series_recode(spec["mapping"], spec["na_label"]) for name, spec in PLOT_RECODE_SPECS.items()}

    panel_specs = [
        {
            "features": weak_panel_features,
            "suptitle": "Feature distributions across 6 clusters (weak separators — mostly uniform distributions)",
            "legend_mode": "per_subplot",
        },
        {
            "features": strong_panel_features,
            "suptitle": "Feature distributions across 6 clusters (strong separators)",
            "legend_mode": "shared",
            "recode_map": {"benefits": recode_fns["benefits"], "leave_easy": recode_fns["leave_easy"]},
        },
        {
            "features": prev_panel_features,
            "suptitle": "Previous employment feature distributions across 6 clusters",
            "legend_mode": "shared",
            "recode_map": {feat: recode_fns["prev"] for feat in prev_panel_features},
            "legend_order": ["Yes, all", "Some", "No", "Only became aware later", "I don't know", "Not applicable"],
        },
        {
            "features": stigma_panel_features,
            "suptitle": "Distribution of stigma-related indicators across 6 clusters",
            "legend_mode": "per_subplot",
            "recode_map": {feat: recode_fns["stigma"] for feat in stigma_panel_features},
        },
    ]

    for spec in panel_specs:
        plot_raw_feature_panel(df_raw=df_raw, labels_aligned=labels_aligned, ncols=3, color_map=RAW_GLOBAL_COLOR_MAP, **spec)


def pct_rows(df_norm, features, mask_a, mask_b, left_col, right_col, selected_features_lc=None):
    rows = []
    for feat in features:
        s_all = df_norm[feat]
        s_a, s_b = s_all[mask_a], s_all[mask_b]
        vc_a, vc_b = s_a.value_counts(dropna=False), s_b.value_counts(dropna=False)
        denom_a, denom_b = len(s_a), len(s_b)

        for cat in s_all.value_counts(dropna=False).index:
            feature_key = f"{feat}={cat}"
            if selected_features_lc and feature_key.strip().lower() not in selected_features_lc:
                continue

            p_a, p_b = vc_a.get(cat, 0) / denom_a, vc_b.get(cat, 0) / denom_b
            rows.append({"feature": feature_key, left_col: f"{p_a*100:.1f}%", right_col: f"{p_b*100:.1f}%", "_diff": abs(p_a - p_b)})
    return rows


def cluster_vs_rest_table(df_raw, labels_aligned, top_n=15, exclude_subs=None):
    df_norm, features = _normalize_categorical_frame(df_raw)
    clusters = sorted(np.unique(labels_aligned).tolist())
    results = {}
    exclude_subs = {} if exclude_subs is None else exclude_subs

    for c in clusters:
        in_cluster = labels_aligned == c
        rows = pct_rows(df_norm, features, in_cluster, ~in_cluster, "cluster_%", "rest_%")

        df_all = pd.DataFrame(rows).sort_values("_diff", ascending=False)
        subs = [s.strip().lower() for s in exclude_subs.get(int(c), [])]
        df_all = df_all[df_all["feature"].str.lower().map(lambda lx: not any(sub in lx for sub in subs))]
        df = df_all.head(top_n).drop(columns="_diff")
        results[int(c)] = df.reset_index(drop=True)

    return results


def build_raw_cluster_vs_rest_selected(df_raw, labels_aligned, cluster_id, selected_features):
    df_norm, features = _normalize_categorical_frame(df_raw)
    selected_lc = {x.strip().lower() for x in selected_features}

    in_cluster = labels_aligned == cluster_id
    rows = pct_rows(df_norm, features, in_cluster, ~in_cluster, "cluster_%", "rest_%", selected_lc)
    return pd.DataFrame(rows).drop(columns="_diff", errors="ignore")


def pairwise_tables(df_raw, labels_aligned, pairs, top_n=15):
    df_norm, features = _normalize_categorical_frame(df_raw)
    results = {}

    for c1, c2 in pairs:
        mask1 = labels_aligned == c1
        mask2 = labels_aligned == c2
        rows = pct_rows(df_norm, features, mask1, mask2, f"cluster_{c1}_%", f"cluster_{c2}_%")

        df = pd.DataFrame(rows).sort_values("_diff", ascending=False).head(top_n).drop(columns="_diff")
        results[(int(c1), int(c2))] = df.reset_index(drop=True)

    return results


def _same_answers(values, *features):
    values_list = [values] if isinstance(values, str) else list(values)
    return {feature: values_list.copy() for feature in features}


SELECTED = {
    0: {
        "prev_boss": ["Yes"],
        "bad_conseq_mh_boss": ["Maybe"],
        "anonymity_protected": ["I don't know"],
        "prev_boss_mh_discuss": ["None did"],
        "boss_mh_discuss": ["No"],
    },
    1: {
        **_same_answers("No", "resources", "bad_conseq_ph_boss", "mh_interview", "mh_comfort_coworkers", "mh_comfort_supervisor"),
        **_same_answers("I don't know", "mh_ph_boss_serious", "anonymity_protected"),
        "mh_options_known": ["Yes"],
        "boss_mh_discuss": ["None did"],
        "bad_conseq_mh_boss": ["Maybe"],
    },
    2: {
        **_same_answers(["No", "I don't know"], "resources", "mh_ph_boss_serious"),
        **_same_answers(["Some of them", "Yes, all of them"], "bad_conseq_mh_prev_boss", "prev_observed_bad_conseq_mh"),
        "prev_resources": ["Some did", "Yes, they all did"],
        "bad_conseq_mh_boss": ["Maybe"],
        "boss_mh_discuss": ["No"],
        "prev_boss_mh_discuss": ["None did"],
    },
    3: {
        **_same_answers("No", "boss_mh_discuss", "mh_ph_boss_serious", "mh_comfort_supervisor", "mh_comfort_coworkers", "mh_interview", "ph_interview"),
        **_same_answers("Yes", "bad_conseq_mh_boss", "mh_family_history"),
        "anonymity_protected": ["I don't know", "No"],
        "leave_easy": ["Very difficult", "Somewhat difficult"],
        "ever_observed_mhd_bad_response": ["Yes, I observed", "Yes, I experienced"],
        "prev_benefits": ["No, none did"],
    },
    4: {
        **_same_answers("I don't know", "prev_anonymity_protected", "anonymity_protected"),
        **_same_answers("No", "boss_mh_discuss", "mh_family_history"),
        "bad_conseq_mh_boss": ["No", "Maybe"],
        "prev_observed_bad_conseq_mh": ["None of them"],
    },
    5: {
        **_same_answers("Yes", "mh_options_known", "mh_family_history", "bad_conseq_mh_boss"),
        **_same_answers("Maybe", "mh_comfort_coworkers", "mh_comfort_supervisor"),
        "leave_easy": ["Very easy", "Somewhat easy", "Neither easy nor difficult"],
        "friends_family_mhd_comfort": ["Somewhat open", "Very open"],
    },
}

SELECTED_BY_CLUSTER = {
    cluster_id: [f"{feature}={value}" for feature, values in feature_map.items() for value in values]
    for cluster_id, feature_map in SELECTED.items()
}


def heatmap(df_raw, labels_aligned):
    df_norm, features = _normalize_categorical_frame(df_raw)
    clusters = sorted(np.unique(labels_aligned).tolist())
    pairs = [(c1, c2) for i, c1 in enumerate(clusters) for c2 in clusters[i + 1:]]
    cluster_series = pd.Series(labels_aligned, name="cluster")

    rows = []
    for feat in features:
        s = df_norm[feat]
        categories = s.value_counts(dropna=False).index
        mat = pd.crosstab(cluster_series, s, normalize="index").reindex(index=clusters, columns=categories, fill_value=0.0)
        for cat in categories:
            diffs = {f"C{c1}-C{c2}": abs(mat.at[c1, cat] - mat.at[c2, cat]) for c1, c2 in pairs}
            rows.append({"feature": f"{feat}={cat}", **diffs})

    heatmap_data = pd.DataFrame(rows).set_index("feature").loc[lambda d: d.max(axis=1).nlargest(30).index]
    plt.figure(figsize=(max(10, len(heatmap_data.columns) * 0.8), max(6, len(heatmap_data) * 0.35)))
    sns.heatmap(heatmap_data, cmap="viridis")
    plt.title("Raw feature separation (top 30)")
    plt.xticks(rotation=90, ha="center")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()


def pam_cluster_profiles_main():
    labels = pam_labels_by_k[K].astype(int)
    exclusion = {1: ["not applicable"]}
    pairs = [(0, 1), (2, 5), (3, 4)]
    size_df = pd.Series(labels).value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
    size_df["cluster"] = size_df["cluster"].astype(int)
    size_df["percent"] = (size_df["count"] / len(labels) * 100).round(1)

    def show_table(title, df):
        display(HTML(f"<b>{title}</b>" + df.to_html(index=False)))

    display(HTML("<b>k=6 Cluster Sizes</b>" + size_df.to_html(index=False)))

    df_raw = df_drivers.copy()
    labels_raw = labels_by_id(df_raw, labels, out["respondent_id"])
    build_feature_plots(df_raw, labels_raw)

    heatmap(df_raw, labels_raw)

    raw_tables = cluster_vs_rest_table(df_raw, labels_raw, exclude_subs=exclusion)
    for c, table in sorted(raw_tables.items()):
        show_table(f"CLUSTER {c} vs REST (Top deviations)", table)
        show_table(f"CLUSTER {c} vs REST (Selected features)",
                   build_raw_cluster_vs_rest_selected(df_raw, labels_raw, c, SELECTED_BY_CLUSTER[c]))

    pair_tables = pairwise_tables(df_raw, labels_raw, pairs=pairs)
    for c1, c2 in pairs:
        show_table(f"CLUSTER {c1} vs CLUSTER {c2} (Top deviations)", pair_tables[(c1, c2)])


pam_cluster_profiles_main()

#%% [markdown]
# ### 11. Overlay Profiles
# Clusters are characterized using overlay variables that were excluded from the clustering drivers.

#%% 


def recode_overlay(col, val):
    s = normalize_text(val, lower=True, normalize_quotes=True)

    if col == "gender":
        sp = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-\/]", " ", s)).strip()
        tokens = set(sp.split())
        if sp in {"m", "mail"} or ("male" in tokens and "female" not in tokens) or ("man" in tokens and "woman" not in tokens):
            return "Male"
        if sp == "f" or ("female" in tokens and "male" not in tokens) or ("woman" in tokens and "man" not in tokens):
            return "Female"
        return "Other"

    if col == "age":
        v = pd.to_numeric(s, errors="coerce")
        if not np.isfinite(v):
            return "NA"
        for upper, label in (
            (20, "less than 20"),
            (26, "from 20 to 25"),
            (31, "from 26 to 30"),
            (36, "from 31 to 35"),
            (41, "from 36 to 40"),
            (51, "from 41 to 50"),
            (np.inf, "more than 51"),
        ):
            if v < upper:
                return label

    if col == "med_pro_condition":
        if any(k in s for k in ("no diagnosis", "no dx", "none", "healthy", "no condition", "no mental")):
            return "No diagnosis/NA"
        if ("mood" in s) and ("disorder" in s):
            return "Mood disorder"
        if ("anxiety" in s) and ("disorder" in s):
            return "anxiety disorder"
        return "other"

    if col == "work_position":
        has_fe = ("front-end" in s) or ("front end" in s)
        has_be = ("back-end" in s) or ("back end" in s)
        has_designer = "designer" in s
        for label, tokens in (
            ("Sales", ("sales",)),
            ("Dev Evangelist/Advocate", ("dev evangelist", "developer evangelist", "advocat")),
            ("Leadership (Supervisor/Exec)", ("supervisor/team lead", "team lead", "supervisor", "executive leadership", "executive")),
            ("DevOps/SysAdmin", ("devops", "sysadmin", "sys admin", "sys-admin")),
            ("Support", ("support",)),
        ):
            if any(token in s for token in tokens):
                return label
        if has_fe and has_be:
            return "Full-stack (FE+BE)"
        if has_fe:
            return "Front-end Developer"
        if has_be:
            return "Back-end Developer"
        if has_designer:
            return "Designer"
        return "other"

    raise ValueError(col)


def cluster_stats(series, labels, max_categories, min_prop):
    s = normalize_text(series, na_label="NA")
    for keep in (
        lambda vc: vc.index[:max_categories],
        lambda vc: vc.index[(vc / len(s)) >= min_prop],
    ):
        vc = s.value_counts(dropna=False)
        s = s.where(s.isin(keep(vc)), other="Other")

    cats = s.value_counts(dropna=False).index.tolist()
    labels = np.asarray(labels, dtype=int)
    clusters = sorted(np.unique(labels).tolist())
    counts = {c: s[labels == c].value_counts(dropna=False) for c in clusters}
    totals = {c: float((labels == c).sum()) for c in clusters}
    return clusters, cats, counts, totals


#%% 
labels = pam_labels_by_k[K].astype(int)
df = df_overlays.copy()
labels_aligned = labels_by_id(df, labels, out["respondent_id"])
for col in ("gender", "age", "med_pro_condition", "work_position"):
    df[col] = df[col].map(lambda v: recode_overlay(col, v))


#%%
OVERLAY_PANELS = [
    ["age", "gender", ("country_work", 15, 0.01), "work_position"],
    ["mhd_past", "current_mhd", "pro_treatment", "no_treat_mhd_bad_work"],
    ["mhd_hurt_career", "coworkers_view_neg_mhd", "med_pro_condition", "mhd_by_med_pro"],
]

for panel in OVERLAY_PANELS:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
    for ax, spec in zip(axes.ravel(), panel):
        feat, max_categories, min_prop = spec if isinstance(spec, tuple) else (spec, 10**9, 0.0)
        clusters, cats, counts, totals = cluster_stats(df[feat], labels_aligned, max_categories, min_prop)
        x = np.arange(len(clusters))
        bottom = np.zeros(len(clusters), dtype=float)
        for cat in cats:
            heights = np.array([counts[c].get(cat, 0) / totals[c] for c in clusters], dtype=float)
            ax.bar(x, heights, bottom=bottom, label=str(cat), edgecolor="none", linewidth=0.0, antialiased=False)
            bottom += heights
        ax.set_xticks(x, [str(c) for c in clusters])
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Proportion within cluster")
        ax.set_title(f"Overlay distribution by cluster (k={K}): {feat}")
        ax.legend(title="Answer", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

#%%
dom_feats = ["mhd_by_med_pro", "pro_treatment", "mhd_past", "current_mhd", "med_pro_condition"]
pct_feats = ["no_treat_mhd_bad_work", "mhd_hurt_career", "coworkers_view_neg_mhd"]
stats_by_feat = {feat: cluster_stats(df[feat], labels_aligned, 10**9, 0.0) for feat in dom_feats + pct_feats}

#%% 
clusters = next(iter(stats_by_feat.values()))[0]
dom_rows = []
for c in clusters:
    row = {"cluster": c}
    for feat in dom_feats:
        _, _, counts, totals = stats_by_feat[feat]
        top = counts[c]
        row[feat] = f"{top.index[0]} ({top.iloc[0] / totals[c] * 100:.1f}%)"
    dom_rows.append(row)
dom_df = pd.DataFrame(dom_rows)
display(HTML("<b>Dominant answers (overlays)</b>" + dom_df.to_html(index=False)))

for feat in pct_feats:
    _, cats, counts, totals = stats_by_feat[feat]
    pct_rows = []
    for c in clusters:
        row = {"cluster": c}
        for cat in cats:
            row[str(cat)] = f"{counts[c].get(cat, 0) / totals[c] * 100:.1f}%"
        pct_rows.append(row)
    pct_df = pd.DataFrame(pct_rows)
    display(HTML(f"<b>{feat}</b>" + pct_df.to_html(index=False)))
