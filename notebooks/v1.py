#%% [markdown]
# ### 0. Setup
# Project imports, plotting conventions, and file-system paths used throughout the analysis are specified.

#%%

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import itertools
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

def _normalize_categorical_frame(df_raw):
    features = [c for c in df_raw.columns if c != "respondent_id"]
    df_norm = df_raw[features].copy()
    for col in features:
        s = df_norm[col].astype(object)
        s[pd.isna(s)] = "NA"
        df_norm[col] = s.astype(str).str.strip().replace({"": "NA"})
    return df_norm, features


def _build_raw_pairwise_diff_matrix(df_raw, labels_aligned):
    df_norm, features = _normalize_categorical_frame(df_raw)
    clusters = sorted(np.unique(labels_aligned).tolist())
    pairs = list(itertools.combinations(clusters, 2))

    rows = []
    for feat in features:
        s_all = df_norm[feat]
        labels_use = labels_aligned
        cats = s_all.value_counts(dropna=False).index.tolist()

        for cat in cats:
            row = {"feature": f"{feat}={cat}"}
            for c1, c2 in pairs:
                s1 = s_all[labels_use == c1]
                s2 = s_all[labels_use == c2]
                denom1 = float(len(s1))
                denom2 = float(len(s2))
                p1 = float((s1 == cat).sum()) / denom1
                p2 = float((s2 == cat).sum()) / denom2
                row[f"C{c1}-C{c2}"] = abs(p1 - p2)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("feature")
    return df

def _align_labels_by_respondent_id(df_target, labels, respondent_ids_source):
    id_to_label = pd.Series(labels, index=respondent_ids_source).to_dict()
    mapped = df_target["respondent_id"].map(id_to_label).to_numpy()
    return mapped.astype(int)


def plot_raw_feature_panel(df_raw, labels_aligned, features, suptitle, ncols=3, color_map=None, recode_map=None, legend_mode="shared", legend_order=None):
    n = len(features)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.6 * nrows), dpi=150)
    axes = np.asarray(axes).ravel()

    clusters = sorted(np.unique(labels_aligned).tolist())
    x = np.arange(len(clusters))

    recode_map = {} if recode_map is None else recode_map
    legend_order = [] if legend_order is None else legend_order
    def _normalize_for_plot(series):
        series = series.astype(object)
        series[pd.isna(series)] = "NA"
        return series.astype(str).str.strip().replace({"": "NA"})

    series_by_feature = {
        feat: _normalize_for_plot(recode_map.get(feat, lambda s: s)(df_raw[feat].copy()))
        for feat in features
    }
    all_categories = []
    for feat in features:
        all_categories.extend(series_by_feature[feat].value_counts(dropna=False).index.tolist())
    all_categories = list(dict.fromkeys(all_categories))


    color_for_cat = dict(color_map)
    palette = plt.get_cmap("tab20").colors
    for idx, c in enumerate([c for c in all_categories if c not in color_for_cat]):
        color_for_cat[c] = palette[idx % len(palette)]

    for i, feat in enumerate(features):
        ax = axes[i]
        series = series_by_feature[feat]

        overall_counts = series.value_counts(dropna=False)
        categories = overall_counts.index.tolist()

        mat = []
        for cluster_id in clusters:
            cluster_series = series[labels_aligned == cluster_id]
            cluster_value_counts = cluster_series.value_counts(dropna=False)
            total = len(cluster_series)
            props = [float(cluster_value_counts.get(cat, 0)) / total for cat in categories]
            mat.append(props)
        mat = np.array(mat, dtype=float)

        bottom = np.zeros(len(clusters), dtype=float)
        for j, cat in enumerate(categories):
            color = color_for_cat.get(cat)
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

    ordered = [c for c in legend_order if c in all_categories]
    extras = [c for c in all_categories if c not in ordered]
    legend_cats = ordered + extras
    legend_handles = [mpatches.Patch(color=color_for_cat[c], label=c) for c in legend_cats]
    if legend_mode == "shared":
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
    plt.show()
    plt.close(fig)


def make_all_raw_feature_plots(df_raw, labels_aligned):
    raw_global_color_map = {}
    for color, labels in [
        ("#2ca02c", ["Yes", "1", "Always", "100-500", "None of them", "Yes, all", "easy", "open"]),
        ("#d62728", ["No", "0", "Never", "6-25", "Yes, all of them", "difficult", "No or not eligible for coverage", "not open"]),
        ("#1f77b4", ["Maybe", "Sometimes", "1-5", "Some of them", "Maybe/Not sure", "I am not sure"]),
        ("#ff7f0e", ["Some", "Yes, I observed"]),
        ("#9467bd", ["Only became aware later", "No response", "Neither easy nor difficult"]),
        ("#4aa3df", ["I don't know"]),
        ("#f1c40f", ["Neutral"]),
        ("#8c564b", ["Yes, I experienced"]),
        ("#7f7f7f", ["Not applicable"]),
    ]:
        raw_global_color_map.update({label: color for label in labels})

    def _normalize_text_with_na(s, na_label):
        return s.astype(object).where(~pd.isna(s), other=na_label).astype(str).str.strip().replace({"": na_label})


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
    plot_raw_feature_panel(
        df_raw=df_raw,
        labels_aligned=labels_aligned,
        features=panel_features,
        suptitle="Feature distributions across 6 clusters (weak separators — mostly uniform distributions)",
        ncols=3,
        color_map=raw_global_color_map,
        legend_mode="per_subplot",
    )


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

    recode_benefits_map = {"Not eligible for coverage / N/A": "No or not eligible for coverage", "No": "No or not eligible for coverage"}
    recode_leave_easy_map = {"Very difficult": "difficult", "Somewhat difficult": "difficult", "Very easy": "easy", "Somewhat easy": "easy"}

    def recode_benefits(s):
        return _normalize_text_with_na(s, "NA").replace(recode_benefits_map)

    def recode_leave_easy(s):
        return _normalize_text_with_na(s, "NA").replace(recode_leave_easy_map)

    recode_map = {
        "benefits": recode_benefits,
        "leave_easy": recode_leave_easy,
    }
    plot_raw_feature_panel(
        df_raw=df_raw,
        labels_aligned=labels_aligned,
        features=panel_features,
        suptitle="Feature distributions across 6 clusters (strong separators)",
        ncols=3,
        color_map=raw_global_color_map,
        recode_map=recode_map,
        legend_mode="shared",
    )


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

    recode_prev_map = {
        raw: canonical
        for canonical, raws in {
            "No": ["No, none did", "None of them", "No, at none of my previous employers", "No", "N/A (not currently aware)", "None did"],
            "Yes, all": ["Yes, they all did", "Yes, I was aware of all of them", "Yes, all of them", "Yes, at all of my previous employers", "Yes, always", "yes, always"],
            "Some": ["Some did", "I was aware of some", "Some of them", "Some of my previous employers", "Sometimes"],
            "Only became aware later": ["No, I only became aware later"],
        }.items()
        for raw in raws
    }

    def recode_prev_answers(s):
        return _normalize_text_with_na(s, "Not applicable").replace(recode_prev_map)

    recode_map = {feat: recode_prev_answers for feat in panel_features}
    legend_order = ["Yes, all", "Some", "No", "Only became aware later", "I don't know", "Not applicable"]
    plot_raw_feature_panel(
        df_raw=df_raw,
        labels_aligned=labels_aligned,
        features=panel_features,
        suptitle="Previous employment feature distributions across 6 clusters",
        ncols=3,
        color_map=raw_global_color_map,
        recode_map=recode_map,
        legend_mode="shared",
        legend_order=legend_order,
    )


    panel_features = [
        "prev_observed_bad_conseq_mh",
        "mh_family_history",
        "mhdcoworker_you_not_reveal",
        "friends_family_mhd_comfort",
        "ever_observed_mhd_bad_response",
        "boss_mh_discuss",
    ]

    recode_stigma_map = {
        raw: canonical
        for canonical, raws in {
            "Not applicable": ["", "Not applicable to me (I do not have a mental illness)"],
            "open": ["Somewhat open", "Very open"],
            "not open": ["Somewhat not open", "Not open at all"],
        }.items()
        for raw in raws
    }

    def recode_stigma(s):
        return _normalize_text_with_na(s, "Not applicable").replace(recode_stigma_map)

    recode_map = {feat: recode_stigma for feat in panel_features}
    plot_raw_feature_panel(
        df_raw=df_raw,
        labels_aligned=labels_aligned,
        features=panel_features,
        suptitle="Distribution of stigma-related indicators across 6 clusters",
        ncols=3,
        color_map=raw_global_color_map,
        recode_map=recode_map,
        legend_mode="per_subplot",
    )


def build_raw_cluster_vs_rest_tables(df_raw, labels_aligned, top_n=15, exclude_answer_substrings=None):
    df_norm, features = _normalize_categorical_frame(df_raw)
    clusters = sorted(np.unique(labels_aligned).tolist())
    results = {}
    exclude_answer_substrings = {} if exclude_answer_substrings is None else exclude_answer_substrings

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
            denom_c = float(len(s_c))
            denom_r = float(len(s_r))

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
        subs = [s.strip().lower() for s in exclude_answer_substrings.get(int(c), [])]
        df_all = df_all[df_all["feature"].str.lower().map(lambda lx: not any(sub in lx for sub in subs))]
        df = df_all.head(top_n).drop(columns="_diff")
        results[int(c)] = df.reset_index(drop=True)

    return results


def build_raw_cluster_vs_rest_selected(df_raw, labels_aligned, cluster_id, selected_features):
    df_norm, features = _normalize_categorical_frame(df_raw)
    selected_lc = {x.strip().lower() for x in selected_features}

    in_cluster = labels_aligned == cluster_id
    rows = []
    for feat in features:
        s_all = df_norm[feat]
        cats = s_all.value_counts(dropna=False).index.tolist()

        s_c = s_all[in_cluster]
        s_r = s_all[~in_cluster]
        vc_c = s_c.value_counts(dropna=False)
        vc_r = s_r.value_counts(dropna=False)
        denom_c = float(len(s_c))
        denom_r = float(len(s_r))

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
    df_norm, features = _normalize_categorical_frame(df_raw)
    results = {}

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
            denom1 = float(len(s_1))
            denom2 = float(len(s_2))

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


def pam_cluster_profiles_main():
    labels = pam_labels_by_k[K].astype(int)

    clusters = sorted(np.unique(labels))

    size_rows = [
        {
            "cluster": int(c),
            "count": int(np.sum(labels == c)),
            "percent": round(int(np.sum(labels == c)) / len(labels) * 100, 1),
        }
        for c in clusters
    ]
    size_df = pd.DataFrame(size_rows)
    display(HTML("<b>k=6 Cluster Sizes</b>" + size_df.to_html(index=False)))


    df_raw = df_drivers.copy()
    labels_raw = _align_labels_by_respondent_id(df_raw, labels, out["respondent_id"])
    make_all_raw_feature_plots(df_raw, labels_raw)


    raw_all = _build_raw_pairwise_diff_matrix(df_raw, labels_raw)
    heatmap_data = raw_all.loc[raw_all.max(axis=1).sort_values(ascending=False).head(30).index]
    n_cols = max(1, len(heatmap_data.columns))
    fig_width = max(10, n_cols * 0.8)
    fig_height = max(6, len(heatmap_data) * 0.35)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(heatmap_data, cmap="viridis")
    plt.title("Raw feature separation (top 30 by max diff) — k=6")
    plt.xticks(rotation=90, ha="center")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()


    exclude_subs = {
        1: ["not applicable"],
    }
    raw_tables = build_raw_cluster_vs_rest_tables(
        df_raw,
        labels_raw,
        exclude_answer_substrings=exclude_subs,
    )
    selected_by_cluster = {
        0: [
            "prev_boss=Yes",
            "bad_conseq_mh_boss=Maybe",
            "anonymity_protected=I don't know",
            "prev_boss_mh_discuss=None did",
            "boss_mh_discuss=No",
        ],
        1: [
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
        ],
        2: [
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
        ],
        3: [
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
        ],
        4: [
            "prev_anonymity_protected=I don't know",
            "anonymity_protected=I don't know",
            "boss_mh_discuss=No",
            "bad_conseq_mh_boss=No",
            "bad_conseq_mh_boss=Maybe",
            "prev_observed_bad_conseq_mh=None of them",
            "mh_family_history=No",
        ],
        5: [
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
        ],
    }
    for c in sorted(raw_tables.keys()):
        display(HTML(f"<b>CLUSTER {c} vs REST (Top deviations)</b>" + raw_tables[c].to_html(index=False)))
        forced_df = build_raw_cluster_vs_rest_selected(
            df_raw=df_raw,
            labels_aligned=labels_raw,
            cluster_id=c,
            selected_features=selected_by_cluster[c],
        )
        display(HTML(f"<b>CLUSTER {c} vs REST (Selected features)</b>" + forced_df.to_html(index=False)))


    pairs = [(0, 1), (2, 5), (3, 4)]
    pair_tables = build_raw_pairwise_tables(df_raw, labels_raw, pairs=pairs)
    for c1, c2 in pairs:
        title = f"CLUSTER {c1} vs CLUSTER {c2} (Top deviations)"
        display(HTML("<b>" + title + "</b>" + pair_tables[(c1, c2)].to_html(index=False)))


pam_cluster_profiles_main()

#%% [markdown]
# ### 11. Overlay Profiles
# Clusters are characterized using overlay variables that were excluded from the clustering drivers.

#%% 


def normalize_text_for_matching(x):
    s = str(x).strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("\u00a0", " ")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


GENDER_TARGET_ORDER = [
    "Male",
    "Female",
    "Transgender Male",
    "Transgender Female",
    "Nonbinary / Gender Diverse",
    "Other",
    "Prefer Not to Say / Invalid Response",
]

TRANS_MALE_MARKERS = ["ftm", "f2m", "trans man", "transman", "trans male", "transmale", "male (trans", "transmasc", "trans masc"]
TRANS_FEMALE_MARKERS = ["mtf", "m2f", "trans woman", "transwoman", "trans female", "transfemale", "transfeminine", "trans feminine", "other/transfeminine"]
NONBINARY_MARKERS = [
    "nonbinary", "non-binary", "enby", "genderqueer", "agender", "genderfluid",
    "bigender", "androgynous", "genderflux", "multi-gender", "multigender", "gender diverse",
]

def recode_gender(val):
    s = normalize_text_for_matching(val)

    sp = re.sub(r"[^a-z0-9\s\-\/]", " ", s)
    sp = re.sub(r"\s+", " ", sp).strip()

    if any(marker in sp for marker in TRANS_MALE_MARKERS):
        return "Transgender Male"
    if any(marker in sp for marker in TRANS_FEMALE_MARKERS):
        return "Transgender Female"

    if any(marker in sp for marker in NONBINARY_MARKERS):
        return "Nonbinary / Gender Diverse"
    toks = sp.split()
    if "nb" in toks or any(t.startswith("nb") for t in toks):
        return "Nonbinary / Gender Diverse"
    if "queer" in toks and ("male" not in toks and "female" not in toks):
        return "Nonbinary / Gender Diverse"

    tokens = set(toks)


    if sp == "mail":
        return "Male"


    if ("male" in tokens) or ("man" in tokens) or ("dude" in tokens) or (sp == "m") or ("sex is male" in sp) or ("cis male" in sp) or ("cis man" in sp):
        if "female" not in tokens and "woman" not in tokens:
            return "Male"

    if ("female" in tokens) or ("woman" in tokens) or (sp == "f") or ("cis female" in sp) or ("cis-woman" in sp) or ("cisgender female" in sp):
        if "male" not in tokens and "man" not in tokens:
            return "Female"

    return "Other"


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
REGION_SETS = [
    (EU_COUNTRIES, "European Union"),
    (OTHER_EUROPE, "Other Europe"),
    (MIDDLE_EAST, "Middle East"),
    (OCEANIA, "Oceania"),
    (ASIA, "Asia"),
    (AFRICA, "Africa"),
    (LATAM, "Latin America & Caribbean"),
]

def recode_country_region(val):
    s = normalize_text_for_matching(val)

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

    for region_set, region_label in REGION_SETS:
        if s in region_set:
            return region_label

    if "united states" in s:
        return "US"
    if "kingdom" in s:
        return "UK"

    return "Other/NA"


MED_GROUPS_ORDER = [
    "No diagnosis/NA",
    "Mood disorder",
    "anxiety disorder",
    "other",
]

def recode_med_condition(val):
    s = normalize_text_for_matching(val)


    if any(k in s for k in ["no diagnosis", "no dx", "none", "healthy", "no condition", "no mental"]):
        return "No diagnosis/NA"

    if ("mood" in s) and ("disorder" in s):
        return "Mood disorder"
    if ("anxiety" in s) and ("disorder" in s):
        return "anxiety disorder"

    return "other"


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

WORK_POSITION_REPLACEMENTS = {
    "decops": "devops",
    "front'end": "front-end",
    "frontend": "front-end",
    "backend": "back-end",
    "sys admin": "sysadmin",
    "sys-admin": "sysadmin",
}

WORK_CLASSIFIER_RULES = [
    ("Sales", lambda s: "sales" in s),
    ("Dev Evangelist/Advocate", lambda s: any(n in s for n in ["dev evangelist", "developer evangelist", "advocat", "advocate"])),
    ("Leadership (Supervisor/Exec)", lambda s: any(n in s for n in ["supervisor/team lead", "team lead", "supervisor", "executive leadership", "executive"])),
    ("DevOps/SysAdmin", lambda s: any(n in s for n in ["devops", "sysadmin", "sys admin", "sys-admin"])),
    ("Support", lambda s: "support" in s),
]

def recode_work_position(val):
    s = normalize_text_for_matching(val)

    for old, new in WORK_POSITION_REPLACEMENTS.items():
        s = s.replace(old, new)

    has_fe = any(marker in s for marker in ["front-end developer", "front end developer", "front-end"])
    has_be = any(marker in s for marker in ["back-end developer", "back end developer", "back-end"])
    has_designer = "designer" in s

    for label, predicate in WORK_CLASSIFIER_RULES:
        if predicate(s):
            return label
    if has_fe and has_be:
        return "Full-stack (FE+BE)"
    if has_fe and not has_be:
        return "Front-end Developer"
    if has_be and not has_fe:
        return "Back-end Developer"
    if has_designer:
        return "Designer"
    return "other"


def _normalize_overlay_categorical(series):
    return series.astype(object).where(series.notna(), "NA").astype(str).str.strip().replace({"": "NA"})


def plot_categorical_stacked_by_cluster(series, labels, title, category_order=(), legend_title="Answer", min_prop_to_keep=0.0, max_categories=10**9, ax=None):
    s = _normalize_overlay_categorical(series)

    overall = s.value_counts(dropna=False)
    total_n = float(len(s))
    cats = overall.index.tolist()


    keep = set(cats[:max_categories])
    s = s.where(s.isin(keep), other="Other")
    overall = s.value_counts(dropna=False)
    cats = overall.index.tolist()

    keep = set(overall[(overall / total_n) >= min_prop_to_keep].index.tolist())
    s = s.where(s.isin(keep), other="Other")
    overall = s.value_counts(dropna=False)
    cats = overall.index.tolist()

    ordered = [c for c in category_order if c in cats]
    extras = [c for c in cats if c not in ordered]
    cats = ordered + extras

    clusters = sorted(np.unique(labels).astype(int).tolist())

    mat = []
    for cluster_id in clusters:
        cluster_series = s[labels == cluster_id]
        cluster_value_counts = cluster_series.value_counts(dropna=False)
        denom = float(len(cluster_series))
        mat.append([float(cluster_value_counts.get(cat, 0)) / denom for cat in cats])

    mat = np.asarray(mat, dtype=float)

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
    for cluster_id in clusters:
        row = {"cluster": cluster_id}
        for feat in features:
            s = _normalize_overlay_categorical(df[feat])
            cluster_series = s[labels == cluster_id]
            denom = float(len(cluster_series))
            cluster_value_counts = cluster_series.value_counts(dropna=False)
            top_val = str(cluster_value_counts.index[0])
            pct = (cluster_value_counts.iloc[0] / denom) * 100.0
            row[feat] = f"{top_val} ({pct:.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_answer_pct_table(series, labels):
    s = _normalize_overlay_categorical(series)
    clusters = sorted(np.unique(labels).astype(int).tolist())
    cats = s.value_counts(dropna=False).index.tolist()
    rows = []
    for cluster_id in clusters:
        cluster_series = s[labels == cluster_id]
        denom = float(len(cluster_series))
        cluster_value_counts = cluster_series.value_counts(dropna=False)
        row = {"cluster": cluster_id}
        for cat in cats:
            row[str(cat)] = f"{(cluster_value_counts.get(cat, 0) / denom) * 100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


#%% 
OVERLAY_K = 6

df_over = df_overlays.copy()
labels = pam_labels_by_k[OVERLAY_K].astype(int)
labels_aligned = _align_labels_by_respondent_id(df_over, labels, out["respondent_id"])
df = df_over.copy()
for col, fn in {
    "gender": recode_gender,
    "age": recode_age,
    "country_live": recode_country_region,
    "med_pro_condition": recode_med_condition,
    "work_position": recode_work_position,
}.items():
    df[col] = df[col].apply(fn)


#%%
overlay_panels = [
    [("age", {"category_order": AGE_LABELS + ["NA"]}), ("gender", {"category_order": GENDER_TARGET_ORDER}), ("country_work", {"max_categories": 15, "min_prop_to_keep": 0.01}), ("work_position", {"category_order": WORK_ORDER})],
    ["mhd_past", "current_mhd", "pro_treatment", "no_treat_mhd_bad_work"],
    ["mhd_hurt_career", "coworkers_view_neg_mhd", ("med_pro_condition", {"category_order": MED_GROUPS_ORDER}), "mhd_by_med_pro"],
]
for panel_specs in overlay_panels:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=150)
    axes = axes.ravel()
    for ax, spec in zip(axes, panel_specs):
        feature, kwargs = spec if isinstance(spec, tuple) else (spec, {})
        plot_categorical_stacked_by_cluster(
            series=df[feature],
            labels=labels_aligned,
            title=f"Overlay distribution by cluster (k={OVERLAY_K}): {feature}",
            ax=ax,
            **kwargs,
        )
    plt.tight_layout()
    plt.show()
    plt.close(fig)

#%% 
dom_feats = ["mhd_by_med_pro", "pro_treatment", "mhd_past", "current_mhd", "med_pro_condition"]
dom_df = dominant_answer_table(df, labels_aligned, dom_feats)
display(HTML("<b>Dominant answers (overlays)</b>" + dom_df.to_html(index=False)))

pct_feats = ["no_treat_mhd_bad_work", "mhd_hurt_career", "coworkers_view_neg_mhd"]
for feat in pct_feats:
    pct_df = cluster_answer_pct_table(df[feat], labels_aligned)
    display(HTML(f"<b>{feat}</b>" + pct_df.to_html(index=False)))
