import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

import hdbscan
import umap


DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "mental-heath-in-tech-2016_20161114.csv"
RNG_SEED = 0
NA_TOKEN = "__not_applicable__"
MISSING_TOKEN = "__missing__"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def snake_case(text: str) -> str:
    t = str(text).strip().lower()
    t = re.sub(r"[\u00a0\s]+", " ", t)
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        return "col"
    if t[0].isdigit():
        t = f"q_{t}"
    return t


def normalize_gender(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"", "nan", "none"}:
        return None
    s = re.sub(r"[,;:/\\\\|\\[\\]()\"']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if "female" in s or s in {"f", "woman", "cis female", "cis-female", "cis woman", "ciswoman"}:
        return "female"
    if s in {"m", "man", "cis male", "cis-male", "cis man", "cisman"}:
        return "male"
    if "male" in s:
        return "male"
    if "nonbinary" in s or "non-binary" in s or "non binary" in s or "genderqueer" in s or "enby" in s:
        return "non_binary"
    if "trans" in s:
        return "trans"
    return "other"


def split_multiselect(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def build_question_measure_map():
    return [
        {"measure": "self_employed", "question": "01: Are you self-employed?"},
        {"measure": "company_size", "question": "02: How many employees does your company or organization have?"},
        {"measure": "tech_company", "question": "03: Is your employer primarily a tech company/organization?"},
        {"measure": "it_worker", "question": "04: Is your primary role within your company related to tech/IT?"},
        {"measure": "benefits", "question": "05: Does your employer provide mental health benefits as part of healthcare coverage?"},
        {"measure": "benefits_options_known", "question": "06: Do you know the options for mental health care available under your employer-provided coverage?"},
        {
            "measure": "formal_discussion",
            "question": "07: Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?",
        },
        {"measure": "resources_available", "question": "08: Does your employer offer resources to learn more about mental health concerns and options for seeking help?"},
        {
            "measure": "anonymity_protected",
            "question": "09: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?",
        },
        {"measure": "leave_ease", "question": "10: If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:"},
        {"measure": "mental_health_consequences", "question": "11: Do you think that discussing a mental health disorder with your employer would have negative consequences?"},
        {"measure": "physical_health_consequences", "question": "12: Do you think that discussing a physical health issue with your employer would have negative consequences?"},
        {"measure": "coworker_comfort", "question": "13: Would you feel comfortable discussing a mental health disorder with your coworkers?"},
        {"measure": "supervisor_comfort", "question": "14: Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"},
        {"measure": "employer_serious", "question": "15: Do you feel that your employer takes mental health as seriously as physical health?"},
        {
            "measure": "observed_negative_consequences",
            "question": "16: Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?",
        },
        {"measure": "medical_coverage_mh", "question": "17: Do you have medical coverage (private insurance or state-provided) which includes treatment of mental health issues?"},
        {"measure": "external_resources_known", "question": "18: Do you know local or online resources to seek help for a mental health disorder?"},
        {
            "measure": "reveal_to_clients",
            "question": "19: If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?",
        },
        {"measure": "client_reveal_impact", "question": "20: If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?"},
        {"measure": "reveal_to_coworkers", "question": "21: If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?"},
        {"measure": "coworker_reveal_impact", "question": "22: If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?"},
        {"measure": "productivity_affected", "question": "23: Do you believe your productivity is ever affected by a mental health issue?"},
        {"measure": "productivity_impact_percent", "question": "24: If yes, what percentage of your work time is affected by a mental health issue?"},
        {"measure": "previous_employers", "question": "25: Do you have previous employers?"},
        {"measure": "prev_benefits", "question": "26: Have your previous employers provided mental health benefits?"},
        {"measure": "prev_benefits_known", "question": "27: Were you aware of the options for mental health care provided by your previous employers?"},
        {"measure": "prev_formal_discussion", "question": "28: Did your previous employers ever formally discuss mental health?"},
        {"measure": "prev_resources_available", "question": "29: Did your previous employers provide resources to learn more about mental health issues?"},
        {"measure": "prev_anonymity_protected", "question": "30: Was your anonymity protected with previous employers?"},
        {"measure": "prev_mh_consequences", "question": "31: Do you think that discussing a mental health disorder with previous employers would have negative consequences?"},
        {"measure": "prev_ph_consequences", "question": "32: Do you think that discussing a physical health issue with previous employers would have negative consequences?"},
        {"measure": "prev_coworker_comfort", "question": "33: Would you have been willing to discuss a mental health issue with your previous co-workers?"},
        {"measure": "prev_supervisor_comfort", "question": "34: Would you have been willing to discuss a mental health issue with your direct supervisor(s)?"},
        {"measure": "prev_employer_serious", "question": "35: Did you feel that your previous employers took mental health as seriously as physical health?"},
        {"measure": "prev_observed_consequences", "question": "36: Did you hear of or observe negative consequences for co-workers in your previous workplaces?"},
        {"measure": "interview_ph_reveal", "question": "37: Would you be willing to bring up a physical health issue with a potential employer in an interview?"},
        {"measure": "interview_ph_why", "question": "38: Why or why not?"},
        {"measure": "interview_mh_reveal", "question": "39: Would you bring up a mental health issue with a potential employer in an interview?"},
        {"measure": "interview_mh_why", "question": "40: Why or why not?"},
        {"measure": "career_harm", "question": "41: Do you feel that being identified as a person with a mental health issue would hurt your career?"},
        {"measure": "team_views_negative", "question": "42: Do you think that team members/co-workers would view you more negatively?"},
        {"measure": "family_friends_share", "question": "43: How willing would you be to share with friends and family that you have a mental illness?"},
        {"measure": "unsupportive_response_observed", "question": "44: Have you observed or experienced an unsupportive response in your workplace?"},
        {"measure": "reveal_deterred_by_others", "question": "45: Have observations of others made you less likely to reveal your own issue?"},
        {"measure": "family_history", "question": "46: Do you have a family history of mental illness?"},
        {"measure": "past_disorder", "question": "47: Have you had a mental health disorder in the past?"},
        {"measure": "current_disorder", "question": "48: Do you currently have a mental health disorder?"},
        {"measure": "cond_dx_current", "question": "49: If yes, what condition(s) have you been diagnosed with?"},
        {"measure": "cond_believe_current", "question": "50: If maybe, what condition(s) do you believe you have?"},
        {"measure": "diagnosed", "question": "51: Have you been diagnosed with a mental health condition by a medical professional?"},
        {"measure": "cond_dx_previous", "question": "52: If so, what condition(s) were you diagnosed with?"},
        {"measure": "treatment_sought", "question": "53: Have you ever sought treatment for a mental health issue from a mental health professional?"},
        {"measure": "work_interference_treated", "question": "54: If you have a mental health issue, does it interfere with work when treated effectively?"},
        {"measure": "work_interference_untreated", "question": "55: If you have a mental health issue, does it interfere with work when NOT being treated effectively?"},
        {"measure": "age", "question": "56: What is your age?"},
        {"measure": "gender", "question": "57: What is your gender?"},
        {"measure": "country_live", "question": "58: What country do you live in?"},
        {"measure": "state_live", "question": "59: What US state or territory do you live in?"},
        {"measure": "country_work", "question": "60: What country do you work in?"},
        {"measure": "state_work", "question": "61: What US state or territory do you work in?"},
        {"measure": "work_position", "question": "62: Which of the following best describes your work position?"},
        {"measure": "remote_work", "question": "63: Do you work remotely?"},
    ]


def load_df_raw(path: str = DATA_CSV):
    df_raw = pd.read_csv(path)
    qmm = build_question_measure_map()
    if len(qmm) != df_raw.shape[1]:
        raise ValueError(f"question_measure_map has {len(qmm)} entries but dataset has {df_raw.shape[1]} columns")
    df = df_raw.copy()
    df.columns = [d["measure"] for d in qmm]
    col_map = pd.DataFrame(
        {
            "idx": list(range(1, len(qmm) + 1)),
            "question_raw": df_raw.columns,
            "question_label": [d["question"] for d in qmm],
            "measure": df.columns,
        }
    )
    return df, df_raw, col_map


def as_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return None
    return str(x)


def mean_ignore_none(values):
    arr = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not arr:
        return None, 0
    return float(np.mean(arr)), int(len(arr))


def score_yes_no_maybe(x):
    s = as_str(x)
    if s is None or s == NA_TOKEN:
        return None
    s = s.strip()
    if s == "Yes":
        return 1.0
    if s == "No":
        return 0.0
    if s == "Maybe":
        return 0.5
    return None


def score_yes_no(x):
    s = as_str(x)
    if s is None or s == NA_TOKEN:
        return None
    s = s.strip()
    if s == "Yes":
        return 1.0
    if s == "No":
        return 0.0
    return None


def score_negative_consequence(x):
    s = as_str(x)
    if s is None or s == NA_TOKEN:
        return None
    s = s.strip()
    if s == "No":
        return 1.0
    if s == "Maybe":
        return 0.5
    if s == "Yes":
        return 0.0
    return None


def score_leave_ease(x):
    s = as_str(x)
    if s is None or s == NA_TOKEN:
        return None
    s = s.strip()
    mapping = {
        "Very difficult": 0.0,
        "Somewhat difficult": 0.25,
        "Neither easy nor difficult": 0.5,
        "Somewhat easy": 0.75,
        "Very easy": 1.0,
    }
    return mapping.get(s)


def score_employer_seriousness(x):
    s = as_str(x)
    if s is None or s == NA_TOKEN:
        return None
    s = s.strip()
    if s == "Yes":
        return 1.0
    if s == "No":
        return 0.0
    return None


def score_career_harm(x):
    s = as_str(x)
    if s is None or s == NA_TOKEN:
        return None
    s = s.strip()
    if s in {"No, I don't think it would", "No, it has not"}:
        return 1.0
    if s == "Maybe":
        return 0.5
    if s in {"Yes, I think it would", "Yes, it has"}:
        return 0.0
    return None


def score_team_negative(x):
    s = as_str(x)
    if s is None or s == NA_TOKEN:
        return None
    s = s.strip()
    if s in {"No, I don't think they would", "No, they do not"}:
        return 1.0
    if s == "Maybe":
        return 0.5
    if s in {"Yes, I think they would", "Yes, they do"}:
        return 0.0
    return None


def build_df_feat(path: str = DATA_CSV):
    df, _, _ = load_df_raw(path)
    df2 = df.copy()

    age_num = pd.to_numeric(df2["age"], errors="coerce")
    df2["age_out_of_range"] = (~age_num.between(16, 80)) & age_num.notna()
    df2["age_clean"] = age_num.where(age_num.between(16, 80))
    df2["gender_norm"] = df2["gender"].map(normalize_gender) if "gender" in df2.columns else None

    if "work_position" in df2.columns:
        role_base = sorted({r for x in df2["work_position"].dropna().astype(str) for r in split_multiselect(x)})
        for r in role_base:
            df2[f"role__{snake_case(r)}"] = df2["work_position"].map(lambda x, rr=r: int(rr in split_multiselect(x)))
        role_cols = [c for c in df2.columns if c.startswith("role__")]
        df2["role__count"] = df2[role_cols].sum(axis=1)
        df2["role__multi_role"] = (df2["role__count"] >= 2).astype(int)

    cond_text_cols = ["cond_dx_current", "cond_believe_current", "cond_dx_previous"]
    present_cond = [c for c in cond_text_cols if c in df2.columns]
    cond_base = sorted({r for c in present_cond for x in df2[c].dropna().astype(str) for r in split_multiselect(x)})
    for cond in cond_base:
        col = f"cond__{snake_case(cond)}"
        df2[col] = 0
        for c in present_cond:
            df2[col] = df2[col] | df2[c].map(lambda x, cc=cond: int(cc in split_multiselect(x)))
    cond_cols = [c for c in df2.columns if c.startswith("cond__")]
    df2["cond__any_reported"] = df2[cond_cols].max(axis=1) if cond_cols else 0

    for c in df2.columns:
        if df2[c].dtype == object:
            df2[c] = df2[c].astype("string").str.strip()

    df_emp = df2[df2["self_employed"] == 0].copy() if "self_employed" in df2.columns else df2.copy()

    if "tech_company" in df_emp.columns and "it_worker" in df_emp.columns:
        tech_oriented = df_emp["tech_company"].fillna(0).eq(1) | df_emp["it_worker"].fillna(0).eq(1)
        df_emp = df_emp.loc[tech_oriented].copy()
        df_emp["it_worker"] = df_emp["it_worker"].astype(object)
        df_emp.loc[df_emp["tech_company"].eq(1) & df_emp["it_worker"].isna(), "it_worker"] = NA_TOKEN

    if "previous_employers" in df_emp.columns:
        prev0 = df_emp["previous_employers"] == 0
        prev1 = df_emp["previous_employers"] == 1
        skip_cols = []
        for c in df_emp.columns:
            if pd.api.types.is_numeric_dtype(df_emp[c].dtype):
                continue
            mr0 = float(df_emp.loc[prev0, c].isna().mean())
            mr1 = float(df_emp.loc[prev1, c].isna().mean())
            if mr0 >= 0.95 and mr1 <= 0.05:
                skip_cols.append(c)
        for c in skip_cols:
            df_emp.loc[prev0 & df_emp[c].isna(), c] = NA_TOKEN

    if "benefits" in df_emp.columns and "benefits_options_known" in df_emp.columns:
        b = df_emp["benefits"].astype(object).fillna("").astype(str).str.strip()
        mask = df_emp["benefits_options_known"].isna() & b.isin(["No", "Not eligible for coverage / N/A"])
        df_emp.loc[mask, "benefits_options_known"] = NA_TOKEN

    support_items = [
        ("sc_support__employer_serious", df_emp["employer_serious"].map(score_employer_seriousness)),
        ("sc_support__formal_discussion", df_emp["formal_discussion"].map(score_yes_no_maybe)),
        ("sc_support__resources_available", df_emp["resources_available"].map(score_yes_no_maybe)),
        ("sc_support__benefits", df_emp["benefits"].map(score_yes_no_maybe)),
        ("sc_support__options_known", df_emp["benefits_options_known"].map(score_yes_no)),
        ("sc_support__anonymity_protected", df_emp["anonymity_protected"].map(score_yes_no)),
        ("sc_support__leave_ease", df_emp["leave_ease"].map(score_leave_ease)),
    ]
    for name, series in support_items:
        df_emp[name] = series

    safety_items = [
        ("sc_safety__supervisor_comfort", df_emp["supervisor_comfort"].map(score_yes_no_maybe)),
        ("sc_safety__coworker_comfort", df_emp["coworker_comfort"].map(score_yes_no_maybe)),
        ("sc_safety__mh_consequences", df_emp["mental_health_consequences"].map(score_negative_consequence)),
        ("sc_safety__career_harm", df_emp["career_harm"].map(score_career_harm)),
        ("sc_safety__team_views_negative", df_emp["team_views_negative"].map(score_team_negative)),
    ]
    for name, series in safety_items:
        df_emp[name] = series

    def support_row(row):
        vals = [row.get(k) for k, _ in support_items]
        return mean_ignore_none(vals)

    def safety_row(row):
        vals = [row.get(k) for k, _ in safety_items]
        return mean_ignore_none(vals)

    support_vals = df_emp.apply(support_row, axis=1, result_type="expand")
    df_emp["idx_support"] = support_vals[0]
    df_emp["idx_support_n"] = support_vals[1]

    safety_vals = df_emp.apply(safety_row, axis=1, result_type="expand")
    df_emp["idx_safety"] = safety_vals[0]
    df_emp["idx_safety_n"] = safety_vals[1]

    workplace_block = [
        "benefits",
        "benefits_options_known",
        "formal_discussion",
        "resources_available",
        "anonymity_protected",
        "leave_ease",
        "employer_serious",
        "supervisor_comfort",
        "coworker_comfort",
        "mental_health_consequences",
        "career_harm",
        "team_views_negative",
    ]
    workplace_block = [c for c in workplace_block if c in df_emp.columns]

    unknown_set = {"I don't know", "I am not sure", "Not sure", "Unsure", "I don't know."}

    def qc_unknown(row):
        s = row.astype(object).dropna().astype(str).str.strip()
        return int(s.isin(list(unknown_set)).sum())

    def qc_missing(row):
        return int(row.isna().sum())

    df_emp["qc_unknown_count"] = df_emp[workplace_block].apply(qc_unknown, axis=1)
    df_emp["qc_missing_count"] = df_emp[workplace_block].apply(qc_missing, axis=1)

    mask_benefits_yes = df_emp["benefits"].astype(object).fillna("").astype(str).str.strip().eq("Yes")
    mask_options_yes = df_emp["benefits_options_known"].astype(object).fillna("").astype(str).str.strip().eq("Yes")
    df_emp["gap__benefits_yes_but_not_yes_options"] = (mask_benefits_yes & ~mask_options_yes).astype(int)

    return df_emp


def feature_sets(df_feat: pd.DataFrame):
    workplace_cols = [
        "benefits",
        "benefits_options_known",
        "formal_discussion",
        "resources_available",
        "anonymity_protected",
        "leave_ease",
        "employer_serious",
        "mental_health_consequences",
        "coworker_comfort",
        "supervisor_comfort",
        "career_harm",
        "team_views_negative",
        "company_size",
        "remote_work",
        "gap__benefits_yes_but_not_yes_options",
    ]
    workplace_cols = [c for c in workplace_cols if c in df_feat.columns]

    workplace_no_gap = [c for c in workplace_cols if c != "gap__benefits_yes_but_not_yes_options"]
    role_cols = sorted([c for c in df_feat.columns if c.startswith("role__") and c not in {"role__count", "role__multi_role"}])
    role_summary = [c for c in ["role__count", "role__multi_role"] if c in df_feat.columns]

    scored_cols = sorted([c for c in df_feat.columns if c.startswith("sc_support__") or c.startswith("sc_safety__")])

    return {
        "workplace_no_roles": workplace_cols,
        "workplace_no_roles_no_gap": workplace_no_gap,
        "workplace_plus_roles": workplace_cols + role_cols,
        "workplace_plus_role_summaries": workplace_cols + role_summary,
        "indices_only": [c for c in ["idx_support", "idx_safety", "qc_unknown_count", "qc_missing_count"] if c in df_feat.columns],
        "scored_items": scored_cols + [c for c in ["company_size", "remote_work", "qc_unknown_count", "qc_missing_count", "gap__benefits_yes_but_not_yes_options"] if c in df_feat.columns],
        "scored_items_no_gap": scored_cols + [c for c in ["company_size", "remote_work", "qc_unknown_count", "qc_missing_count"] if c in df_feat.columns],
    }


def build_preprocessor(df: pd.DataFrame, min_frequency: int | None = 10):
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in df.columns if c not in num]
    cat_steps = [("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN))]
    if min_frequency is None:
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
    else:
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency)))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
            ("cat", Pipeline(cat_steps), cat),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def to_object_for_cat(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c].dtype):
            out[c] = out[c].astype(object)
    return out


def embed_svd(Xt, n_components: int = 15):
    d_eff = min(int(n_components), Xt.shape[1] - 1) if Xt.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=d_eff, random_state=RNG_SEED).fit(Xt)
    return svd, svd.transform(Xt)


def eval_kmeans(Z, k):
    km = KMeans(n_clusters=int(k), n_init=30, random_state=RNG_SEED)
    labels = km.fit_predict(Z)
    return {"labels": labels, "silhouette": float(silhouette_score(Z, labels)), "inertia": float(km.inertia_)}


def eval_gmm(Z, k, covariance_type="diag"):
    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type=covariance_type,
        n_init=5,
        reg_covar=1e-6,
        random_state=RNG_SEED,
    )
    labels = gmm.fit_predict(Z)
    return {
        "labels": labels,
        "silhouette": float(silhouette_score(Z, labels)),
        "bic": float(gmm.bic(Z)),
        "aic": float(gmm.aic(Z)),
    }


def eval_hdbscan(Z, min_cluster_size=25, min_samples=None, metric="euclidean"):
    cl = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
        gen_min_span_tree=True,
    )
    labels = cl.fit_predict(Z)
    n = len(labels)
    n_noise = int((labels == -1).sum())
    n_inliers = int(n - n_noise)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    sil_in = np.nan
    if n_clusters >= 2 and n_inliers >= 10:
        sil_in = float(silhouette_score(Z[labels != -1], labels[labels != -1]))
    rel = getattr(cl, "relative_validity_", np.nan)
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "noise_frac": float(n_noise / max(n, 1)),
        "silhouette_inliers": sil_in,
        "relative_validity": float(rel) if rel is not None else np.nan,
    }


def eval_umap_hdbscan(Z, umap_dim=5, n_neighbors=10, min_dist=0.1, min_cluster_size=50, min_samples=None, seed=0):
    Zu = umap.UMAP(
        n_components=int(umap_dim),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(seed),
    ).fit_transform(Z)
    cl = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric="euclidean",
        gen_min_span_tree=True,
    )
    labels = cl.fit_predict(Zu)
    n = len(labels)
    n_noise = int((labels == -1).sum())
    n_inliers = int(n - n_noise)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    sil_umap = np.nan
    sil_svd = np.nan
    if n_clusters >= 2 and n_inliers >= 10:
        sil_umap = float(silhouette_score(Zu[labels != -1], labels[labels != -1]))
        sil_svd = float(silhouette_score(Z[labels != -1], labels[labels != -1]))
    rel = getattr(cl, "relative_validity_", np.nan)
    return {
        "labels": labels,
        "Zu": Zu,
        "n_clusters": n_clusters,
        "noise_frac": float(n_noise / max(n, 1)),
        "silhouette_umap_inliers": sil_umap,
        "silhouette_svd_inliers": sil_svd,
        "relative_validity": float(rel) if rel is not None else np.nan,
    }


def stability_ari(labels_list):
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    return float(np.mean(aris)) if aris else float("nan")


def decision_tree_explainability(X, labels, max_depth=3, n_splits=5, seed=0):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c].dtype):
            X[c] = X[c].astype("string").fillna("__MISSING__")
    labels = np.asarray(labels)
    mask = labels != -1
    y = labels[mask]
    if mask.sum() < 50 or len(set(y)) < 2:
        return {"tree_macro_f1": float("nan"), "tree_n": int(mask.sum())}
    Xn = X.loc[mask].copy()
    num_cols = Xn.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in Xn.columns if c not in num_cols]
    X_enc = pd.get_dummies(Xn, columns=cat_cols, drop_first=False)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed, min_samples_leaf=10)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s = []
    for tr, te in cv.split(X_enc, y):
        clf.fit(X_enc.iloc[tr], y[tr])
        pred = clf.predict(X_enc.iloc[te])
        f1s.append(f1_score(y[te], pred, average="macro"))
    return {"tree_macro_f1": float(np.mean(f1s)), "tree_n": int(mask.sum())}


def top_lift_drivers(df, labels, features, top_n=10):
    labels = np.asarray(labels)
    mask = labels != -1
    d = df.loc[mask].copy()
    labs = labels[mask]

    rows = []
    for feat in features:
        if feat not in d.columns:
            continue
        s = d[feat].astype(object).fillna("__MISSING__").astype(str).str.strip()
        overall = (s.value_counts(normalize=True) * 100.0).to_dict()
        for cl in sorted(set(labs)):
            sc = s[labs == cl]
            dist = (sc.value_counts(normalize=True) * 100.0).to_dict()
            for ans, pct in dist.items():
                rows.append(
                    {
                        "cluster": int(cl),
                        "feature": feat,
                        "answer": str(ans),
                        "pct_in_cluster": float(pct),
                        "pct_overall": float(overall.get(ans, 0.0)),
                        "lift_pp": float(pct - overall.get(ans, 0.0)),
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_lift"] = out["lift_pp"].abs()
    out = out.sort_values(["cluster", "abs_lift"], ascending=[True, False]).groupby("cluster").head(top_n).drop(columns=["abs_lift"])
    return out.reset_index(drop=True)

