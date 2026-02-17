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

from __future__ import annotations

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


if __name__ == "__main__":
    main()
