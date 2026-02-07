import pandas as pd
from pathlib import Path

# ---- paths ----

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "survey.csv"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "survey_renamed.csv"

# ---- load ----

df = pd.read_csv(RAW_PATH)

# ---- rename mapping ----

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

# ---- apply rename ----

df_renamed = df.rename(columns=rename_map)

# ---- save ----

df_renamed.to_csv(OUT_PATH, index=False)

print("Renamed file saved to:", OUT_PATH)
print("\nNew columns:")
print(list(df_renamed.columns))
