"""
Exp 3 subgroup analyses for biomarker distortion across ASR systems.

Subgroups:
  1. Speech task   : MONO (DEM FREE + DEM MONO + OLD STOR) vs PICT (DEM COOK + OLD PICT)
  2. Age group     : <75 vs >=75
  3. Sex           : Male vs Female
  4. Dementia subtype: each DEM subtype vs ALL OLD (DEM-only split)
  5. Transcript length quartile: Q1-Q4 by ground-truth word count

For each subgroup the Exp 3 deviation is recomputed:
  deviation = mean(DEM delta) - mean(OLD delta)
  where delta = ASR_biomarker - Human_biomarker per recording
  95% CI and p-value from Welch t-test on deltas.
  FDR correction (BH) within each subgroup x biomarker block.
"""

import re, warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA   = '/Users/thaya/Desktop/Validation/ASR_Validation/Disflu_detect/V3_downstream/All_data.csv'
META   = '/Users/thaya/Desktop/Validation/ASR_Validation/Meta-Data.xlsx'
OUTDIR = '/Users/thaya/Desktop/Journals/NPJ_DIGITALMED/V1/'

# ── Dementia subtype labels (update if codes differ) ─────────────────────────
SUBTYPE_LABELS = {
    2.0: 'Subtype_2',
    3.0: 'Subtype_3',
    4.0: 'Subtype_4',
    5.0: 'Subtype_5',
    6.0: 'Subtype_6',
    7.0: 'Subtype_7',
}

# ── Biomarker feature extraction (matches main analysis) ─────────────────────
PRONOUNS = {"i","me","my","mine","you","your","yours","he","him","his",
            "she","her","hers","it","its","we","us","our","ours",
            "they","them","their","theirs"}
FUNCTION_WORDS = {"the","a","an","in","on","at","for","to","of","and","but","or",
                  "is","are","was","were","be","been","being"}
BIOMARKERS = ["ttr", "pronoun_ratio", "content_ratio", "mean_word_len"]

def tokenize(text):
    return re.findall(r"\b[a-z']+\b", str(text).lower())

def extract_features(text):
    tokens = tokenize(text)
    if len(tokens) == 0:
        return {b: np.nan for b in BIOMARKERS}
    unique         = len(set(tokens))
    pronoun_count  = sum(t in PRONOUNS for t in tokens)
    function_count = sum(t in FUNCTION_WORDS for t in tokens)
    content_count  = len(tokens) - function_count
    return {
        "ttr":           unique / len(tokens),
        "pronoun_ratio": pronoun_count / len(tokens),
        "content_ratio": content_count / len(tokens),
        "mean_word_len": np.mean([len(t) for t in tokens]),
    }

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df['ParticipantID'] = df['Filename'].str.extract(r'((?:DEM|OLD)\d+)')

# ── Metadata: age, sex, subtype ───────────────────────────────────────────────
meta = pd.read_excel(META)

# DEM metadata: latest timepoint per participant
dem_meta = meta[meta['Name'].str.startswith('DEM', na=False)].copy()
dem_meta['ParticipantID'] = dem_meta['Name'].str.extract(r'(DEM\d+)')
dem_latest = (dem_meta.dropna(subset=['Age'])
                       .sort_values('Age')
                       .groupby('ParticipantID')
                       .last()
                       .reset_index()[['ParticipantID','Age','Sex','Group']]
                       .rename(columns={'Group':'Subtype_code'}))
# Sex: 1=Male, 2=Female in metadata
dem_latest['Sex_label'] = dem_latest['Sex'].map({1:'Male', 2:'Female', '1':'Male', '2':'Female'})

# OLD metadata: match on short key (old16_NNN)
old_meta = meta[meta['Name'].str.startswith('old', na=False)][['Name','Age']].copy()
old_meta['short_key'] = old_meta['Name'].str.extract(r'(old\d+_\d+)')

old_rows = df[df['Group'] == 'OLD'].drop_duplicates('Filename')[['Filename']].copy()
old_rows['short_key']   = old_rows['Filename'].str.extract(r'(OLD\d+_\d+)')[0].str.lower()
old_rows['Sex_label']   = old_rows['Filename'].str.extract(r'_([MF])_')[0].map({'M':'Male','F':'Female'})
old_rows['ParticipantID'] = old_rows['Filename'].str.extract(r'(OLD\d+_\d+_\d+_\d+)')[0]
old_rows = old_rows.merge(old_meta[['short_key','Age']], on='short_key', how='left')

# Merge demographics back onto full df
# DEM
df = df.merge(dem_latest[['ParticipantID','Age','Sex_label','Subtype_code']],
              on='ParticipantID', how='left')
# OLD sex/age (won't overwrite DEM columns - OLD rows have NaN from DEM merge)
df = df.merge(old_rows[['Filename','Age','Sex_label']].rename(
                  columns={'Age':'Age_old','Sex_label':'Sex_old'}),
              on='Filename', how='left')
df['Age']       = df['Age'].fillna(df['Age_old'])
df['Sex_label'] = df['Sex_label'].fillna(df['Sex_old'])
df = df.drop(columns=['Age_old','Sex_old'])

# ── Task remapping ─────────────────────────────────────────────────────────────
task_remap = {
    'FREE': 'MONO',
    'MONO': 'MONO',
    'STOR': 'MONO',
    'COOK': 'PICT',
    'PICT': 'PICT',
}
df['Task_cat'] = df['Task'].map(task_remap)

# ── Transcript length (word count from ground truth) ──────────────────────────
gt_wc = (df[df['ASR_Model'] == df['ASR_Model'].iloc[0]][['Filename','Ground_Truth']]
           .copy())
gt_wc['wc'] = gt_wc['Ground_Truth'].str.split().str.len()
df = df.merge(gt_wc[['Filename','wc']], on='Filename', how='left')
q25, q50, q75 = df['wc'].quantile([0.25, 0.5, 0.75])
def wc_quartile(x):
    if   x <= q25: return 'Q1'
    elif x <= q50: return 'Q2'
    elif x <= q75: return 'Q3'
    else:          return 'Q4'
df['WC_quartile'] = df['wc'].apply(lambda x: wc_quartile(x) if pd.notna(x) else np.nan)

# ── Biomarker features for all rows ──────────────────────────────────────────
for bm in BIOMARKERS:
    df[f'gt_{bm}']  = df['Ground_Truth'].apply(lambda t: extract_features(t)[bm])
    df[f'asr_{bm}'] = df['Prediction'].apply(lambda t: extract_features(t)[bm])
    df[f'delta_{bm}'] = df[f'asr_{bm}'] - df[f'gt_{bm}']

# ── Core analysis function ────────────────────────────────────────────────────
def run_exp3(subset_df, label_col=None):
    """
    For a given subset of recordings, compute Exp 3 biomarker distortion
    per ASR model. Returns a DataFrame of results.
    """
    rows = []
    models = sorted(subset_df['ASR_Model'].unique())
    # Human reference
    human_df = subset_df[subset_df['ASR_Model'] == subset_df['ASR_Model'].iloc[0]].copy()

    for model in models:
        asr_df = subset_df[subset_df['ASR_Model'] == model]
        for bm in BIOMARKERS:
            dem_deltas = asr_df[asr_df['Group'] == 'DEM'][f'delta_{bm}'].dropna()
            old_deltas = asr_df[asr_df['Group'] == 'OLD'][f'delta_{bm}'].dropna()
            if len(dem_deltas) < 3 or len(old_deltas) < 3:
                continue
            deviation = dem_deltas.mean() - old_deltas.mean()
            # Human group effect for this biomarker
            dem_gt = human_df[human_df['Group'] == 'DEM'][f'gt_{bm}'].dropna()
            old_gt = human_df[human_df['Group'] == 'OLD'][f'gt_{bm}'].dropna()
            human_effect = dem_gt.mean() - old_gt.mean()
            asr_dem = asr_df[asr_df['Group'] == 'DEM'][f'asr_{bm}'].dropna()
            asr_old = asr_df[asr_df['Group'] == 'OLD'][f'asr_{bm}'].dropna()
            asr_effect = asr_dem.mean() - asr_old.mean()

            t_stat, p_val = stats.ttest_ind(dem_deltas, old_deltas, equal_var=False)
            se = np.sqrt(dem_deltas.sem()**2 + old_deltas.sem()**2)
            df_t = ((dem_deltas.var()/len(dem_deltas) + old_deltas.var()/len(old_deltas))**2 /
                    ((dem_deltas.var()/len(dem_deltas))**2/(len(dem_deltas)-1) +
                     (old_deltas.var()/len(old_deltas))**2/(len(old_deltas)-1)))
            ci_half = stats.t.ppf(0.975, df_t) * se

            rows.append({
                'ASR_Model': model,
                'Biomarker': bm,
                'Human_effect': round(human_effect, 4),
                'ASR_effect':   round(asr_effect, 4),
                'Deviation':    round(deviation, 4),
                'CI_low':       round(deviation - ci_half, 4),
                'CI_high':      round(deviation + ci_half, 4),
                'p_val':        round(p_val, 4),
                'n_DEM':        len(dem_deltas),
                'n_OLD':        len(old_deltas),
            })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    # FDR correction across all tests in this subgroup
    _, p_fdr, _, _ = multipletests(res['p_val'], method='fdr_bh')
    res['p_FDR'] = np.round(p_fdr, 4)
    return res

# ── Run all subgroup analyses ─────────────────────────────────────────────────
all_results = []

# 1. Speech task
for task_cat in ['MONO', 'PICT']:
    sub = df[df['Task_cat'] == task_cat]
    res = run_exp3(sub)
    res.insert(0, 'Subgroup_value', task_cat)
    res.insert(0, 'Subgroup_type',  'Task')
    all_results.append(res)

# 2. Age group
for age_label, age_mask in [('<75', df['Age'] < 75), ('>=75', df['Age'] >= 75)]:
    sub = df[age_mask]
    res = run_exp3(sub)
    res.insert(0, 'Subgroup_value', age_label)
    res.insert(0, 'Subgroup_type',  'Age')
    all_results.append(res)

# 3. Sex
for sex_val in ['Male', 'Female']:
    sub = df[df['Sex_label'] == sex_val]
    res = run_exp3(sub)
    res.insert(0, 'Subgroup_value', sex_val)
    res.insert(0, 'Subgroup_type',  'Sex')
    all_results.append(res)

# 4. Dementia subtype (each subtype vs ALL OLD)
old_df = df[df['Group'] == 'OLD']
for code, label in SUBTYPE_LABELS.items():
    dem_sub = df[(df['Group'] == 'DEM') & (df['Subtype_code'] == code)]
    if len(dem_sub) == 0:
        continue
    sub = pd.concat([dem_sub, old_df], ignore_index=True)
    res = run_exp3(sub)
    if res.empty:
        continue
    n_dem_pts = dem_sub['ParticipantID'].nunique()
    res.insert(0, 'Subgroup_value', f'{label} (n_pts={n_dem_pts})')
    res.insert(0, 'Subgroup_type',  'Dementia_subtype')
    all_results.append(res)

# 5. Transcript length quartile
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = df[df['WC_quartile'] == q]
    res = run_exp3(sub)
    res.insert(0, 'Subgroup_value', q)
    res.insert(0, 'Subgroup_type',  'Transcript_length')
    all_results.append(res)

# ── Combine and save ──────────────────────────────────────────────────────────
final = pd.concat(all_results, ignore_index=True)
out_csv = OUTDIR + 'exp3_subgroup_results.csv'
final.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')
print(f'Total rows: {len(final)}')

# ── Summary print ─────────────────────────────────────────────────────────────
print('\n=== SUMMARY (significant deviations p_FDR < 0.05) ===')
sig = final[final['p_FDR'] < 0.05]
if len(sig) > 0:
    print(sig[['Subgroup_type','Subgroup_value','ASR_Model','Biomarker',
               'Human_effect','ASR_effect','Deviation','CI_low','CI_high',
               'p_val','p_FDR']].to_string(index=False))
else:
    print('None significant at FDR < 0.05')

print('\n=== FULL RESULTS BY SUBGROUP ===')
for sg_type in final['Subgroup_type'].unique():
    print(f'\n--- {sg_type} ---')
    sub_res = final[final['Subgroup_type'] == sg_type]
    for sg_val in sub_res['Subgroup_value'].unique():
        print(f'\n  {sg_val}:')
        print(sub_res[sub_res['Subgroup_value'] == sg_val][
            ['ASR_Model','Biomarker','Human_effect','ASR_effect',
             'Deviation','CI_low','CI_high','p_val','p_FDR','n_DEM','n_OLD']
        ].to_string(index=False))
