"""
Rerun Experiments 1, 2, and 3 using participant-level cross-validation.
StratifiedGroupKFold ensures all recordings from one participant stay
in the same fold, preventing data leakage across participants.

Old results (recording-level CV) are preserved in results_recording_level_cv/
New results are saved to results_participant_level_cv/
"""

import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

DATA   = "/Users/thaya/Desktop/Validation/ASR_Validation/Disflu_detect/V3_downstream/All_data.csv"
OUT    = "/Users/thaya/Desktop/Validation/ASR_Validation/Disflu_detect/V3_downstream/results_participant_level_cv/"

import os; os.makedirs(OUT, exist_ok=True)

# ── Load and prepare data ────────────────────────────────────────────────────
df = pd.read_csv(DATA)

# Task remapping
task_map = {'COOK': 'PICT', 'FREE': 'MONO', 'STOR': 'MONO', 'MONO': 'MONO', 'PICT': 'PICT'}
df['Task'] = df['Task'].str.strip().str.upper().replace(task_map)
df['Group'] = df['Group'].str.strip().str.upper().replace({'OLD': 'Healthy', 'DEM': 'Dementia'})
df['label'] = df['Group'].map({'Healthy': 0, 'Dementia': 1})

# Repeated n-gram capping
def _tokenize_words(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []

def cap_repeated_ngrams(text, max_repeats=3, min_n=2, max_n=5,
                         also_cap_single_words=True, max_word_repeats=4):
    toks = _tokenize_words(text)
    if not toks:
        return ""
    out = []
    i = 0
    L = len(toks)
    while i < L:
        matched = False
        for n in range(min(max_n, L - i), min_n - 1, -1):
            pattern = toks[i:i+n]
            count = 1
            j = i + n
            while j + n <= L and toks[j:j+n] == pattern:
                count += 1
                j += n
            if count > max_repeats:
                for _ in range(max_repeats):
                    out.extend(pattern)
                i = j
                matched = True
                break
        if not matched:
            if also_cap_single_words:
                count = 1
                j = i + 1
                while j < L and toks[j] == toks[i]:
                    count += 1
                    j += 1
                if count > max_word_repeats:
                    out.extend([toks[i]] * max_word_repeats)
                    i = j
                else:
                    out.append(toks[i])
                    i += 1
            else:
                out.append(toks[i])
                i += 1
    return " ".join(out)

df['Ground_Truth'] = df['Ground_Truth'].apply(cap_repeated_ngrams)
df['Prediction']   = df['Prediction'].apply(cap_repeated_ngrams)

# Build combined transcript df
human_df = df[['Filename','Group','Task','Ground_Truth','label']].drop_duplicates('Filename').copy()
human_df = human_df.rename(columns={'Ground_Truth': 'Transcript'})
human_df['Source'] = 'Human'

asr_df = df[['Filename','Group','Task','Prediction','ASR_Model','label']].copy()
asr_df = asr_df.rename(columns={'Prediction': 'Transcript', 'ASR_Model': 'Source'})

final_df = pd.concat([human_df, asr_df], ignore_index=True)

# ── Extract participant IDs ───────────────────────────────────────────────────
def get_participant_id(fname):
    if fname.startswith('OLD16'):
        parts = fname.split('_')
        return parts[0] + '_' + parts[1]
    m = re.match(r'^([A-Z]+\d+)', fname)
    return m.group(1) if m else fname

human_df = final_df[final_df['Source'] == 'Human'].drop_duplicates('Filename').copy()
human_df['ParticipantID'] = human_df['Filename'].apply(get_participant_id)

X         = human_df['Transcript'].astype(str)
y         = human_df['label'].astype(int)
filenames = human_df['Filename'].astype(str)
groups    = human_df['ParticipantID'].astype(str)

print(f"Recordings: {len(X)}, Participants: {groups.nunique()}, Classes: {y.value_counts().to_dict()}")

# ── CV setup ─────────────────────────────────────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
asr_sources = sorted(final_df['Source'].unique())

# ── Experiment 1 & 2: Classification ─────────────────────────────────────────
print("\nRunning Exp 1 & 2 (participant-level CV)...")

all_preds = []
fold_src_rates = []

for fold_id, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
    train_files = filenames.iloc[train_idx]
    test_files  = filenames.iloc[test_idx]

    # Verify no participant overlap
    train_participants = set(groups.iloc[train_idx])
    test_participants  = set(groups.iloc[test_idx])
    assert len(train_participants & test_participants) == 0, "Participant leakage detected!"

    train_text   = X.iloc[train_idx]
    train_labels = y.iloc[train_idx]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    X_train = vectorizer.fit_transform(train_text)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, train_labels)

    for src in asr_sources:
        test_subset = final_df[
            (final_df['Filename'].isin(test_files)) &
            (final_df['Source'] == src)
        ].copy()

        if len(test_subset) == 0:
            continue

        X_test  = vectorizer.transform(test_subset['Transcript'])
        y_test  = test_subset['label']
        probs   = model.predict_proba(X_test)[:, 1]
        preds   = (probs > 0.5).astype(int)

        try:
            auc  = roc_auc_score(y_test, probs)
            f1   = f1_score(y_test, preds, zero_division=0)
            bacc = balanced_accuracy_score(y_test, preds)
        except Exception:
            auc = f1 = bacc = np.nan

        for fname, yt, yp, prob in zip(test_subset['Filename'], y_test, preds, probs):
            all_preds.append({
                'fold': fold_id, 'Filename': fname, 'Source': src,
                'y_true': yt, 'y_pred': yp, 'prob': prob,
                'Task': test_subset.loc[test_subset['Filename']==fname, 'Task'].values[0]
                if fname in test_subset['Filename'].values else None,
                'Group': test_subset.loc[test_subset['Filename']==fname, 'Group'].values[0]
                if fname in test_subset['Filename'].values else None,
            })

        # Exp 2 rates
        TP = np.sum((y_test == 1) & (preds == 1))
        FN = np.sum((y_test == 1) & (preds == 0))
        TN = np.sum((y_test == 0) & (preds == 0))
        FP = np.sum((y_test == 0) & (preds == 1))
        TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        TNR = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        fold_src_rates.append({
            'fold': fold_id, 'Source': src,
            'TP': TP, 'FN': FN, 'TN': TN, 'FP': FP,
            'TPR_sens': TPR, 'TNR_spec': TNR,
            'FNR': 1 - TPR if not np.isnan(TPR) else np.nan,
            'FPR': 1 - TNR if not np.isnan(TNR) else np.nan,
        })

pred_long = pd.DataFrame(all_preds)
fold_rates_df = pd.DataFrame(fold_src_rates)

# Exp 1 summary
exp1_summary = (pred_long.groupby('Source')
    .apply(lambda g: pd.Series({
        'AUC':  roc_auc_score(g['y_true'], g['prob']) if g['y_true'].nunique() > 1 else np.nan,
        'F1':   f1_score(g['y_true'], g['y_pred'], zero_division=0),
        'BAcc': balanced_accuracy_score(g['y_true'], g['y_pred'])
    })).reset_index())

print("\nExp 1 AUC summary:")
print(exp1_summary.sort_values('AUC', ascending=False).to_string(index=False))

# Exp 2 summary
exp2_summary = fold_rates_df.groupby('Source')[['TPR_sens','TNR_spec','FNR','FPR']].mean().reset_index()
human_row = exp2_summary[exp2_summary['Source'] == 'Human'].iloc[0]
for col in ['TPR_sens','TNR_spec','FNR','FPR']:
    exp2_summary[f'Delta_{col}_vs_Human'] = exp2_summary[col] - human_row[col]

print("\nExp 2 FNR summary:")
print(exp2_summary[['Source','FNR','Delta_FNR_vs_Human']].sort_values('Delta_FNR_vs_Human', ascending=False).to_string(index=False))

# Save Exp 1 & 2
pred_long.to_csv(OUT + "exp1_filewise_predictions_long.csv", index=False)
fold_rates_df.to_csv(OUT + "exp2_fold_source_error_rates.csv", index=False)
exp2_summary.to_csv(OUT + "exp2_error_rates_summary_vs_human.csv", index=False)
print(f"\nExp 1 & 2 saved to {OUT}")

# ── Experiment 3: Biomarkers ──────────────────────────────────────────────────
print("\nRunning Exp 3 (biomarker distortion)...")

FILLED_PAUSES = {"um","uh","ah","er","mm","hmm","hm","eh"}

def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b[a-z']+\b", text)

pronouns = {"i","me","my","mine","you","your","yours","he","him","his",
            "she","her","hers","it","its","we","us","our","ours",
            "they","them","their","theirs"}
function_words = {"the","a","an","in","on","at","for","to","of","and","but","or",
                  "is","are","was","were","be","been","being"}

def extract_features(text):
    tokens = tokenize(text)
    if len(tokens) == 0:
        return pd.Series({"n_words":0,"ttr":0,"pronoun_ratio":0,"content_ratio":0,"mean_word_len":0})
    unique = len(set(tokens))
    pronoun_count  = sum(t in pronouns for t in tokens)
    function_count = sum(t in function_words for t in tokens)
    content_count  = len(tokens) - function_count
    return pd.Series({
        "n_words":       len(tokens),
        "ttr":           unique / len(tokens),
        "pronoun_ratio": pronoun_count / len(tokens),
        "content_ratio": content_count / len(tokens),
        "mean_word_len": np.mean([len(t) for t in tokens])
    })

bio_rows = []
for _, row in final_df.iterrows():
    feats = extract_features(row['Transcript'])
    feats['Filename'] = row['Filename']
    feats['Group']    = row['Group']
    feats['Source']   = row['Source']
    feats['Task']     = row['Task']
    bio_rows.append(feats)

bio_df = pd.DataFrame(bio_rows)

def effect_size(g):
    healthy  = g[g["Group"] == "Healthy"]
    dementia = g[g["Group"] == "Dementia"]
    results  = {}
    for col in ["ttr","pronoun_ratio","content_ratio","mean_word_len"]:
        results[col + "_effect"] = dementia[col].mean() - healthy[col].mean()
    return pd.Series(results)

effects = bio_df.groupby("Source").apply(effect_size).reset_index()
print("\nExp 3 biomarker distortion:")
print(effects.to_string(index=False))

bio_df.to_csv(OUT + "exp3_filewise_predictions_long.csv", index=False)
effects.to_csv(OUT + "exp3_biomarker_distortion.csv", index=False)
print(f"\nExp 3 saved to {OUT}")

print("\n=== All experiments complete. Results in:", OUT)
