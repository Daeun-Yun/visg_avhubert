from plot_utils import *
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, spearmanr
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import os, json
import csv
from tqdm import tqdm
from matplotlib_venn import venn2
import seaborn as sns

model = 'base'
csv_path = os.getcwd() + f'/avhubert/selected_{model}.csv'

df = pd.read_csv(csv_path)   
# noise_type = None
exp = 'Exp5' if model == 'large' else 'Exp7'
experiment_params = {
    'Name': exp
}

save_directory = get_experiment_directory(csv_path, experiment_params, model)

print(f"Saving outputs to: {save_directory}")
# snr = -10
# Initialize totals
error_counts = {
    'babble': {'baseline': {'subs': 0, 'ins': 0, 'dels': 0},
               'viseme': {'subs': 0, 'ins': 0, 'dels': 0}},
    'music': {'baseline': {'subs': 0, 'ins': 0, 'dels': 0},
              'viseme': {'subs': 0, 'ins': 0, 'dels': 0}},
    'speech': {'baseline': {'subs': 0, 'ins': 0, 'dels': 0},
               'viseme': {'subs': 0, 'ins': 0, 'dels': 0}},
    'noise': {'baseline': {'subs': 0, 'ins': 0, 'dels': 0},
              'viseme': {'subs': 0, 'ins': 0, 'dels': 0}}
}

# Store per-dB data
snr_errors = {snr: {'subs': [], 'ins': [], 'dels': []} for snr in [-10, -5, 0, 5, 10]}

for noise_type in ['babble', 'music', 'speech', 'noise']:
    for snr in [-10, -5, 0, 5, 10]:
        noise_path = f'{noise_type}_snr{snr}' if noise_type else None

        if noise_type is not None:
            with open(f'/home/aristosp/models/av_hubert/{model}_decode/{noise_path}/hypo-244018.json', 'r') as baseline_file:
                baseline_data = json.load(baseline_file)
            with open(f'/home/aristosp/viseme_{model}_models/{experiment_params["Name"]}/decode/{noise_path}/hypo-244018.json', 'r') as f2:
                vsm_data = json.load(f2)  
        else:
            with open(f'/home/aristosp/models/av_hubert/{model}_decode/hypo-244018.json', 'r') as baseline_file:
                baseline_data = json.load(baseline_file)
            # Read viseme model JSON
            with open(f'/home/aristosp/viseme_{model}_models/{experiment_params["Name"]}/decode/hypo-244018.json', 'r') as f2:
                vsm_data = json.load(f2)


        utt_ids = baseline_data['utt_id'] # same if read from vsm data

        try:
            assert baseline_data['utt_id'] == vsm_data['utt_id'], "Utterance mismatch between models!"
        except (AssertionError, ValueError):
        # Create a mapping of utt_id positions from vsm_data
            # Mapping of utt_id -> index in vsm_data
            vsm_order = {utt_id: idx for idx, utt_id in enumerate(vsm_data['utt_id'])}

            # Indices to sort baseline_data according to vsm_data
            sorted_indices = sorted(range(len(baseline_data['utt_id'])), key=lambda i: vsm_order.get(baseline_data['utt_id'][i], float('inf')))

            # Reorder all dictionary entries
            baseline_data = {key: [values[i] for i in sorted_indices] for key, values in baseline_data.items()}
            assert baseline_data['utt_id'] == vsm_data['utt_id'], "Utterance mismatch between models!"

        baseline_per_utt_wer, avg_baseline_wer, baseline_sids = per_utterance_wer(baseline_data)

        vsm_per_utt, vsm_avg, vsm_sids = per_utterance_wer(vsm_data)
        # Per Utterance Character Error Rate (CER)
        baseline_per_utt_cer, avg_baseline_cer = per_utterance_cer(baseline_data)
        vsm_per_utt_cer, avg_vsm_cer = per_utterance_cer(vsm_data)

        # Per Utterance Phoneme Error Rate (PER)
        baseline_per_utt_per, avg_baseline_per= per_utterance_per(baseline_data)
        vsm_per_utt_per, avg_vsm_per = per_utterance_per(vsm_data)

        # Per Utterance Viseme Error Rate (VER)
        baseline_per_utt_ver, avg_baseline_ver = per_utterance_ver(baseline_data)
        vsm_per_utt_ver, avg_vsm_ver = per_utterance_ver(vsm_data)

        wer_difference = baseline_per_utt_wer - vsm_per_utt
        cer_difference = baseline_per_utt_cer - vsm_per_utt_cer


        baseline_subs = sum([s[0] for s in baseline_sids])
        baseline_ins = sum([s[1] for s in baseline_sids])
        baseline_dels = sum([s[2] for s in baseline_sids])
        viseme_subs = sum([s[0] for s in vsm_sids])
        viseme_ins = sum([s[1] for s in vsm_sids])
        viseme_dels = sum([s[2] for s in vsm_sids])

        # Update totals
        error_counts[noise_type]['baseline']['subs'] += baseline_subs
        error_counts[noise_type]['baseline']['ins'] += baseline_ins
        error_counts[noise_type]['baseline']['dels'] += baseline_dels

        error_counts[noise_type]['viseme']['subs'] += viseme_subs
        error_counts[noise_type]['viseme']['ins'] += viseme_ins
        error_counts[noise_type]['viseme']['dels'] += viseme_dels

        # Store per SNR
        snr_errors[snr]['subs'].append(viseme_subs - baseline_subs)
        snr_errors[snr]['ins'].append(viseme_ins - baseline_ins)
        snr_errors[snr]['dels'].append(viseme_dels - baseline_dels)

    

        

noise_types = ['Babble', 'Music', 'Speech', 'Noise']
error_types = ['Substitutions', 'Insertions', 'Deletions']

bar_width = 0.35  # width of each individual bar
group_spacing = 0.6  # spacing between each noise type group
# spacing = group_width / len(error_types)  # space between error-type subgroups

fig, ax = plt.subplots(figsize=(15, 11))

x_base = np.arange(len(noise_types)) * (len(error_types) * 2 * bar_width + group_spacing)

# Plot bars: each noise type gets its own cluster of (Subs, Ins, Dels)
err_key_map = {
    'Substitutions': 'subs',
    'Insertions': 'ins',
    'Deletions': 'dels'
}
for j, err_type in enumerate(error_types):
    err_key = err_key_map[err_type]  # map 'Substitutions' -> 'subs', 'Insertions' -> 'ins', 'Deletions' -> 'dels'
    baseline_vals = [error_counts[n.lower()]['baseline'][err_key] for n in noise_types]
    viseme_vals = [error_counts[n.lower()]['viseme'][err_key] for n in noise_types]

    # Calculate center offset per error type (subs, ins, dels)
    offset = j * (2 * bar_width)  # distance between error categories inside one noise type cluster

    # Plot bars for baseline and viseme, side by side
    baseline_bars = ax.bar(x_base + offset, baseline_vals, bar_width, label=f'Baseline {err_type}')
    viseme_bars = ax.bar(x_base + offset + bar_width, viseme_vals, bar_width, label=f'Viseme {err_type}')
    # Add value labels on top of each bar
    for bar in baseline_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(baseline_vals) * 0.01,
                f"{height:.0f}", ha='center', va='bottom', fontsize=14)
    for bar in viseme_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(viseme_vals) * 0.01,
                f"{height:.0f}", ha='center', va='bottom', fontsize=14)

# X-axis labels at the center of each noise type cluster
ax.set_xticks(x_base + bar_width * len(error_types))
title_noises = ['Babble', 'Music', 'Speech', 'Random Noise']
ax.set_xticklabels(title_noises, fontsize=14)

# Labels and title
ax.set_ylabel('Number of Errors', fontsize=14)
ax.set_xlabel('Noise Type', fontsize=14)
ax.set_title(f'Total Errors by Noise Type and Error Type ({model} AV-HuBERT vs VisG AV-HuBERT)', fontsize=14)

# Tick and legend styling
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=14)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_directory + 'total_errors_summed.pdf'), )

# ----------------------------
#  Plot average errors per SNR across noise types
# ----------------------------
avg_per_snr = {snr: {k: np.mean(v) for k, v in snr_errors[snr].items()} for snr in snr_errors}

snr_vals = sorted(avg_per_snr.keys())
subs_vals = [avg_per_snr[snr]['subs'] for snr in snr_vals]
ins_vals = [avg_per_snr[snr]['ins'] for snr in snr_vals]
dels_vals = [avg_per_snr[snr]['dels'] for snr in snr_vals]

plt.figure(figsize=(15, 11))
x = np.arange(len(snr_vals))
width = 0.25

bars_subs = plt.bar(x - width, subs_vals, width, label='Substitutions')
bars_ins = plt.bar(x, ins_vals, width, label='Insertions')
bars_dels = plt.bar(x + width, dels_vals, width, label='Deletions')

# Add value labels on top of each bar
for bars in [bars_subs, bars_ins, bars_dels]:
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            y = height + max(max(subs_vals), max(ins_vals), max(dels_vals)) * 0.01
            va = 'bottom'
        else:
            y = height - max(max(subs_vals), max(ins_vals), max(dels_vals)) * 0.01
            va = 'top'
        plt.text(bar.get_x() + bar.get_width()/2,
                 y,
                 f"{height:.1f}",
                 ha='center', va=va, fontsize=14)

plt.xticks(x, [f"{snr} dB" for snr in snr_vals], fontsize=14)
plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('Average Error Difference (VisG AV-HuBERT - AV-HuBERT)', fontsize=14)
plt.title('Average Error Differences per SNR across Noise Types', fontsize=14)
plt.legend(fontsize=14)
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_directory + 'avg_per_snr.pdf'), )

# -----------------------------
# Compute average errors per noise type across SNRs
# Number of SNRs you iterated over (used to compute average)
num_snrs = 5  # [-10, -5, 0, 5, 10]

# noise order used throughout script
noise_order = ['babble', 'music', 'speech', 'noise']
noise_labels = ['Babble', 'Music', 'Speech', 'Noise']

# Build averaged values per noise type (average across SNRs = total / num_snrs)
avg_error_counts = {}
for noise in noise_order:
    avg_error_counts[noise] = {}
    for model_type in ['baseline', 'viseme']:
        avg_error_counts[noise][model_type] = {
            'subs': error_counts[noise][model_type]['subs'] / num_snrs,
            'ins' : error_counts[noise][model_type]['ins']  / num_snrs,
            'dels': error_counts[noise][model_type]['dels'] / num_snrs
        }




# ---------------------------------------
#  Average Error Differences per SNR for each Noise Type
# ---------------------------------------

noise_types = ['Babble', 'Music', 'Speech', 'Noise']
error_types = ['Substitutions', 'Insertions', 'Deletions']
err_key_map = {'Substitutions': 'subs', 'Insertions': 'ins', 'Deletions': 'dels'}

# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.flatten()

# Iterate over noise types
for i, noise in enumerate(noise_types):
    ax = axes[i]

    # Find the index of this noise type in the order they were added to snr_errors
    noise_idx = i  # Since noise_types order matches the loop order
    
    # Extract values for THIS specific noise type across all SNRs
    snr_vals = sorted(snr_errors.keys())
    subs_vals = [snr_errors[snr]['subs'][noise_idx] for snr in snr_vals]
    ins_vals  = [snr_errors[snr]['ins'][noise_idx]  for snr in snr_vals]
    dels_vals = [snr_errors[snr]['dels'][noise_idx] for snr in snr_vals]
    
    x = np.arange(len(snr_vals))
    width = 0.25
    
    bars_subs = ax.bar(x - width, subs_vals, width, label='Substitutions')
    bars_ins  = ax.bar(x, ins_vals, width, label='Insertions')
    bars_dels = ax.bar(x + width, dels_vals, width, label='Deletions')

    # Label positions above/below depending on bar sign
    max_val = max(max(subs_vals), max(ins_vals), max(dels_vals), key=abs)
    offset = abs(max_val) * 0.02
    for bars in [bars_subs, bars_ins, bars_dels]:
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                y = height + offset
                va = 'bottom'
            else:
                y = height - offset
                va = 'top'
            ax.text(bar.get_x() + bar.get_width()/2, y, f"{height:.1f}",
                    ha='center', va=va, fontsize=10)

    # Axis labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels([f"{snr} dB" for snr in snr_vals], fontsize=12)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Error Difference (VisG AV-HuBERT - AV-HuBERT)', fontsize=11)
    title = f"Random {noise}" if noise == 'Noise' else f"{noise}"
    ax.set_title(title, fontsize=14)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=14)
    ax.tick_params(axis='y', labelsize=12)

plt.suptitle("Error Differences per SNR for Each Noise Type", fontsize=16, y=1.02)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_directory + 'diff_per_snr_for_eachnoise.pdf'), )

# -----------------------------
# Print summary
# -----------------------------


print("\nOverall Error Totals:")
for noise in noise_types:
    print(f"{noise}:")  # optional: keep nice capitalization
    for model_type in ['baseline', 'viseme']:
        print(f"  {model_type.capitalize()} - Subs: {error_counts[noise.lower()][model_type]['subs']}, "
              f"Ins: {error_counts[noise.lower()][model_type]['ins']}, "
              f"Dels: {error_counts[noise.lower()][model_type]['dels']}")

print("\nAverage Error Differences per SNR (Viseme - Baseline):")
print("-------------------------------------------------------")
print(f"{'SNR (dB)':>8} | {'Subs':>10} | {'Ins':>10} | {'Dels':>10}")
print("-" * 46)
for snr in snr_vals:
    subs = avg_per_snr[snr]['subs']
    ins = avg_per_snr[snr]['ins']
    dels = avg_per_snr[snr]['dels']
    print(f"{snr:>8} | {subs:>10.2f} | {ins:>10.2f} | {dels:>10.2f}")
print("-" * 46)




