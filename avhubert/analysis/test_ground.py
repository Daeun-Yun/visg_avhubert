import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy import stats
from plot_utils import *
from sentence_transformers import SentenceTransformer, util
from statsmodels.stats.contingency_tables import mcnemar
import json
from collections import defaultdict
import glob
from g2p_en import G2p
from jiwer import process_words
from matplotlib_venn import venn2
model = 'base'
noise_type = None
exp = 'Exp7'
snr = -10
# exps = ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6', 'Exp7'] if model=='base' else ['Exp1', 'Exp2', 'Exp3']
experiment_params = {
                'Name': exp
            }    
#0. Load data
noise_type = noise_type
snr = snr
noise_path = f'{noise_type}_snr{snr}' if noise_type else None
# Usage: Specify which experiment you're analyzing
# Option 1: By row index (0 for first row, 1 for second, etc.)
csv_path = os.getcwd() + f'/avhubert/selected_{model}.csv'
save_dir = get_experiment_directory(csv_path, experiment_params, model)


# Load CSV
df = pd.read_csv(csv_path)


# model = 'base'
# noise_type = None
# exp = 'Exp7'
# snr = -10
# # exps = ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6', 'Exp7'] if model=='base' else ['Exp1', 'Exp2', 'Exp3']
    
# #0. Load data
# noise_type = noise_type
# snr = snr
# noise_path = f'{noise_type}_snr{snr}' if noise_type else None
# # Usage: Specify which experiment you're analyzing
# # Option 1: By row index (0 for first row, 1 for second, etc.)
# csv_path = os.getcwd() + f'/avhubert/selected_{model}.csv'
# # experiment_row_index = 4  # Change this to match your experiment

# experiment_params = {
#     'Name': exp
# }
# save_directory = get_experiment_directory(csv_path, experiment_params, model)

# save_directory = get_experiment_directory(csv_path, experiment_row_index)
# print(f"Saving outputs to: {save_directory}")

# if noise_type is not None:
#     with open(f'/home/aristosp/models/av_hubert/{model}_decode/{noise_path}/hypo-244018.json', 'r') as baseline_file:
#         baseline_data = json.load(baseline_file)
#     with open(f'/home/aristosp/viseme_{model}_models/{experiment_params["Name"]}/decode/{noise_path}/hypo-244018.json', 'r') as f2:
#         vsm_data = json.load(f2)  
# else:
#     with open(f'/home/aristosp/models/av_hubert/{model}_decode/hypo-244018.json', 'r') as baseline_file:
#         baseline_data = json.load(baseline_file)
#     # Read viseme model JSON
#     with open(f'/home/aristosp/viseme_{model}_models/{experiment_params["Name"]}/decode/hypo-244018.json', 'r') as f2:
#         vsm_data = json.load(f2)


# utt_ids = baseline_data['utt_id'] # same if read from vsm data

# try:
#     assert baseline_data['utt_id'] == vsm_data['utt_id'], "Utterance mismatch between models!"
# except (AssertionError, ValueError):
# # Create a mapping of utt_id positions from vsm_data
#     # Mapping of utt_id -> index in vsm_data
#     vsm_order = {utt_id: idx for idx, utt_id in enumerate(vsm_data['utt_id'])}

#     # Indices to sort baseline_data according to vsm_data
#     sorted_indices = sorted(range(len(baseline_data['utt_id'])), key=lambda i: vsm_order.get(baseline_data['utt_id'][i], float('inf')))

#     # Reorder all dictionary entries
#     baseline_data = {key: [values[i] for i in sorted_indices] for key, values in baseline_data.items()}
#     assert baseline_data['utt_id'] == vsm_data['utt_id'], "Utterance mismatch between models!"

# baseline_per_utt_wer, avg_baseline_wer, baseline_sids = per_utterance_wer(baseline_data)

# vsm_per_utt, vsm_avg, vsm_sids = per_utterance_wer(vsm_data)



# wer_difference = baseline_per_utt_wer - vsm_per_utt
# # same_wers = np.where((baseline_per_utt_wer == vsm_per_utt) & (baseline_per_utt_wer != 0))[0]
# same_nonzero_idx = np.where((baseline_per_utt_wer == vsm_per_utt) & (baseline_per_utt_wer != 0))[0]

# same_nonzero_baseline_wers_values = baseline_per_utt_wer[same_nonzero_idx]


# baseline_per_utt_cer, avg_baseline_cer = per_utterance_cer(baseline_data)
# vsm_per_utt_cer, avg_vsm_cer = per_utterance_cer(vsm_data)

# g2p = G2p()

# baseline_per_utt_cer, avg_baseline_cer = per_utterance_cer(baseline_data)
# vsm_per_utt_cer, avg_vsm_cer = per_utterance_cer(vsm_data)

# baseline_per_utt_per, avg_baseline_per = per_utterance_per(baseline_data)
# vsm_per_utt_per, avg_vsm_per = per_utterance_per(vsm_data)

# baseline_per_utt_ver, avg_baseline_ver = per_utterance_ver(baseline_data)
# vsm_per_utt_ver, avg_vsm_ver = per_utterance_ver(vsm_data)

# Plot WER - db per noise type
df = pd.read_csv(csv_path)

# Separate baseline and experiments
baseline_row = df.iloc[-1]  # last row = baseline
baseline_row = baseline_row.iloc[7:]

# Remove unnecessary columns
mask = baseline_row.index.str.contains('Babble Musan|AVG')
baseline_row = baseline_row[~mask]

# Single experiment processing
experiment = df[df.isin([exp]).any(axis=1)]
experiment = experiment.iloc[:, 7:]
experiment_mask = experiment.columns.str.contains('Babble Musan|AVG')
experiment = experiment.loc[:, ~experiment_mask]

print(f"Creating noise-level baseline WER comparisons for {exp}...\n")


# Combined plot for all noise types
snr_levels = [-10, -5, 0, 5, 10]
noise_types = ['Babble', 'Speech', 'Music', 'Noise']

plt.figure(figsize=(15, 10))

# Large
if model == 'large':
    baseline_noisy = [33.2558,	6.2892,	9.18,	9.6866,
                    14.7624,	3.4884,	4.2973,	4.6,			
                    5.6117,	2.366,	2.5177,	2.5784,			
                    2.6795,	2.0627,	1.8402,	1.8301,			
                    1.9818,	2.0728,	1.6178,	1.6178	]
    baseline_noisy_row_column_names = baseline_row.index.tolist()[1:]
    baseline_noisy_row = pd.DataFrame([baseline_noisy], columns=baseline_noisy_row_column_names)
else:
    baseline_noisy = [41.6987,	13.0536,	15.4096,	15.2376,
                      22.8615,	8.5,	9.4237,	8.9484,
                      10.8089,	6.5521,	6.0566,	6.1982,
                      6.6026,	5.541,	5.0859,	4.8635,
                      5.05,	4.9343,	4.4894,	4.4995]
    baseline_noisy_row_column_names = baseline_row.index.tolist()[1:]
    baseline_noisy_row = pd.DataFrame([baseline_noisy], columns=baseline_noisy_row_column_names)
# Iterate through noise types and plot on the same figure
for noise_type in noise_types:
    baseline_values = baseline_row.filter(like=noise_type).values
    experiment_values = experiment.filter(like=noise_type).iloc[0].values
    
    plt.plot(
        snr_levels,
        baseline_values,
        label=f'{noise_type} - AV-HuBERT',
        marker='o',
        linestyle='--'
    )
    
    plt.plot(
        snr_levels,
        experiment_values,
        label=f'{noise_type} - VisG AV-HuBERT',
        marker='P',
        linestyle='-'
    )

plt.legend()
plt.ylabel('WER (%)')
plt.xlabel('Noise level (dB)')
plt.xticks(ticks=snr_levels, labels=[f'{db}dB' for db in snr_levels])
plt.title('AV-HuBERT vs VisG AV-HuBERT WER Across Noise Types')
plt.grid(True)
plt.show()

snr_levels = [-10, -5, 0, 5, 10]
noise_types = ['Babble', 'Speech', 'Music', 'Noise']

# fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)

# for i, noise_type in enumerate(noise_types):
#     ax = axes[i]
    
#     baseline_values = baseline_row.filter(like=noise_type).values
#     experiment_values = experiment.filter(like=noise_type).iloc[0].values
#     baseline_noisy_values = baseline_noisy_row.filter(like=noise_type).iloc[0].values
    
#     ax.plot(
#         snr_levels,
#         baseline_values,
#         label='AV-HuBERT',
#         marker='o',
#         linestyle='--'
#     )
#     ax.plot(
#         snr_levels,
#         experiment_values,
#         label='VisG AV-HuBERT',
#         marker='P',
#         linestyle='-'
#     )
    
#     # ax.plot(
#     #     snr_levels,
#     #     baseline_noisy_values,
#     #     label='Baseline Noisy',
#     #     marker='x',
#     #     linestyle=':'
#     # )

#     ax.set_ylabel('WER (%)')
#     ax.set_title(f'Random {noise_type}' if noise_type == 'Noise' else f"{noise_type}")
#     ax.grid(True, axis='y')
#     ax.legend()

# # Shared x-axis label
# axes[-1].set_xlabel('Noise level (dB)')
# plt.xticks(ticks=snr_levels, labels=[f'{db}dB' for db in snr_levels])
# plt.savefig(f'{save_dir}/4x1_wer_comparison.svg')
# plt.suptitle('AV-HuBERT vs VisG AV-HuBERT WER per Noise Type', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.97])

fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
axes = axes.flatten()

for i, noise_type in enumerate(noise_types):
    ax = axes[i]
    
    baseline_values = baseline_row.filter(like=noise_type).values
    experiment_values = experiment.filter(like=noise_type).iloc[0].values
    baseline_noisy_values = baseline_noisy_row.filter(like=noise_type).iloc[0].values
    
    ax.plot(
        snr_levels,
        baseline_values,
        label='AV-HuBERT',
        marker='o',
        linestyle='--'
    )
    ax.plot(
        snr_levels,
        experiment_values,
        label='VisG AV-HuBERT',
        marker='P',
        linestyle='-'
    )

    ax.set_ylabel('WER (%)', fontsize=14)
    ax.set_title(f'Random {noise_type}' if noise_type == 'Noise' else f"{noise_type}", fontsize=14)
    ax.grid(True, axis='y')
    ax.legend(fontsize=14)
    ax.set_xticks(snr_levels)
    ax.tick_params(labelsize=14)

# shared x-axis label
fig.supxlabel('Noise level (dB)', fontsize=14)
# show dB labels on bottom row
for ax in axes[-2:]:
    ax.set_xticklabels([f'{db}dB' for db in snr_levels], fontsize=14)

plt.tight_layout()
plt.savefig(f'{save_dir}/2x2_wer_comparison.svg')
plt.show()



        
# for noise_type in ['Babble', 'Speech', 'Music', 'Noise']:
    # experiment_noise = experiment.filter(like=noise_type).iloc[0]
    # baseline_noise = baseline_row.filter(like=noise_type)

    # # Create plot
    # plt.figure(figsize=(15, 9))
    # plt.plot(baseline_noise, label='Baseline', marker='o')
    # plt.plot(experiment_noise, label='Viseme Model', marker='P')
    # plt.legend()
    # plt.ylabel('WER (%)')
    # plt.xlabel('Noise level (dB)')
    # plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['-10dB', '-5dB', '0dB', '5dB', '10dB'])
    # plt.title(f'Baseline WER vs Viseme-Based Experiment in {noise_type}')
