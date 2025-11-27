import matplotlib.pyplot as plt
from plot_utils import per_utterance_cer, get_experiment_directory, compute_cross_entropy
import os, json, pickle
import numpy as np

model = 'large'
exp = 'Exp5'
noise_type = 'speech'
snr = -10
noise_path = f'{noise_type}_snr{snr}' if noise_type else None
csv_path = os.getcwd() + f'/avhubert/analysis/selected_{model}.csv'
# experiment_row_index = 4  # Change this to match your experiment
if os.path.exists('/home/aristosp/plots/433h_char_lm_5gram.pkl'):
        print(f"Loading existing word-level LM...")
        with open('/home/aristosp/plots/433h_char_lm_5gram.pkl', "rb") as f:
            lm = pickle.load(f)
experiment_params = {
    'Name': exp
}
base_save_directory = get_experiment_directory(csv_path, experiment_params, model)
if noise_type is not None:
    save_directory = os.path.join(base_save_directory, noise_type, f'snr_{snr}dB')
else:
    save_directory = os.path.join(base_save_directory, 'clean')
os.makedirs(save_directory, exist_ok=True)

print(f"Saving outputs to: {save_directory}")

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

baseline_per_utt_cer, avg_baseline_cer = per_utterance_cer(baseline_data)
vsm_per_utt_cer, avg_vsm_cer = per_utterance_cer(vsm_data)

# CER Difference vs Cross-entropy Plot
cer_difference = vsm_per_utt_cer - baseline_per_utt_cer
baseline_better_cer_idx = cer_difference > 0  # Positive difference = baseline better
viseme_better_cer_idx = cer_difference < 0   #  Negative difference = viseme better
test_refs = baseline_data['ref']
cross_entropies = []
for ref in test_refs:
    cross_entropy = compute_cross_entropy(lm, ref, n=5)
    cross_entropies.append(cross_entropy)
cross_entropies = np.array(cross_entropies)
plt.figure(figsize=(12, 9))
plt.scatter(cross_entropies[baseline_better_cer_idx], cer_difference[baseline_better_cer_idx], 
            marker='x', label=f'AV-HuBERT Better (n={np.sum(baseline_better_cer_idx)})', s=60)
plt.scatter(cross_entropies[viseme_better_cer_idx], cer_difference[viseme_better_cer_idx], 
            marker='.', label=f'VisG AV-HuBERT Better (n={np.sum(viseme_better_cer_idx)})', s=60)
plt.xlabel('Cross-entropy', fontsize=20)
plt.ylabel('CER Difference (%)', fontsize=20)
# plt.title(f'CER Difference vs Sentence Predictability\n{exp} - {model}, {noise_type} - {snr}dB' if noise_type is not None else f'CER Difference vs Sentence Predictability in Clean Conditions\n{exp} - {model}')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=20)

# Add labels for better/worse regions
xlim = plt.xlim()
ylim = plt.ylim()

# Arrow pointing upwards (Baseline Better)
plt.text(
    s='AV-HuBERT\nBetter',
    x=xlim[1] * 0.94,
    fontsize=16,
    y=0.15,          # slightly above y=0
    ha='center',
    va='bottom'     # align bottom of text at this y
)
# plt.annotate('AV-HuBERT\nBetter', 
#             xy=(xlim[1] * 0.95, ylim[1] * 0.5),  # Arrow tip
#             xytext=(xlim[1] * 0.95, ylim[1] * 0.3),  # Arrow tail (below the tip)
#             fontsize=11,
#             ha='center',
#             va='center',
#             arrowprops=dict(
#                 arrowstyle='-|>', 
#                 lw=2.5,
#                 color='black'
#             ))

# Arrow pointing downwards (Viseme Better)
plt.text(
    s='VisG AV-HuBERT\nBetter',
    x=xlim[1] * 0.94,
    fontsize=16,
    y=-1.15,         # slightly below y=0
    ha='center',
    va='top'        # align top of text at this y
)
# plt.annotate('VisG AV-HuBERT\nBetter', 
#             xy=(xlim[1] * 0.95, ylim[0] * 0.5),  # Arrow tip
#             xytext=(xlim[1] * 0.95, ylim[0] * 0.3),  # Arrow tail (above the tip)
#             fontsize=11,
#             ha='center',
#             va='center',
#             arrowprops=dict(
#                 arrowstyle='-|>', 
#                 lw=2.5,
#                 color='black'
#             ))
cer_perplexity_savepath = os.path.join(save_directory, f'cer_perplexity_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'cer_perplexity.pdf')
plt.tight_layout()
plt.savefig(cer_perplexity_savepath)
# plt.show()
plt.close()