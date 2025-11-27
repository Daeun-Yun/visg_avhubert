import json
import matplotlib.pyplot as plt
from plot_utils import per_utterance_wer

def load_data(model, noise_type, snr, experiment_name):
    """Load baseline and viseme model JSON data for a given model."""
    noise_path = f'{noise_type}_snr{snr}' if noise_type else None

    if noise_type is not None:
        baseline_file_path = f'/home/aristosp/models/av_hubert/{model}_decode/{noise_path}/hypo-244018.json'
        vsm_file_path = f'/home/aristosp/viseme_{model}_models/{experiment_name}/decode/{noise_path}/hypo-244018.json'
    else:
        baseline_file_path = f'/home/aristosp/models/av_hubert/{model}_decode/hypo-244018.json'
        vsm_file_path = f'/home/aristosp/viseme_{model}_models/{experiment_name}/decode/hypo-244018.json'

    with open(baseline_file_path, 'r') as f:
        baseline_data = json.load(f)
    with open(vsm_file_path, 'r') as f:
        vsm_data = json.load(f)

    # Compute per-utterance WER
    _, _, baseline_sids = per_utterance_wer(baseline_data)
    _, _, vsm_sids = per_utterance_wer(vsm_data)

    # Sum totals per S/I/D category
    totals_baseline = [sum(col) for col in zip(*baseline_sids)]
    totals_vsm = [sum(col) for col in zip(*vsm_sids)]

    return totals_baseline, totals_vsm

# Experiment settings
noise_type = None
snr = -10
base_experiment_params = {'Name': 'Exp7'}
large_experiment_params = {'Name': 'Exp5'}
categories = ['Substitutions', 'Insertions', 'Deletions']
labels = ["AV-HuBERT", "VisG AV-HuBERT"]
width = 0.35

# Load data for both models
totals_base = load_data('base', noise_type, snr, base_experiment_params['Name'])
totals_large = load_data('large', noise_type, snr, large_experiment_params['Name'])
fontsize = 26
# Setup figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(10,5), sharey=True)

model_titles = ['Base Configuration', 'Large Configuration']

for ax, totals, title in zip(axes, [totals_base, totals_large], model_titles):
    totals_baseline, totals_vsm = totals
    x = range(len(categories))

    rects1 = ax.bar([i - width/2 for i in x], totals_baseline, width, label=labels[0])
    rects2 = ax.bar([i + width/2 for i in x], totals_vsm, width, label=labels[1])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel("Total errors", fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=fontsize)

    # Annotate bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=fontsize)

fig.suptitle(
    f"S/I/D Errors Comparison {'under '+noise_type+' at '+str(snr)+' dB' if noise_type else 'in Clean Conditions'}", 
    fontsize=fontsize
)
plt.tight_layout()
plt.show()