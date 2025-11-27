from tqdm import tqdm
import pandas as pd
import json, os
from plot_utils import get_experiment_directory, per_utterance_cer

model = 'base'

# Experiments
exps = ["Exp7"]

# Prepare result dictionary to accumulate CER
results = {
    'babble': {},
    'speech': {},
    'music': {},
    'noise': {}
}

snr_levels = [-10, -5, 0, 5, 10]

# Initialize structure:
for noise in results:
    results[noise]["AV-HuBERT Base"] = {s: None for s in snr_levels}
    results[noise]["VisG AV-HuBERT Base"] = {s: None for s in snr_levels}
    results[noise]["AV-HuBERT Large"] = {s: None for s in snr_levels}
    results[noise]["VisG AV-HuBERT Large"] = {s: None for s in snr_levels}

csv_path = os.getcwd() + f'/avhubert/analysis/lrs2_results.csv'

for exp in tqdm(exps):
    conditions = []
    for noise in ['babble', 'speech', 'music', 'noise']:
        for db in snr_levels:
            conditions.append({'noise_type': noise, 'snr': db})
    conditions.append({'noise_type': None, 'snr': None})  # clean

    for cond in conditions:
        noise_type = cond['noise_type']
        snr = cond['snr']
        noise_path = f'{noise_type}_snr{snr}' if noise_type else None
        # Usage: Specify which experiment you're analyzing
        # Option 1: By row index (0 for first row, 1 for second, etc.)
        csv_path = os.getcwd() + f'/avhubert/analysis/lrs2_results.csv'
        # experiment_row_index = 4  # Change this to match your experiment

        experiment_params = {
            'Name': exp
        }
        # base_save_directory, selected_row_index = get_experiment_directory(csv_path, experiment_params, model)
        # if noise_type is not None:
        #     save_directory = os.path.join(base_save_directory, noise_type, f'snr_{snr}dB')
        # else:
        #     save_directory = os.path.join(base_save_directory, 'clean')
        # os.makedirs(save_directory, exist_ok=True)

        # print(f"Saving outputs to: {save_directory}")

        if noise_type is not None:
            with open(f'/home/aristosp/models/av_hubert/{model}_decode/LRS2_decode/{noise_path}/hypo-244018.json', 'r') as baseline_file:
                baseline_data = json.load(baseline_file)
            with open(f'/home/aristosp/viseme_{model}_models/{experiment_params["Name"]}/LRS2_decode/{noise_path}/hypo-244018.json', 'r') as f2:
                vsm_data = json.load(f2)  
        else:
            with open(f'/home/aristosp/models/av_hubert/{model}_decode/LRS2_decode/hypo-244018.json', 'r') as baseline_file:
                baseline_data = json.load(baseline_file)
            # Read viseme model JSON
            with open(f'/home/aristosp/viseme_{model}_models/{experiment_params["Name"]}/LRS2_decode/hypo-244018.json', 'r') as f2:
                vsm_data = json.load(f2)

        # Compute CER
        _, base_avg_cer = per_utterance_cer(baseline_data)
        _, vsm_avg_cer = per_utterance_cer(vsm_data)

        # Store only if noisy (clean is ignored in table)
        if noise_type:
            results[noise_type]["AV-HuBERT Base"][snr] = round(base_avg_cer, 2)
            results[noise_type]["VisG AV-HuBERT Base"][snr] = round(vsm_avg_cer, 2)

# --------------------------------------------------------
#           PRINT FINAL TABLE EXACTLY AS REQUESTED
# --------------------------------------------------------

print("\n\n====================== FINAL CER TABLE ======================\n")
print("Model\tNoise Type\t-10 dB\t-5 dB\t0 dB\t5 dB\t10 dB")

for noise_type in ["babble", "speech", "music", "noise"]:
    print(f"\nAV-HuBERT Base\n{noise_type.capitalize()}")
    row = results[noise_type]["AV-HuBERT Base"]
    print("\t" + "\t".join(str(row[s]) for s in snr_levels))

    print(f"VisG AV-HuBERT Base")
    row = results[noise_type]["VisG AV-HuBERT Base"]
    print("\t" + "\t".join(str(row[s]) for s in snr_levels))

    # print(f"AV-HuBERT Large")
    # # placeholder values (fill in when your code produces large model results)
    # print("\t" + "\t".join(["--"]*5))

    # print(f"VisG AV-HuBERT Large")
    # print("\t" + "\t".join(["--"]*5))
