from plot_utils import create_noise_baseline_plots, get_experiment_directory
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

# baseline_per_utt_cer, avg_baseline_cer = per_utterance_cer(baseline_data)
# vsm_per_utt_cer, avg_vsm_cer = per_utterance_cer(vsm_data)
print(save_directory)

create_noise_baseline_plots(csv_path, experiment_params["Name"], model=model, output_base_dir=save_directory)