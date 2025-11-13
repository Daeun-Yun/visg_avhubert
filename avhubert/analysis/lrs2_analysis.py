import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(os.getcwd(),"avhubert/analysis/", 'lrs2_results.csv'))
snr_levels = [-10, -5, 0, 5, 10]
noise_types = ['Babble', 'Speech', 'Music', 'Noise']

baseline_base = df.iloc[1][7:]
mask = baseline_base.index.str.contains('Babble Musan|AVG')
mymodel_base = df.iloc[0][7:]
mymodel_large = df.iloc[2][7:]
baseline_large = df.iloc[3][7:]

baseline_base = baseline_base[~mask]
baseline_large = baseline_large[~mask]
mymodel_base = mymodel_base[~mask]
mymodel_large = mymodel_large[~mask]


fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
axes = axes.flatten()

for i, noise_type in enumerate(noise_types):
    ax = axes[i]
    
    baseline_values = baseline_base.filter(like=noise_type).values
    experiment_values = mymodel_base.filter(like=noise_type).values
    
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
# plt.show()
plt.savefig('/home/aristosp/plots/selected_base/Exp7_0.3prob-2.5db_2 + 2 dropout0.3 + Layernorm_0.2_nonconvex_20k_4k/base_2x2_wer_comparison_LRS2.pdf')

fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
axes = axes.flatten()

for i, noise_type in enumerate(noise_types):
    ax = axes[i]
    
    baseline_values = baseline_large.filter(like=noise_type).values
    experiment_values = mymodel_large.filter(like=noise_type).values
    
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
plt.savefig('/home/aristosp/plots/selected_large/Exp5_0.3prob-2.5db_3 + 3 dropout0.3 + Layernorm_0.15_nonconvex_0k (freezefinetune at 40k)/large_2x2_wer_comparison_LRS2.pdf')
plt.show()