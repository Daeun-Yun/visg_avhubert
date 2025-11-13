# Statistical analyses
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
import argparse

#0. Load data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# model = 'large'
# exps = ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6', 'Exp7'] if model=='base' else ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6']

def run_analysis(model, exps):
    # Load or train language model for perplexity calculations
    if os.path.exists('/home/aristosp/plots/433h_char_lm_5gram.pkl'):
        print(f"Loading existing word-level LM...")
        with open('/home/aristosp/plots/433h_char_lm_5gram.pkl', "rb") as f:
            lm = pickle.load(f)
    else:
        train_wrd_file = '/home/aristosp/datasets/LRS3/30h_data/train.wrd'
        valid_wrd_file = '/home/aristosp/datasets/LRS3/30h_data/valid.wrd'
        train_texts = []
        for wrd_file in [train_wrd_file, valid_wrd_file]:
            with open(wrd_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        train_texts.append(line)


        lm = build_character_ngram_model(train_texts, 5, smoothing=None)

    # Load sentence transformer model for semantic similarity
    roberta_model = SentenceTransformer('/home/aristosp/models/all-roberta-large-v1')

    for exp in tqdm(exps):
        conditions= []
        for noise in ['babble', 'speech', 'music', 'noise']:
            for db in [-10, -5, 0, 5, 10]:
                conditions.append({'noise_type': noise, 'snr': db})
        
        conditions.append({'noise_type': None, 'snr': None})

        for cond in conditions:
            noise_type = cond['noise_type']
            snr = cond['snr']
            noise_path = f'{noise_type}_snr{snr}' if noise_type else None
            # Usage: Specify which experiment you're analyzing
            # Option 1: By row index (0 for first row, 1 for second, etc.)
            csv_path = os.getcwd() + f'/avhubert/analysis/selected_{model}.csv'
            # experiment_row_index = 4  # Change this to match your experiment

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


            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #1. Statistical analysis
            df = pd.read_csv(csv_path)
            # Separate baseline and experiments
            baseline_row = df.iloc[-1]  # last row = baseline
            baseline_row = baseline_row.iloc[7:]

            # Remove unnecessary columns
            mask = baseline_row.index.str.contains('Babble Musan|AVG')
            baseline_row = baseline_row[~mask]
            # Per utterance Word Error Rate (WER)
            baseline_per_utt_wer, avg_baseline_wer, baseline_sids = per_utterance_wer(baseline_data)

            vsm_per_utt, vsm_avg, vsm_sids = per_utterance_wer(vsm_data)
            # Sanity check
            if noise_type is not None:
                vsm_col = f"{noise_type.capitalize()} {snr}dB" 
                vsm_baseline = df.loc[df['Name'] == experiment_params['Name'], vsm_col].values[0]

                assert abs(vsm_avg - vsm_baseline) < 0.02, \
                    f"Viseme WER mismatch! {vsm_avg:.4f} vs {vsm_baseline:.4f}"

                baseline_col = f"{noise_type.capitalize()} {snr}dB"
                baseline_value = baseline_row[baseline_col]

                assert abs(avg_baseline_wer - baseline_value) < 0.02, \
                    f"Baseline WER mismatch! {avg_baseline_wer:.4f} vs {baseline_value:.4f}"
            # Per Utterance Character Error Rate (CER)
            baseline_per_utt_cer, avg_baseline_cer = per_utterance_cer(baseline_data)
            vsm_per_utt_cer, avg_vsm_cer = per_utterance_cer(vsm_data)

            # Per Utterance Phoneme Error Rate (PER)
            baseline_per_utt_per, avg_baseline_per= per_utterance_per(baseline_data)
            vsm_per_utt_per, avg_vsm_per = per_utterance_per(vsm_data)

            # Per Utterance Viseme Error Rate (VER)
            baseline_per_utt_ver, avg_baseline_ver = per_utterance_ver(baseline_data)
            vsm_per_utt_ver, avg_vsm_ver = per_utterance_ver(vsm_data)

            # Viseme Confusion Matrices
            baseline_cm, viseme_labels = compute_viseme_confusion_matrix(baseline_data)
            vsm_cm, _ = compute_viseme_confusion_matrix(vsm_data)
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            sns.heatmap(baseline_cm, annot=True, fmt='d', xticklabels=viseme_labels, 
                        yticklabels=viseme_labels, ax=axes[0], cmap='Blues')
            axes[0].set_title('AV-HuBERT Viseme Confusion Matrix')
            axes[0].set_xlabel('Predicted Viseme')
            axes[0].set_ylabel('True Viseme')

            sns.heatmap(vsm_cm, annot=True, fmt='d', xticklabels=viseme_labels, 
                        yticklabels=viseme_labels, ax=axes[1], cmap='Oranges')
            axes[1].set_title('VisG AV-HuBERT Viseme Confusion Matrix')
            axes[1].set_xlabel('Predicted Viseme')
            axes[1].set_ylabel('True Viseme')
            plt.savefig(os.path.join(save_directory, f'viseme_confusion_matrices_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'viseme_confusion_matrices.pdf'))
            plt.close()
            # Null hypothesis: Viseme improved the WER
            # Alternate: it did not
            t_value, pvalue = stats.ttest_rel(baseline_per_utt_wer, vsm_per_utt)
            print(f"T-Value = {t_value}, P-Value = {pvalue}")
            # Wilcoxon signed-rank test
            stat, p_val = wilcoxon(baseline_per_utt_wer, vsm_per_utt)
            print(f"Wilcoxon test: W={stat:.3f}, p={p_val:.5f}")
            # DEFINE ΔWER = Viseme Model - baseline Model
            # If ΔWER > 0 -> Viseme model worse
            # If ΔWER < 0 -> Viseme model better
            diff = vsm_per_utt - baseline_per_utt_wer 
            # Effect Size (Cohen's d)
            cohen_d = diff.mean() / diff.std(ddof=1)
            print(f"Cohen's d = {cohen_d:.3f}") # negative is improvement
            # Spearman correlation
            model_corr, model_p = stats.spearmanr(vsm_per_utt, baseline_per_utt_wer)
            print(f"Spearman correlation between VisG AV-HuBERT and AV-HuBERT: r={model_corr:.3f}, p={model_p:.4f}")
            # 95% Confidence Intervals for mean difference
            mean_diff = diff.mean()
            sem = stats.sem(diff)
            ci = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=sem)
            print(f"95% CI for mean difference: {ci}")

            # Mean Absolute Deviation
            mad_baseline = np.mean(np.abs(baseline_per_utt_wer - np.mean(baseline_per_utt_wer)))
            mad_vsm = np.mean(np.abs(vsm_per_utt - np.mean(vsm_per_utt)))
            print(f"MAD {model} AV-HuBERT: {mad_baseline:.4f}")
            print(f"MAD VisG AV-HuBERT: {mad_vsm:.4f}")
            # Root Mean Square Error between models
            rmse = np.sqrt(np.mean((vsm_per_utt - baseline_per_utt_wer) ** 2))
            print(f"RMSE between models (VisG AV-HuBERT - {model} AV-HuBERT): {rmse:.4f}")

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # 2. WER - db Plot, SIDs comparison
            if noise_type is None:
                create_noise_baseline_plots(csv_path, experiment_params["Name"], model=model, output_base_dir=save_directory)
            sid_savepath = os.path.join(save_directory, f'sids_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'sids.pdf')
            title = f"S/I/D Errors Comparison under {noise_type} at {snr} dB" if noise_type is not None else "S/I/D Errors Comparison in Clean Conditions"
            plot_sid_comparison(baseline_sids, vsm_sids, title, savepath=sid_savepath,labels=(f"{model} AV-HuBERT", "VisG AV-HuBERT"))
            plt.close()

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # 3. Plot the number samples that have gotten better or worse from the viseme model

            # Remove zeros, i.e. only focus on changed WERs
            nonzero_idx = diff != 0
            diff_nonzero = diff[nonzero_idx]

            # Get indices for plotting
            indices = np.arange(len(diff))[nonzero_idx]

            # Separate positive and negative differences
            pos_idx = diff_nonzero > 0 # baseline better
            neg_idx = diff_nonzero < 0 # viseme better
            print(f"Baseline WER {avg_baseline_wer} ---------------- Viseme-based WER {vsm_avg}\n")
            print(f"Viseme model degraded in {len(indices[pos_idx])} samples and improved in {len(indices[neg_idx])} samples, compared to the baseline.")

            # Bar Plot to highlight improvement in samples of baseline and viseme model
            plt.figure(figsize=(15, 11))
            bar1 = plt.bar(f'VisG AV-HuBERT worse than {model} AV-HuBERT', len(indices[pos_idx]), label='ΔWER > 0%')
            bar2 = plt.bar(f'VisG AV-HuBERT better than {model} AV-HuBERT', len(indices[neg_idx]), label='ΔWER < 0%')
            plt.text(bar1[0].get_x() + bar1[0].get_width()/2, bar1[0].get_height(), str(len(indices[pos_idx])), 
                    ha='center', va='bottom', fontsize=14, color='black')
            plt.text(bar2[0].get_x() + bar2[0].get_width()/2, bar2[0].get_height(), str(len(indices[neg_idx])), 
                    ha='center', va='bottom', fontsize=14, color='black')
            plt.xlabel('Correct Counts per Model')
            plt.ylabel('Number of samples')
            plt.title(f'Performance Difference between VisG AV-HuBERT and {model} AV-HuBERT in {noise_type} - {snr}dB' if noise_type is not None else f'Performance Difference between VisG AV-HuBERT and {model} AV-HuBERT in Clean Conditions')
            plt.legend()
            bar_plot_savepath = os.path.join(save_directory, f'performance_difference_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'performance_difference.pdf')
            plt.savefig(bar_plot_savepath)
            plt.close()
            
            # Correct overlap
            baseline_correct = set(np.where(baseline_per_utt_wer == 0.0)[0])
            model_correct = set(np.where(vsm_per_utt == 0.0)[0])
            
            # Create Venn diagram
            plt.figure(figsize=(15, 11))
            venn2([baseline_correct, model_correct], set_colors=('blue', 'orange'), set_labels=(f'{model} AV-HuBERT correct', 'VisG AV-HuBERT correct'))
            plt.title(f'Per-utterance correctness overlap in {noise_type} - {snr} dB' if noise_type is not None else f'Per-utterance correctness overlap in clean conditions')
            venn_savepath = os.path.join(save_directory, f'venn_correct_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'venn_correct.pdf')
            plt.savefig(venn_savepath)
            plt.close()
            
            # Wrong overlap
            baseline_wrong = set(np.where(baseline_per_utt_wer != 0.0)[0])
            model_wrong = set(np.where(vsm_per_utt != 0.0)[0])
            plt.figure(figsize=(15, 11))
            venn2([baseline_wrong, model_wrong], set_colors=('blue', 'orange'), set_labels=(f'{model} AV-HuBERT wrong', 'VisG AV-HuBERT wrong'))
            plt.title(f'Per-utterance errors overlap in {noise_type} - {snr} dB' if noise_type is not None else f'Per-utterance errors overlap in clean conditions')
            venn_wrong_savepath = os.path.join(save_directory, f'venn_wrong_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'venn_wrong.pdf')
            plt.savefig(venn_wrong_savepath)
            plt.close()

            # Iterate through to find the utterances that have gotten better or worse
            worse_utts = []
            better_utts = []

            better_utts, worse_utts = find_utts(indices[neg_idx], indices[pos_idx], vsm_data, baseline_data)

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # 4. Plot an example utterance that has gotten better and one that has gotten worse
            examples = {
                "better": better_utts[:5],
                "worse": worse_utts[:5]
            }

            for quality, example_list in examples.items():
                for example in example_list:
                    dataset, speaker, file = example.split('/')
                    example_filename = f"{speaker}_{file}"

                    video_path = f"/home/aristosp/datasets/LRS3/roi/{example}.mp4"
                    alignment_path = f"/home/aristosp/datasets/LRS3/audio/aligned/{example}.csv"

                    idx = vsm_data['utt_id'].index(example)
                    ref = vsm_data['ref'][idx]
                    hypo = vsm_data['hypo'][idx]

                    save_name = (
                        f"{quality}_{example_filename}_{noise_type}_{snr}.pdf"
                        if noise_type is not None
                        else f"{quality}_{example_filename}.pdf"
                    )
                    save_path = os.path.join(save_directory, save_name)

                    show_wrong_segments_with_context(video_path, ref, visg_hypo=hypo, avhubert_hypo=baseline_data['hypo'][idx], savepath=save_path, alignment_path=alignment_path, mode=quality)
                    plt.close()


            # ------------------------------------------------------------------------------------------------------------------------------------
            # 5. Perplexity - WER

            test_refs = baseline_data['ref']
            cross_entropies = []
            for ref in test_refs:
                cross_entropy = compute_cross_entropy(lm, ref, n=5)
                cross_entropies.append(cross_entropy)
            cross_entropies = np.array(cross_entropies)

            baseline_better_idx = diff > 0  # Positive difference = baseline better
            viseme_better_idx = diff < 0   #  Negative difference = viseme better

            # WER Difference vs Cross-entropy Plot
            plt.figure(figsize=(15, 11))
            plt.scatter(cross_entropies[baseline_better_idx], diff[baseline_better_idx], marker='x', label=f'{model} AV-HuBERT Better (n={np.sum(baseline_better_idx)})', s=50)
            plt.scatter(cross_entropies[viseme_better_idx], diff[viseme_better_idx], marker='.', label=f'VisG AV-HuBERT Better (n={np.sum(viseme_better_idx)})', s=50)
            plt.xlabel('Cross-entropy', fontsize=12)
            plt.ylabel('WER Difference (%)', fontsize=12)
            plt.title(f'WER Difference vs Sentence Predictability\n{exp} - {model}, {noise_type} - {snr}dB' if noise_type is not None else f'WER Difference vs Sentence Predictability in Clean Conditions\n{exp} - {model}')
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            # plt.grid(True, alpha=0.3)
            plt.legend()

            # Add labels for better/worse regions
            xlim = plt.xlim()
            ylim = plt.ylim()

            # Arrow pointing upwards (Baseline Better)
            plt.annotate('AV-HuBERT\nBetter', 
                        xy=(xlim[1] * 0.95, ylim[1] * 0.5),  # Arrow tip
                        xytext=(xlim[1] * 0.95, ylim[1] * 0.3),  # Arrow tail (below the tip)
                        fontsize=11,
                        ha='center',
                        va='center',
                        arrowprops=dict(
                            arrowstyle='->', 
                            lw=2.5,
                            color='black'
                        ))

            # Arrow pointing downwards (Viseme Better)
            plt.annotate('VisG AV-HuBERT\nBetter', 
                        xy=(xlim[1] * 0.95, ylim[0] * 0.5),  # Arrow tip
                        xytext=(xlim[1] * 0.95, ylim[0] * 0.3),  # Arrow tail (above the tip)
                        fontsize=11,
                        ha='center',
                        va='center',
                        arrowprops=dict(
                            arrowstyle='->', 
                            lw=2.5,
                            color='black'
                        ))

            perplexity_savepath = os.path.join(save_directory, f'wer_perplexity_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'wer_perplexity.pdf')
            plt.savefig(perplexity_savepath)
            plt.close()

            # CER Difference vs Cross-entropy Plot
            cer_difference = vsm_per_utt_cer - baseline_per_utt_cer
            baseline_better_cer_idx = cer_difference > 0  # Positive difference = baseline better
            viseme_better_cer_idx = cer_difference < 0   #  Negative difference = viseme better
            plt.figure(figsize=(15, 11))
            plt.scatter(cross_entropies[baseline_better_cer_idx], cer_difference[baseline_better_cer_idx], 
                       marker='x', label=f'AV-HuBERT Better (n={np.sum(baseline_better_cer_idx)})', s=50)
            plt.scatter(cross_entropies[viseme_better_cer_idx], cer_difference[viseme_better_cer_idx], 
                       marker='.', label=f'VisG AV-HuBERT Better (n={np.sum(viseme_better_cer_idx)})', s=50)
            plt.xlabel('Cross-entropy', fontsize=12)
            plt.ylabel('CER Difference (%)', fontsize=12)
            # plt.title(f'CER Difference vs Sentence Predictability\n{exp} - {model}, {noise_type} - {snr}dB' if noise_type is not None else f'CER Difference vs Sentence Predictability in Clean Conditions\n{exp} - {model}')
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.legend()

            # Add labels for better/worse regions
            xlim = plt.xlim()
            ylim = plt.ylim()

            # Arrow pointing upwards (Baseline Better)
            plt.text(
                s='AV-HuBERT\nBetter',
                x=xlim[1] * 0.95,
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
                x=xlim[1] * 0.95,
                y=-0.15,         # slightly below y=0
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
            plt.savefig(cer_perplexity_savepath)
            plt.close()
            # plt.show()

            # ------------------------------------------------------------------------------------------------------------------------------------
            # 6. Nonzero WER analysis
            same_nonzero_idx = np.where((baseline_per_utt_wer == vsm_per_utt) & (baseline_per_utt_wer != 0))[0]
            print(f"Number of utterances with same nonzero WER: {len(same_nonzero_idx)}")
            word_overlaps = []
            baseline_nonzero = baseline_per_utt_wer[np.where(baseline_per_utt_wer != 0)[0]]
            viseme_nonzero = vsm_per_utt[np.where(vsm_per_utt != 0)[0]]
            for idx in same_nonzero_idx:
                words_base = set(baseline_data['hypo'][idx].split())
                words_viseme = set(vsm_data['hypo'][idx].split())
                # Calculate Jaccard similarity for textual similarity
                overlap = len(words_base.intersection(words_viseme)) / len(words_base.union(words_viseme))
                word_overlaps.append(overlap)
            plt.figure(figsize=(15, 11))
            sns.histplot(word_overlaps, kde=True)
            plt.title("Word Overlap (Jaccard Similarity) Between AV-HuBERT and VisG AV-HuBERT Predictions (Same Non-Zero WER) {}".format(f"under {noise_type} at {snr} dB" if noise_type is not None else "in Clean Conditions"))
            plt.xlabel("Jaccard Similarity")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(save_directory, f'word_overlap_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'word_overlap.pdf'))
            plt.close()

            plt.figure(figsize=(15, 11))
            plt.hist(baseline_nonzero, bins=20, edgecolor='black', alpha=0.7, label='AV-HuBERT')
            plt.hist(viseme_nonzero, bins=20, edgecolor='black', alpha=0.5, label='VisG AV-HuBERT')

            # Labels and title
            plt.xlabel('WER (%)', fontsize=12)
            plt.ylabel('Number of Utterances', fontsize=12) 
            title = 'Distribution of WER for Utterances with Non-Zero WER: AV-HuBERT vs VisG AV-HuBERT' if noise_type is None else f'Distribution of WER for Utterances with Non-Zero WER: AV-HuBERT vs VisG AV-HuBERT under {noise_type} at {snr} dB'
            plt.title(title, fontsize=14)

            # Add legend to distinguish the two models
            plt.legend(loc='upper right', fontsize=11)

            # Optional: Add grid for better readability
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_directory, f'baseline_vs_viseme_nonzero_wer_distribution_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'baseline_vs_viseme_same_nonzero_wer_distribution.pdf'))
            plt.close()
            
            plt.figure(figsize=(15, 11))
            plt.hist(baseline_nonzero, bins=20, edgecolor='black', alpha=0.7)
            plt.xlabel('WER (%)', fontsize=12)
            plt.ylabel('Number of Utterances', fontsize=12) 
            title = 'Distribution of WER for Utterances with Non-Zero WER for AV-HuBERT' if noise_type is None else f'Distribution of WER for Utterances with Non-Zero WER for {model} AV-HuBERT under {noise_type} at {snr} dB'
            plt.title(title, fontsize=14)
            plt.savefig(os.path.join(save_directory, f'baseline_nonzero_wer_distribution_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'baseline_same_nonzero_wer_distribution.pdf'))
            plt.close()

            plt.figure(figsize=(15, 11))
            plt.hist(viseme_nonzero, bins=20, edgecolor='black', alpha=0.7)
            plt.xlabel('WER (%)', fontsize=12)
            plt.ylabel('Number of Utterances', fontsize=12) 
            title = 'Distribution of WER for Utterances with Non-Zero WER for VisG AV-HuBERT' if noise_type is None else f'Distribution of WER for Utterances with Non-Zero WER for VisG AV-HuBERT under {noise_type} at {snr} dB'
            plt.title(title, fontsize=14)
            plt.savefig(os.path.join(save_directory, f'viseme_nonzero_wer_distribution_{noise_type}_{snr}.pdf') if noise_type is not None else os.path.join(save_directory, 'viseme_same_nonzero_wer_distribution.pdf'))
            plt.close()

            # ------------------------------------------------------------------------------------------------------------------------------------
            # 8. Semantic Similarity analysis (compute for every utterance)
            # Prepare full lists for all utterances
            refs_all = baseline_data['ref']
            baseline_preds_all = baseline_data['hypo']
            viseme_preds_all = vsm_data['hypo']

            # Encode all texts (batched for memory)
            emb_refs = roberta_model.encode(refs_all, convert_to_tensor=True, show_progress_bar=False, batch_size=64)
            emb_baseline_all = roberta_model.encode(baseline_preds_all, convert_to_tensor=True, show_progress_bar=False, batch_size=64)
            emb_viseme_all = roberta_model.encode(viseme_preds_all, convert_to_tensor=True, show_progress_bar=False, batch_size=64)

            # Compute cosine similarities for every utterance
            sim_baseline_ref_all = util.cos_sim(emb_baseline_all, emb_refs).diagonal().cpu().numpy()
            sim_viseme_ref_all = util.cos_sim(emb_viseme_all, emb_refs).diagonal().cpu().numpy()
            sim_baseline_viseme_all = util.cos_sim(emb_baseline_all, emb_viseme_all).diagonal().cpu().numpy()

            # Create a DataFrame for same_nonzero_idx (reuse existing same_nonzero_idx)
            subset = []
            for idx in same_nonzero_idx:
                subset.append({
                    'utt_id': baseline_data['utt_id'][idx],
                    'ref': refs_all[idx],
                    'baseline_pred': baseline_preds_all[idx],
                    'viseme_pred': viseme_preds_all[idx],
                    'sim_baseline_ref': float(sim_baseline_ref_all[idx]),
                    'sim_viseme_ref': float(sim_viseme_ref_all[idx]),
                    'sim_baseline_viseme': float(sim_baseline_viseme_all[idx])
                })

            sub_df = pd.DataFrame(subset)
            # Exclude diagonal points (equal similarity)
            sub_df_filtered = sub_df[sub_df['sim_baseline_ref'] != sub_df['sim_viseme_ref']].copy()

            # Determine which system performs better (by semantic similarity to ref)
            sub_df_filtered['better_system'] = np.where(
                sub_df_filtered['sim_viseme_ref'] > sub_df_filtered['sim_baseline_ref'],
                'VisG AV-HuBERT better',
                'AV-HuBERT better'
            )

            # Count how many are above/below the diagonal
            viseme_better_count = (sub_df_filtered['better_system'] == 'VisG AV-HuBERT better').sum()
            baseline_better_count = (sub_df_filtered['better_system'] == 'AV-HuBERT better').sum()
            total = len(sub_df_filtered)

            # Plot
            plt.figure(figsize=(15, 11))
            sns.scatterplot(
                x='sim_baseline_ref',
                y='sim_viseme_ref',
                hue='better_system',
                data=sub_df_filtered,
                palette={'VisG AV-HuBERT better': 'orange', 'AV-HuBERT better': 'blue'}
            )

            # Diagonal reference line
            plt.plot([0, 1], [0, 1], linestyle='--', color='red')

            # Labels, title, and legend
            plt.xlabel("AV-HuBERT similarity to reference", fontsize=12)
            plt.ylabel("VisG AV-HuBERT similarity to reference", fontsize=12)
            plt.title(
                f"Semantic similarity comparison (Same Non-Zero WER) "
                f"{'under ' + noise_type + ' at ' + str(snr) + ' dB' if noise_type is not None else 'in Clean Conditions'}\n"
                f"VisG AV-HuBERT better: {viseme_better_count} ({viseme_better_count/total:.1%}) | "
                f"AV-HuBERT better: {baseline_better_count} ({baseline_better_count/total:.1%})",
                fontsize=14
            )

            plt.legend(title="Better system")
            plt.grid(True)
            # plt.show()
            plt.savefig(
                os.path.join(save_directory, f'semantic_similarity_{noise_type}_{snr}.pdf')
                if noise_type is not None else
                os.path.join(save_directory, 'semantic_similarity.pdf')
            )
            plt.close()

            # Plot all utterances' similarities
            plt.figure(figsize=(15, 11))
            sns.scatterplot(
                x=sim_baseline_ref_all,
                y=sim_viseme_ref_all,
                alpha=0.5
            )
            # Diagonal reference line
            plt.plot([0, 1], [0, 1], linestyle='--', color='red')
            # Labels, title, and legend
            plt.xlabel("AV-HuBERT similarity to reference", fontsize=12)
            plt.ylabel("VisG AV-HuBERT similarity to reference", fontsize=12)
            plt.title(
                f"Semantic similarity comparison for all utterances "
                f"{'under ' + noise_type + ' at ' + str(snr) + ' dB' if noise_type is not None else 'in Clean Conditions'}",
                fontsize=14
            )
            plt.grid(True)
            plt.savefig(
                os.path.join(save_directory, f'semantic_similarity_all_utterances_{noise_type}_{snr}.pdf')
                if noise_type is not None else f'semantic_similarity_all_utterances.pdf')

            # ------------------------------------------------------------------------------------------------------------------------------------
            # 9. Summary file
            # Identify unchanged utterances

            # mask: keep rows where at least one model is wrong (non-zero WER)
            mask_wrong = (baseline_per_utt_wer != 0) | (vsm_per_utt != 0)
            wrong_indices = np.where(mask_wrong)[0]

            out_csv = os.path.join(save_directory, 'detailed_analysis.csv')
            header = [
                "Utt_ID", "Ref", "Hypo_Baseline", "Hypo_Viseme",
                "WER_Baseline", "WER_Viseme", "WER_Difference",
                "CER_Baseline", "CER_Viseme", "CER_Difference",
                "Cross_Entropy", "Sim_Baseline_Ref", "Sim_Viseme_Ref", "Sim_Baseline_Viseme"
            ]
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in wrong_indices:
                    writer.writerow([
                        utt_ids[i],
                        vsm_data['ref'][i],
                        baseline_data['hypo'][i],
                        vsm_data['hypo'][i],
                        f"{baseline_per_utt_wer[i]:.2f}",
                        f"{vsm_per_utt[i]:.2f}",
                        f"{diff[i]:.2f}",
                        f"{baseline_per_utt_cer[i]:.2f}",
                        f"{vsm_per_utt_cer[i]:.2f}",
                        f"{cer_difference[i]:.2f}",
                        f"{cross_entropies[i]:.3f}",
                        f"{sim_baseline_ref_all[i]:.4f}",
                        f"{sim_viseme_ref_all[i]:.4f}",
                        f"{sim_baseline_viseme_all[i]:.4f}"
                    ])
            unchanged_idx = diff == 0
            unchanged_nonzero_idx = (diff == 0) & (baseline_per_utt_wer != 0)
            unchanged_zero_idx = (diff == 0) & (baseline_per_utt_wer == 0)

            num_unchanged_nonzero = np.sum(unchanged_nonzero_idx)
            num_unchanged_zero = np.sum(unchanged_zero_idx)

            summary_data = {
                "Metric": [
                    "Num Utterances",
                    "Avg WER - Baseline (%)",
                    "Avg WER - Viseme (%)",
                    "Avg CER - Baseline (%)",
                    "Avg CER - Viseme (%)",
                    "Avg PER - Baseline (%)",
                    "Avg PER - Viseme (%)",
                    "Avg VER - Baseline (%)",
                    "Avg VER - Viseme (%)",
                    "Paired t-test (t-value)",
                    "Paired t-test (p-value)",
                    "Wilcoxon W",
                    "Wilcoxon p-value",
                    "Cohen's d (Viseme - Baseline)",
                    "Mean Difference (Viseme - Baseline)",
                    "95% CI Mean Diff",
                    "Improved Utterances",
                    "Worsened Utterances",
                    "Number of Utterances with Same WER (Non 0%)",
                    "Number of Utterances with Same WER (0%)",
                    "Mean Absolute Deviation for Baseline",
                    "Mean Absolute Deviation for Viseme model",
                    "RMSE (Viseme - Baseline)",
                    # "Spearman r (Length vs ΔWER)",
                    # "Spearman p-value"
                ],
                "Value": [
                    len(baseline_per_utt_wer),
                    f"{avg_baseline_wer:.3f}",
                    f"{vsm_avg:.3f}",
                    f"{avg_baseline_cer:.3f}",
                    f"{avg_vsm_cer:.3f}",
                    f"{avg_baseline_per:.3f}",
                    f"{avg_vsm_per:.3f}",
                    f"{avg_baseline_ver:.3f}",
                    f"{avg_vsm_ver:.3f}",
                    f"{t_value:.3f}",
                    f"{pvalue:.5f}",
                    f"{stat:.3f}",
                    f"{p_val:.5f}",
                    f"{cohen_d:.3f}",
                    f"{mean_diff:.3f}",
                    f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                    len(indices[neg_idx]),
                    len(indices[pos_idx]),
                    num_unchanged_nonzero,
                    num_unchanged_zero,
                    f"{mad_baseline:.4f}",
                    f"{mad_vsm:.4f}",
                    f"{rmse:.4f}"
                    # f"{corr:.3f}",
                    # f"{p:.4f}"
                ],
                "Comments": [
                    "Total number of utterances analyzed",
                    "Average WER of the baseline model",
                    "Average WER of the Viseme-based model; slightly lower indicates improvement",
                    "Average CER of the baseline model",
                    "Average CER of the Viseme-based model; slightly lower indicates improvement",
                    "Average PER of the baseline model",
                    "Average PER of the Viseme-based model; slightly lower indicates improvement",
                    "Average VER of the baseline model",
                    "Average VER of the Viseme-based model; slightly lower indicates improvement",
                    "t-statistic for paired differences (Viseme - Baseline)",
                    "Two-sided p-value for paired t-test; <0.05 indicates statistically significant improvement",
                    "Wilcoxon signed-rank statistic",
                    "Two-sided p-value for Wilcoxon test; <0.05 indicates significant paired differences",
                    "Cohen's d effect size; negative = Viseme improved, small magnitude indicates minor effect",
                    "Mean difference in WER (Viseme - Baseline); negative = improvement",
                    "95% confidence interval for the mean difference; does not include 0 → improvement is statistically nonzero",
                    "Number of utterances where Viseme model performed better than baseline",
                    "Number of utterances where Viseme model performed worse than baseline",
                    "Number of utterances unchanged but with non-zero WER",
                    "Number of utterances unchanged with zero WER",
                    "When comparing the two MADs, the one with lower means that",
                    "WER of that model are more consistent, vary less around their mean",
                    "Higher RMSE than MAD means that they disagree on which samples are easy or hard"
                    # "Spearman correlation between utterance length and WER improvement; near 0 = no correlation",
                    # "p-value for Spearman correlation; >0.05 indicates no significant correlation"
                ]
            }

            summary_df = pd.DataFrame(summary_data)

            print("\n=================== COMPACT STATISTICAL SUMMARY ===================\n")
            print(summary_df.to_string(index=False))
            print("\n===================================================================\n")

            csv_file = f"statistical_summary_{noise_type}_{snr}.csv" if noise_type is not None else "statistical_summary.csv"
            save_path = os.path.join(save_directory, csv_file)

            if os.path.exists(save_path):
                summary_df.to_csv(save_path, mode='a', header=False, index=False)
                print(f"Summary appended to existing file: {save_path}")
            else:
                summary_df.to_csv(save_path, index=False)
                print(f"Summary saved to new file: {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AvHubert models with viseme integration.")
    parser.add_argument('--model', type=str, choices=['base', 'large'], default='large', help='Model type to analyze (base or large)')
    parser.add_argument('--exp', type=str, help='Experiment name to analyze (e.g., Exp1, Exp2, etc.)')
    args = parser.parse_args()
    model = args.model
    exp = args.exp
    run_analysis(model, [exp])