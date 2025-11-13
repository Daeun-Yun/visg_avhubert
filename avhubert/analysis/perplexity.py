import nltk
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
import os, glob, json, math
from plot_utils import *


model = 'base'
noise_type = 'babble'
exp = 'Exp7'
snr = 5
dataset = 'LRS3_30h' # or LRS3_433h
# exps = ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6', 'Exp7'] if model=='base' else ['Exp1', 'Exp2', 'Exp3']
    
#0. Load data
noise_type = noise_type
snr = snr
noise_path = f'{noise_type}_snr{snr}' if noise_type else None
# Usage: Specify which experiment you're analyzing
# Option 1: By row index (0 for first row, 1 for second, etc.)
csv_path = os.getcwd() + f'/avhubert/selected_{model}.csv'
# experiment_row_index = 4  # Change this to match your experiment

experiment_params = {
    'Name': exp
}
save_directory = get_experiment_directory(csv_path, experiment_params, model)

# save_directory = get_experiment_directory(csv_path, experiment_row_index)
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


test_ref = vsm_data['ref']

baseline_per_utt_wer, avg_baseline_wer, baseline_sids = per_utterance_wer(baseline_data)
baseline_per_utt_cer, avg_cer, = per_utterance_cer(baseline_data)
vsm_per_utt, vsm_avg, vsm_sids = per_utterance_wer(vsm_data)
vsm_per_utt_cer, vsm_avg_cer = per_utterance_cer(vsm_data)

wer_difference = vsm_per_utt - baseline_per_utt_wer
delta = vsm_per_utt_cer - baseline_per_utt_cer


if dataset == 'LRS3_30h':
    if os.path.exists('/home/aristosp/plots/30h_char_lm_5gram.pkl'):
            print(f"Loading existing word-level LM...")
            with open('/home/aristosp/plots/30h_char_lm_5gram.pkl', "rb") as f:
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
else:
    train_wrd_file = '/home/aristosp/datasets/LRS3/433h_data/train.wrd'
    valid_wrd_file = '/home/aristosp/datasets/LRS3/433h_data/valid.wrd'
    train_texts = []
    for wrd_file in [train_wrd_file, valid_wrd_file]:
        with open(wrd_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    train_texts.append(line)

    lm = build_character_ngram_model(train_texts, 5, smoothing=None)
cross_entropies = []
# perplexities = []
for ref in test_ref:
    cross_entropy = compute_cross_entropy(lm, ref, n=5)
    # ppx = compute_perplexity(cross_entropy)
    # print(f"Ref: {ref}\nCross-Entropy: {cross_entropy:.3f} bits/char, Perplexity: {ppx:.3f}\n")
    cross_entropies.append(cross_entropy)
    # perplexities.append(ppx)
    

# for ref in test_ref:
#     tokens = word_tokenize(ref.lower())
#     # try:
#         # ppl = lm.perplexity(tokens)
#         # cross_entropy = math.log(ppl)  # or use math.log2(ppl) for bits
#     cross_entropy = lm.entropy(ref)
#     cross_entropies.append(cross_entropy)
#     # except:
#     #     # Handle unknown words - assign high perplexity
#     #     cross_entropies.append(float('inf'))
# perplexities = np.array(perplexities)
cross_entropies = np.array(cross_entropies)
print("Max Cross-Entropy:", np.max(cross_entropies))
print("Min Cross-Entropy:", np.min(cross_entropies))

# Print examples of high and low cross-entropy sentences
sorted_indices = np.argsort(cross_entropies)
for idx in sorted_indices[-10:]:
    print("-------------------------")
    print(f"Cross-Entropy: {cross_entropies[idx]:.3f} Baseline WER: {baseline_per_utt_wer[idx]:.2f}%, Viseme WER: {vsm_per_utt[idx]:.2f}%")
    print(f"Baseline CER: {baseline_per_utt_cer[idx]:.2f}%, Viseme CER: {vsm_per_utt_cer[idx]:.2f}%")
    print(f"Ref: {test_ref[idx]}")
    print(f"Hypo Baseline: {baseline_data['hypo'][idx]}")
    print(f"Hypo Viseme: {vsm_data['hypo'][idx]}")
# # Utterances with CE higher than 4.0 and excluding correctly predicted ones
# idx = np.where((cross_entropies > 4.0) & (wer_difference != 0))[0] 
# print(f"\nNumber of Utterances with Cross-Entropy > 4.0: {len(idx)}")
# # Sort these indices by cross-entropy descending
# idx = idx[np.argsort(-cross_entropies[idx])]

# ce = 4.0  # Example threshold for difficult sentences
# cer_difference = np.array(delta)
# baseline_hard = (cross_entropies > ce) & (cer_difference > 0)
# viseme_hard = (cross_entropies > ce) & (cer_difference < 0)
# hard_num_base = len(np.where(baseline_hard == True)[0])
# hard_num_viseme = len(np.where(viseme_hard == True)[0])
# for i in idx:
#     print("-------------------------"*5)
#     print(f"Cross-Entropy: {cross_entropies[i]:.3f} Baseline WER: {baseline_per_utt_wer[i]:.2f}%, Viseme WER: {vsm_per_utt[i]:.2f}%")
#     print(f"Baseline CER: {baseline_per_utt_cer[i]:.2f}%, Viseme CER: {vsm_per_utt_cer[i]:.2f}%")
#     print(f"Ref: {test_ref[i]}")
#     print(f"Hypo Baseline: {baseline_data['hypo'][i]}")
#     print(f"Hypo Viseme: {vsm_data['hypo'][i]}")

# print(f"Viseme better in {hard_num_viseme} -------- Baseline better in {hard_num_base} for sentences with CE > {ce}")
    


# print("\nTop 5 Most Predictable Sentences (Low Cross-Entropy):")
# for idx in sorted_indices[:5]:
#     print("-------------------------")
#     print(f"Cross-Entropy: {cross_entropies[idx]:.3f} Baseline WER: {baseline_per_utt_wer[idx]:.2f}%, Viseme WER: {vsm_per_utt[idx]:.2f}%")
#     print(f"Ref: {test_ref[idx]}")
#     print(f"Hypo Baseline: {baseline_data['hypo'][idx]}")
#     print(f"Hypo Viseme: {vsm_data['hypo'][idx]}")
# print("\nTop 5 Most Difficult Sentences (High Cross-Entropy):")
# for idx in sorted_indices[-5:]:
#     print("-------------------------")
#     print(f"Cross-Entropy: {cross_entropies[idx]:.3f} Baseline WER: {baseline_per_utt_wer[idx]:.2f}%, Viseme WER: {vsm_per_utt[idx]:.2f}%")
#     print(f"Ref: {test_ref[idx]}")
#     print(f"Hypo Baseline: {baseline_data['hypo'][idx]}")
#     print(f"Hypo Viseme: {vsm_data['hypo'][idx]}")



# Compute Semantic Similarity using Roberta
# from sentence_transformers import SentenceTransformer, util

# roberta_model = SentenceTransformer('/home/aristosp/models/all-roberta-large-v1')
# refs_all = baseline_data['ref']
# baseline_preds_all = baseline_data['hypo']
# viseme_preds_all = vsm_data['hypo']

# # Encode all texts (batched for memory)
# emb_refs = roberta_model.encode(refs_all, convert_to_tensor=True, show_progress_bar=False, batch_size=64)
# emb_baseline_all = roberta_model.encode(baseline_preds_all, convert_to_tensor=True, show_progress_bar=False, batch_size=64)
# emb_viseme_all = roberta_model.encode(viseme_preds_all, convert_to_tensor=True, show_progress_bar=False, batch_size=64)

# # Compute cosine similarities for every utterance
# sim_baseline_ref_all = util.cos_sim(emb_baseline_all, emb_refs).diagonal().cpu().numpy()
# sim_viseme_ref_all = util.cos_sim(emb_viseme_all, emb_refs).diagonal().cpu().numpy()
# sim_baseline_viseme_all = util.cos_sim(emb_baseline_all, emb_viseme_all).diagonal().cpu().numpy()
# same_nonzero_idx = np.where((baseline_per_utt_wer == vsm_per_utt) & (baseline_per_utt_wer != 0))[0]
# # Create a DataFrame for same_nonzero_idx (reuse existing same_nonzero_idx)
# subset = []
# for idx in same_nonzero_idx:
#     subset.append({
#         'utt_id': baseline_data['utt_id'][idx],
#         'ref': refs_all[idx],
#         'baseline_pred': baseline_preds_all[idx],
#         'viseme_pred': viseme_preds_all[idx],
#         'sim_baseline_ref': float(sim_baseline_ref_all[idx]),
#         'sim_viseme_ref': float(sim_viseme_ref_all[idx]),
#         'sim_baseline_viseme': float(sim_baseline_viseme_all[idx])
#     })

# sub_df = pd.DataFrame(subset)
# # Exclude diagonal points (equal similarity)
# sub_df_filtered = sub_df[sub_df['sim_baseline_ref'] != sub_df['sim_viseme_ref']].copy()

# # Determine which system performs better (by semantic similarity to ref)
# sub_df_filtered['better_system'] = np.where(
#     sub_df_filtered['sim_viseme_ref'] > sub_df_filtered['sim_baseline_ref'],
#     'Viseme better',
#     'Baseline better'
# )

# # Count how many are above/below the diagonal
# viseme_better_count = (sub_df_filtered['better_system'] == 'Viseme better').sum()
# baseline_better_count = (sub_df_filtered['better_system'] == 'Baseline better').sum()
# total = len(sub_df_filtered)
# import seaborn as sns
# Plot
# plt.figure(figsize=(15, 11))
# sns.scatterplot(
#     x='sim_baseline_ref',
#     y='sim_viseme_ref',
#     hue='better_system',
#     data=sub_df_filtered,
#     palette={'Viseme better': 'orange', 'Baseline better': 'blue'}
# )

# Diagonal reference line
# plt.plot([0, 1], [0, 1], linestyle='--', color='red')

# # Labels, title, and legend
# plt.xlabel("Baseline similarity to reference", fontsize=12)
# plt.ylabel("Viseme similarity to reference", fontsize=12)
# plt.title(
#     f"Semantic similarity comparison (Same Non-Zero WER) "
#     f"{'under ' + noise_type + ' at ' + str(snr) + ' dB' if noise_type is not None else 'in Clean Conditions'}\n"
#     f"Viseme better: {viseme_better_count} ({viseme_better_count/total:.1%}) | "
#     f"Baseline better: {baseline_better_count} ({baseline_better_count/total:.1%})",
#     fontsize=14
# )

# plt.legend(title="Better system")
# plt.grid(True)

# plt.show()

# # Save everything with example into csv file
# with open(f'{save_directory}/detailed_analysis.csv', 'w') as f:
#     f.write("Utt_ID,Ref,Hypo_Baseline,Hypo_Viseme,WER_Baseline,WER_Viseme,WER_Difference,CER_Baseline,CER_Viseme,CER_Difference,Cross_Entropy,Sim_Baseline_Ref,Sim_Viseme_Ref,Sim_Baseline_Viseme\n")
#     for i in range(len(utt_ids)):
#         f.write(f"{utt_ids[i]},{test_ref[i]},{baseline_data['hypo'][i]},{vsm_data['hypo'][i]},{baseline_per_utt_wer[i]:.2f},{vsm_per_utt[i]:.2f},{wer_difference[i]:.2f},{baseline_per_utt_cer[i]:.2f},{vsm_per_utt_cer[i]:.2f},{delta[i]:.2f},{cross_entropies[i]:.3f},{sim_base_ref[i]:.4f},{sim_viseme_ref[i]:.4f},{sim_base_viseme[i]:.4f}\n")

# baseline_better_idx = wer_difference > 0  # Negative difference = baseline better
# viseme_better_idx = wer_difference < 0   # Positive difference = viseme better
# wer_difference = np.array(wer_difference)
# plt.figure(figsize=(15, 11))
# plt.scatter(cross_entropies[baseline_better_idx], wer_difference[baseline_better_idx], marker='x', label=f'Baseline Better (n={np.sum(baseline_better_idx)})', s=50)
# plt.scatter(cross_entropies[viseme_better_idx], wer_difference[viseme_better_idx], marker='.', label=f'Viseme Better (n={np.sum(viseme_better_idx)})', s=50)
# plt.xlabel('Cross-entropy (Sentence Perplexity)', fontsize=12)
# plt.ylabel('WER Difference (%)', fontsize=12)
# plt.title(f'WER Difference vs Sentence Predictability\n{exp} - {model}')
# plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
# plt.axvline(x=np.mean(cross_entropies), color='gray', linestyle='--', linewidth=1)
# # plt.grid(True, alpha=0.3)
# plt.legend()
# xlim = plt.xlim()
# ylim = plt.ylim()

# # Arrow pointing upwards (Baseline Better)
# plt.annotate('Baseline\nBetter', 
#             xy=(xlim[1] * 0.95, ylim[1] * 0.5),  # Arrow tip
#             xytext=(xlim[1] * 0.95, ylim[1] * 0.3),  # Arrow tail (below the tip)
#             fontsize=11,
#             ha='center',
#             va='center',
#             arrowprops=dict(
#                 arrowstyle='->', 
#                 lw=2.5,
#                 color='black'
#             ))
# plt.annotate('Viseme Model\nBetter', 
#                         xy=(xlim[1] * 0.95, ylim[0] * 0.5),  # Arrow tip
#                         xytext=(xlim[1] * 0.95, ylim[0] * 0.3),  # Arrow tail (above the tip)
#                         fontsize=11,
#                         ha='center',
#                         va='center',
#                         arrowprops=dict(
#                             arrowstyle='->', 
#                             lw=2.5,
#                             color='black'
#                         ))

cer_difference = np.array(delta)
baseline_better_cer_idx = cer_difference > 0
viseme_better_cer_idx = cer_difference < 0

# Difficult limit
# ce = 4.0  # Example threshold for difficult sentences
# baseline_hard = (cross_entropies > ce) & (cer_difference > 0)
# viseme_hard = (cross_entropies > ce) & (cer_difference < 0)
# hard_num_base = len(np.where(baseline_hard == True)[0])
# hard_num_viseme = len(np.where(viseme_hard == True)[0])
plt.figure(figsize=(15, 11))
plt.scatter(cross_entropies[baseline_better_cer_idx], cer_difference[baseline_better_cer_idx], 
           marker='x', label=f'Baseline Better (n={np.sum(baseline_better_cer_idx)})', s=50)
plt.scatter(cross_entropies[viseme_better_cer_idx], cer_difference[viseme_better_cer_idx], 
           marker='.', label=f'Viseme Better (n={np.sum(viseme_better_cer_idx)})', s=50)
plt.xlabel('Cross-entropy (Sentence Predictability)', fontsize=12)
plt.ylabel('CER Difference (%)', fontsize=12)
plt.title(f'CER Difference vs Sentence Predictability\n{exp} - {model}')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
# plt.axvline(x=ce, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
xlim = plt.xlim()
ylim = plt.ylim()

plt.text(x=xlim[1] * 0.97, y=ylim[1]*0.5, s=f'Baseline Better', fontsize=10, ha='right')
plt.text(x=xlim[1] * 0.97, y=ylim[0]*0.5, s=f'Vise-HuBERT Better', fontsize=10, ha='right')
# plt.fill_betweenx(y=[0, ylim[1]], x1=xlim[0], x2=xlim[1], color='blue', alpha=0.3)
# plt.fill_betweenx(y=[0, ylim[0]], x1=0, x2=xlim[1], color='orange', alpha=0.3)
# Arrow pointing upwards (Baseline Better)
# plt.annotate('Baseline\nBetter', 
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

# plt.annotate('Viseme Model\nBetter', 
#                         xy=(xlim[1] * 0.95, ylim[0] * 0.5),  # Arrow tip
#                         xytext=(xlim[1] * 0.95, ylim[0] * 0.3),  # Arrow tail (above the tip)
#                         fontsize=11,
#                         ha='center',
#                         va='center',
#                         arrowprops=dict(
#                             arrowstyle='-|>', 
#                             lw=2.5,
#                             color='black'
#                         ))

plt.show()



# # Add labels for better/worse regions
# plt.text(plt.xlim()[1]*0.85, plt.ylim()[1]*0.9, 'Viseme Better', 
#          fontsize=10, ha='right')
# plt.text(plt.xlim()[1]*0.85, plt.ylim()[0]*0.9, 'Baseline Better', 
#          fontsize=10, ha='right')

# # plt.tight_layout()
# # plt.savefig(f'{save_directory}/wer_diff_vs_perplexity.png', dpi=300)
# plt.show()

# print(f"Mean WER Baseline: {np.mean(avg_baseline_wer):.2f}%")
# print(f"Mean WER Viseme: {np.mean(vsm_avg):.2f}%")
# print(f"Mean WER Difference: {np.mean(wer_difference):.2f}%")
# print(f"Correlation: {np.corrcoef(cross_entropies, wer_difference)[0,1]:.3f}")



# # # --- NEW CHARACTER-LEVEL VERSION ---
# # from nltk.util import everygrams

# n = 5  # n-gram order (you can keep 5)
# train_chars = [list(sent.lower()) for sent in train_texts]  # tokenize to characters
# train_data, vocab = padded_everygram_pipeline(n, train_chars)

# lm2 = MLE(n, vocabulary=vocab)
# lm = train_lm(wrd_file, n)


# print(f"Trained {n}-gram character-level language model on {len(train_chars)} sentences.")

# cross_entropies = []

# for ref in test_ref:
#     tokens = list(ref.lower())  # character tokens
#     try:
#         cross_entropy = lm2.entropy(tokens)
#     except Exception:
#         cross_entropy = float('inf')  # fallback for unseen characters
#     cross_entropies.append(cross_entropy)


# cross_entropies = np.array(cross_entropies)
# baseline_better_idx = wer_difference > 0  # Negative difference = baseline better
# viseme_better_idx = wer_difference < 0   # Positive difference = viseme better
# wer_difference = np.array(wer_difference)
# plt.figure(figsize=(15, 11))
# plt.scatter(cross_entropies[baseline_better_idx], wer_difference[baseline_better_idx], marker='x', label=f'Baseline Better (n={np.sum(baseline_better_idx)})', s=50)
# plt.scatter(cross_entropies[viseme_better_idx], wer_difference[viseme_better_idx], marker='.', label=f'Viseme Better (n={np.sum(viseme_better_idx)})', s=50)
# plt.xlabel('Cross-entropy (Sentence Perplexity)', fontsize=12)
# plt.ylabel('WER Difference (%)', fontsize=12)
# plt.title(f'WER Difference vs Sentence Predictability\n{exp} - {model}')
# plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
# # plt.grid(True, alpha=0.3)
# plt.legend()

# plt.show()
