import numpy as np
import matplotlib.pyplot as plt
import os
import editdistance
import cv2
import pronouncing
import pandas as pd
from g2p_en import G2p
import re
from difflib import SequenceMatcher
from jiwer import process_words
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
import math
import pickle


from sklearn.metrics import confusion_matrix


def compute_viseme_confusion_matrix(data):
    """
    Compute viseme confusion matrix from reference and hypothesis text.
    
    Args:
        data: dict with keys 'ref' and 'hypo' (lists of text)
    
    Returns:
        confusion_matrix: numpy array
        viseme_labels: list of unique viseme labels
    """
    g2p = G2p()
    all_ref_visemes = []
    all_hypo_visemes = []
    
    for ref_text, hypo_text in zip(data['ref'], data['hypo']):
        # Convert to phonemes
        ref_phons = [re.sub(r'\d', '', p) for p in g2p(ref_text) if p.isalpha() or p.isupper()]
        hypo_phons = [re.sub(r'\d', '', p) for p in g2p(hypo_text) if p.isalpha() or p.isupper()]
        
        # Convert to visemes
        ref_visemes = [phoneme_to_viseme(p) for p in ref_phons]
        hypo_visemes = [phoneme_to_viseme(p) for p in hypo_phons]
        
        # Align sequences (handle length mismatch)
        min_len = min(len(ref_visemes), len(hypo_visemes))
        all_ref_visemes.extend(ref_visemes[:min_len])
        all_hypo_visemes.extend(hypo_visemes[:min_len])
    
    # Get unique viseme labels
    viseme_labels = sorted(list(set(all_ref_visemes + all_hypo_visemes)))
    
    # Compute confusion matrix
    cm = confusion_matrix(all_ref_visemes, all_hypo_visemes, labels=viseme_labels)
    
    return cm, viseme_labels


def build_character_ngram_model(train_texts, n=5, smoothing='laplace'):
    """
    Build a character-level n-gram language model
    
    Args:
        train_texts: List of training sentences
        n: n-gram order (default: 5 for character-level)
        smoothing: 'laplace' or 'mle'
    
    Returns:
        Trained language model
    """
    # Tokenize at character level
    train_data = []
    for text in train_texts:
        # Convert to lowercase and split into characters
        chars = list(text.lower())
        train_data.append(chars)
    
    # Create padded n-grams
    train_ngrams, padded_vocab = padded_everygram_pipeline(n, train_data)
    
    # Train model with smoothing
    if smoothing == 'laplace':
        lm = Laplace(n)
    else:
        lm = MLE(n)
    print("Training character-level n-gram model...")
    lm.fit(train_ngrams, padded_vocab)
    lm_path = '/home/aristosp/plots/lrs2_char_lm_5gram.pkl'
    with open(lm_path, 'wb') as f:
        pickle.dump(lm, f)

    return lm

def compute_cross_entropy(lm, sentence, n=5):
    """
    Compute cross-entropy of a sentence given the language model
    
    Args:
        lm: Trained language model
        sentence: Test sentence (string)
        n: n-gram order
    
    Returns:
        Cross-entropy value (in bits per character)
    """
    # Convert sentence to characters
    chars = list(sentence.lower())
    
    # Compute log probability for each character given context
    log_probs = []
    
    # Pad the sentence
    padded_chars = ['<s>'] * (n-1) + chars + ['</s>']
    
    for i in range(n-1, len(padded_chars)):
        # Get context (previous n-1 characters)
        context = tuple(padded_chars[i-(n-1):i])
        # Get current character
        char = padded_chars[i]
        
        # Get probability from model
        prob = lm.score(char, context)
        
        if prob > 0:
            log_probs.append(math.log2(prob))
        else:
            # Handle zero probability (shouldn't happen with Laplace smoothing)
            log_probs.append(-20)  # Very low probability
    
    # Cross-entropy is negative average log probability
    if len(log_probs) > 0:
        cross_entropy = -sum(log_probs) / len(log_probs)
    else:
        cross_entropy = float('inf')
    
    return cross_entropy

def create_noise_baseline_plots(csv_path, name, model='base', output_base_dir='/home/aristosp/plots'):
    """
    Create baseline vs viseme model comparison plots across different noise types and levels.
    
    This function:
    1. Loads the CSV with experiment results
    2. Separates baseline (last row) from experiments
    3. Creates a 2x2 subplot figure comparing baseline vs viseme model
       across different noise types (Babble LRS3, Speech, Music, Noise)
    4. Saves plot in organized directory per experiment
    
    Args:
        csv_path: Path to the CSV file containing experiment results
        name: Experiment Name
        model: Model type (e.g., 'base', 'large')
        output_base_dir: Base directory for saving plots
    """
    print("\n" + "="*80)
    print("Creating Noise-Level Baseline Comparison Plots")
    print("="*80 + "\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Separate baseline and experiments
    baseline_row = df.iloc[-1]  # last row = baseline
    baseline_row = baseline_row.iloc[7:]
    
    # Remove unnecessary columns
    mask = baseline_row.index.str.contains('Babble Musan|AVG')
    baseline_row = baseline_row[~mask]
    
    # Single experiment processing
    experiment = df[df.isin([name]).any(axis=1)]
    experiment = experiment.iloc[:, 7:]
    experiment_mask = experiment.columns.str.contains('Babble Musan|AVG')
    experiment = experiment.loc[:, ~experiment_mask]
    
    print(f"Creating noise-level baseline WER comparisons for {name}...\n")
    
    # Define noise types and SNR levels
    noise_types = ['Babble', 'Speech', 'Music', 'Noise']
    snr_levels = [-10, -5, 0, 5, 10]
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    axes = axes.flatten()
    
    # Iterate through noise types
    for i, noise_type in enumerate(noise_types):
        ax = axes[i]
        
        experiment_values = experiment.filter(like=noise_type).iloc[0].values
        baseline_values = baseline_row.filter(like=noise_type).values
        
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
        
        ax.set_ylabel('WER (%)', fontsize=18)
        ax.set_title(f'Random {noise_type}' if noise_type == 'Noise' else f"{noise_type}", fontsize=18)
        ax.grid(True, axis='y')
        ax.legend(fontsize=16)
        ax.set_xticks(snr_levels)
        ax.tick_params(labelsize=18)
    
    # Shared x-axis label
    fig.supxlabel('Noise level (dB)', fontsize=18)
    
    # Show dB labels on bottom row
    for ax in axes[-2:]:
        ax.set_xticklabels([f'{db}dB' for db in snr_levels], fontsize=18)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_base_dir, 'new_2x2_wer_comparison.pdf')
    plt.savefig(save_path)
    # plt.show()
    plt.close()
    
    print("\nFinished creating noise-level baseline comparison plots.")
    print("="*80 + "\n")



def align_word_sequences(ref_words, hypo_words):
    """
    Align reference and hypothesis word sequences, handling insertions/deletions.
    Returns: list of (ref_idx, hypo_idx, ref_word, hypo_word, is_match)
    """
    matcher = SequenceMatcher(None, ref_words, hypo_words)
    aligned = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                aligned.append((i, j, ref_words[i], hypo_words[j], True))
        elif tag == 'replace':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                aligned.append((i, j, ref_words[i], hypo_words[j], False))
        elif tag == 'delete':
            for i in range(i1, i2):
                aligned.append((i, None, ref_words[i], '<missing>', False))
        elif tag == 'insert':
            for j in range(j1, j2):
                aligned.append((None, j, '<missing>', hypo_words[j], False))
    
    return aligned


def get_experiment_directory(csv_path, experiment_identifier, model):
    """
    Get the same directory path that plots.py uses for a specific experiment.
    
    Args:
        csv_path: Path to the experiment CSV file
        experiment_identifier: Can be row index, or a dict with experiment params
    
    Returns:
        Directory path where outputs should be saved
    """
    
    df = pd.read_csv(csv_path)
    
    # If identifier is an integer, use it as row index
    if isinstance(experiment_identifier, int):
        row = df.iloc[experiment_identifier]
    else:
        # If identifier is a dict, find matching row
        # e.g., {'Viseme Weight': 0.15, 'Loss Calculation': 'nonconvex'}
        mask = pd.Series([True] * len(df))
        for key, value in experiment_identifier.items():
            if key not in df.columns:
                continue
            col = df[key]
            # Handle numeric columns (e.g. WER)
            if pd.api.types.is_numeric_dtype(col):
                mask &= (abs(col - value) < 1e-6)
            else:
                # Handle string columns safely (ignore NaN)
                mask &= (col.astype(str).fillna('') == str(value))
        row = df[mask].iloc[0]
    
    # Get first 6 columns (experiment configuration)
    original_row = row.iloc[:7]
    
    # Create directory name same way as plots.py
    dir_name = '_'.join([str(val) for val in original_row.values if not pd.isna(val)])
    save_directory = f'/home/aristosp/plots/selected_{model}/{dir_name}/'
    
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    return save_directory

def compute_bucket_stats(hypos, wers, bucket_ranges=[(0,5),(5,10),(10,15),(15,20),(20,25),(25,np.inf)]):
    """
    Compute wer and number of samples for specific intervals of utterance length.
    Args:
        hypo: a list containing the hypotheses the model made
        wers: a list containing the wer per utterance
    Returns:
        counts: a list with the number of utterances per bucket
        wers_sum: a list with the summed WER for utterances per bucket
        word_lengths: a list with the length of each utterance
    """
    counts = [0] * len(bucket_ranges)
    wers_sum = [0] * len(bucket_ranges)
    word_lengths = []
    for i, hypo in enumerate(hypos):
        word_count = len(hypo.strip().split())
        for idx, (low, high) in enumerate(bucket_ranges):
            if low <= word_count < high:
                counts[idx] += 1
                wers_sum[idx] += wers[i]
                break
        word_lengths.append(word_count)
    return counts, wers_sum, word_lengths

def per_utterance_wer(data):
    """
    Computes the WER for each utterance
    Args:
        data: a json file containing the hypotheses and the references.
    Returns:
        per_utterance_WER: a list containing WER per utterance
        avg_wer: the average WER for the given data
        sids: A list containing the number of S/I/Ds
    """
    avg_err, avg_total = 0, 0
    per_utterance_WER = []
    sids = []
    for hypo, ref in zip(data['hypo'], data['ref']):
        n_err, n_total = 0, 0 # Per Utterance Counters
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)
        per_utterance_WER.append(100 * n_err / n_total)
        avg_err += editdistance.eval(hypo, ref)
        avg_total += len(ref)
        ref_str  = " ".join(ref)
        hypo_str = " ".join(hypo)
        output = process_words(ref_str, hypo_str)
        # Extract the alignment details
        subs = output.substitutions
        inserts = output.insertions
        dels = output.deletions
        sids.append([subs, inserts, dels])


    
    avg_wer = 100 * avg_err / avg_total

    return np.array(per_utterance_WER), avg_wer, sids

def phoneme_to_viseme(phoneme):
  """
  A function which maps a phoneme to a viseme, depending on the mapping
  :param phoneme: The phoneme to be mapped
  :param map: mapping to be used for the phoneme to viseme.
  Possible mappings:
  - Lee
  - Returns:
  :param viseme: Mapped viseme
  """
  viseme_map = {
       'b': 'P', 'p': 'P', 'm': 'P',
       'd': 'T', 't': 'T', 's': 'T', 'z': 'T', 'th': 'T', 'dh': 'T',
       'g': 'K', 'k': 'K', 'n': 'K', 'ng': 'K', 'l': 'K', 'y': 'K', 'hh': 'K',
       'jh': 'CH', 'ch': 'CH', 'sh': 'CH', 'zh': 'CH',
       'f': 'F', 'v': 'F',
       'r': 'W', 'w': 'W',
       'iy': 'IY', 'ih': 'IY',
       'eh': 'EH', 'ey': 'EH', 'ae': 'EH',
       'aa': 'AA', 'aw': 'AA', 'ay': 'AA', # 'ah': 'EH', duplicate
       'ah': 'AH', 
       'ao': 'AO', 'oy': 'AO', 'ow': 'AO',
       'uh': 'UH', 'uw': 'UH',
       'er': 'ER'
     }
  
  viseme = viseme_map.get(phoneme.lower(), 'S')
  return viseme

def per_utterance_cer(data):
    """
    Compute per-utterance character error rate.
    data: dictionary with keys 'utt_id' and 'hyp' (predicted text) and 'ref' (reference text)
    Returns:
        per_utt_cer: np.array of CER per utterance
        avg_cer: float, average CER
        utt_ids: list of utterance IDs
    """
    import editdistance  # fast Levenshtein distance
    per_utt_cer = []
    for ref, hyp in zip(data['ref'], data['hypo']):
        ref_chars = list(ref.replace(" ", ""))  # remove spaces for CER
        hyp_chars = list(hyp.replace(" ", ""))
        cer = editdistance.eval(hyp_chars, ref_chars) / max(1, len(ref_chars))
        per_utt_cer.append(cer)
    
    return np.array(per_utt_cer), 100 * sum(per_utt_cer)/len(per_utt_cer)


def per_utterance_per(data):
    """
    Compute per-utterance phoneme error rate (PER) using ARPAbet phonemes.
    data: dictionary with keys 'utt_id', 'ref', 'hypo'
    Returns:
        per_utt_per: list of PER per utterance
        avg_per: float, average PER
        utt_ids: list of utterance IDs
    """
    per_utt_per = []
    utt_ids = data['utt_id']
    g2p = G2p()
    for ref_text, hyp_text in zip(data['ref'], data['hypo']):
        # Convert reference and hypothesis to ARPAbet phonemes
        ref_phons = [re.sub(r'\d', '', p) for p in g2p(ref_text) if p.isalpha() or p.isupper()]
        hyp_phons = [re.sub(r'\d', '', p) for p in g2p(hyp_text) if p.isalpha() or p.isupper()]
        
        per_val = editdistance.eval(hyp_phons, ref_phons) / max(1, len(ref_phons))
        per_utt_per.append(per_val)
    
    return np.array(per_utt_per), 100 * sum(per_utt_per)/len(per_utt_per)

def per_utterance_ver(data):
    """
    Compute per-utterance viseme error rate (VER) using ARPAbet phonemes and Lee's viseme mapping.
    data: dictionary with keys 'utt_id', 'ref', 'hypo'
    Returns:
        per_utt_ver: list of VER per utterance
        avg_per: float, average PER
        utt_ids: list of utterance IDs
    """
    per_utt_ver = []
    utt_ids = data['utt_id']
    g2p = G2p()
    for ref_text, hyp_text in zip(data['ref'], data['hypo']):
        # Convert reference and hypothesis to ARPAbet phonemes
        ref_phons = [re.sub(r'\d', '', p) for p in g2p(ref_text) if p.isalpha() or p.isupper()]
        hyp_phons = [re.sub(r'\d', '', p) for p in g2p(hyp_text) if p.isalpha() or p.isupper()]
        ref_viseme_trans = [phoneme_to_viseme(x) for x in ref_phons]
        hypo_viseme_trans = [phoneme_to_viseme(x) for x in hyp_phons]

        per_val = editdistance.eval(hypo_viseme_trans, ref_viseme_trans) / max(1, len(ref_viseme_trans))
        per_utt_ver.append(per_val)
    
    return np.array(per_utt_ver), 100 * sum(per_utt_ver)/len(per_utt_ver)



def plot_sid_comparison(per_utt_model1, per_utt_model2, title, savepath, labels=("Baseline","Viseme Model")):
    """
    Plot side-by-side bar chart of S/I/D errors for two models.

    Args:
        per_utt_model1: list of [S,I,D] per utterance for model 1
        per_utt_model2: list of [S,I,D] per utterance for model 2
    """
    # Convert per-utterance counts to totals per category
    totals_model1 = [sum(col) for col in zip(*per_utt_model1)]
    totals_model2 = [sum(col) for col in zip(*per_utt_model2)]

    categories = ['Substitutions', 'Insertions', 'Deletions']
    x = range(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7,5))
    rects1 = ax.bar([i - width/2 for i in x], totals_model1, width, label=labels[0])
    rects2 = ax.bar([i + width/2 for i in x], totals_model2, width, label=labels[1])

    ax.set_ylabel("Total errors")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title(title)
    ax.legend()

    # Annotate bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(savepath)


def load_viseme_file(filename):
    """
    Reads a viseme file where each line is a space-separated sequence of visemes.
    
    Returns:
        visemes: list of lists of visemes per utterance
    """
    visemes = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            viseme_seq = line.split()
            visemes.append(viseme_seq)
    return visemes

def find_utts(better_idxs, worse_idxs, vsm_data, base_data, verbose=False):
    """
    Finds utterances that the viseme model performs better or under-performs
    Args:
        better_idxs: List of indices where viseme model is better
        worse_idxs: List of indices where viseme model is worse
        vsm_data: Dictionary with keys 'utt_id', 'ref', 'hypo' for viseme model
        base_data: Dictionary with keys 'utt_id', 'ref', 'hypo' for baseline model
        verbose: If True, print the utterances
    Returns:
        better_utts: List of utterance IDs where viseme model is better
        worse_utts: List of utterance IDs where viseme model is worse
    """
    worse_utts = []
    better_utts = []
    for idx1 in better_idxs:
        better_utt = vsm_data['utt_id'][idx1]
        ref_utt = vsm_data['ref'][idx1]
        hypo_utt = vsm_data['hypo'][idx1]
        base_hyp_utt = base_data['hypo'][idx1]
        better_utts.append(better_utt)
        if verbose:
            print("--" * 100)
            print(f"REF: {ref_utt}\n")
            print(f"VIS_HYP: {hypo_utt}\n")
            print(f"Base_HYP: {base_hyp_utt}\n")
            print("--" * 100)
        
    for idx2 in worse_idxs:
        worse_utt = vsm_data['utt_id'][idx2]
        ref_utt = vsm_data['ref'][idx2]
        hypo_utt = vsm_data['hypo'][idx2]
        base_hyp_utt = base_data['hypo'][idx2]
        worse_utts.append(worse_utt)
        if verbose:
            print("--" * 100)
            print(f"REF: {ref_utt}\n")
            print(f"VIS_HYP: {hypo_utt}\n")
            print(f"Base_HYP: {base_hyp_utt}\n")
            print("--" * 100)
        
    return better_utts, worse_utts

def show_wrong_segments_with_context(
    video_path,
    ref,
    visg_hypo,
    avhubert_hypo,
    savepath=None,
    alignment_path=None,
    context=1,
    fps=25,
    mode="better"
):
    """
    Visualize mispredicted words with phone-level alignment for
    baseline AV-HuBERT vs VisG AV-HuBERT, handling missing words.
    """

    # --- Load frames ---
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    total_frames = len(frames)

    # --- Prepare words ---
    ref_words = ref.strip().split()
    visg_words = visg_hypo.strip().split()
    base_words = avhubert_hypo.strip().split()

    # --- Select hypothesis to evaluate ---
    if mode == "better":
        chosen_hypo_words = base_words
        aligned_words = align_word_sequences(ref_words, base_words)
        print("✅ Mode: BETTER → visualizing baseline errors (AV-HuBERT)")
    elif mode == "worse":
        chosen_hypo_words = visg_words
        aligned_words = align_word_sequences(ref_words, visg_words)
        print("❌ Mode: WORSE → visualizing VisG AV-HuBERT errors")
    else:
        raise ValueError("mode must be 'better' or 'worse'")

    # --- Identify mispredicted words ---
    wrong_indices = [i for i, (_, _, _, _, match) in enumerate(aligned_words) if not match]
    if not wrong_indices:
        print("✅ No wrong predictions; nothing to visualize.")
        return

    # --- Expand with context ---
    expanded_indices = set()
    for idx in wrong_indices:
        for offset in range(-context, context + 1):
            new_idx = idx + offset
            if 0 <= new_idx < len(aligned_words):
                expanded_indices.add(new_idx)
    expanded_indices = sorted(expanded_indices)

    # --- Pronunciation helpers ---
    def get_pronunciation(word):
        if not word:
            return None
        phones_list = pronouncing.phones_for_word(word.lower())
        if phones_list:
            return phones_list[0].split()
        return None

    def infer_predicted_phones(word):
        if not word:
            return None
        g2p = G2p()
        phones_list = pronouncing.phones_for_word(word.lower())
        if phones_list:
            return phones_list[0].split()
        try:
            predicted = g2p(word.lower())
            phones = [re.sub(r"\d", "", p) for p in predicted if p not in [" ", ""]]
            if phones:
                return phones
        except Exception:
            pass
        return None

    # --- Load alignment if available ---
    if alignment_path and os.path.exists(alignment_path):
        print("🕒 Using phone-level alignment visualization...")
        alignment_df = pd.read_csv(alignment_path)
        words_df = alignment_df[alignment_df["Type"] == "words"].copy().reset_index(drop=True)
        phones_df = alignment_df[alignment_df["Type"] == "phones"].copy().reset_index(drop=True)

        word_labels_list = words_df["Label"].str.lower().tolist()
        word_begins = words_df["Begin"].tolist()
        word_ends = words_df["End"].tolist()

        # Map reference words to time ranges
        ref_words_lower = [w.lower() for w in ref_words]
        word_time_ranges = {}
        ref_idx_counter = 0
        for i, label in enumerate(word_labels_list):
            if ref_idx_counter < len(ref_words_lower) and ref_words_lower[ref_idx_counter] == label:
                word_time_ranges[ref_idx_counter] = (word_begins[i], word_ends[i])
                ref_idx_counter += 1

        # Map phones → words
        word_to_phones = {}
        for phone_idx, phone_row in phones_df.iterrows():
            for ref_idx, (begin, end) in word_time_ranges.items():
                if (phone_row["Begin"] >= begin and phone_row["Begin"] < end) or \
                   (phone_row["End"] > begin and phone_row["End"] <= end) or \
                   (phone_row["Begin"] <= begin and phone_row["End"] >= end):
                    word_to_phones.setdefault(ref_idx, []).append(phone_idx)
                    break

        # Frames from alignment
        phones_df["begin_frame"] = (phones_df["Begin"] * fps).round().astype(int)
        phones_df["end_frame"] = (phones_df["End"] * fps).round().astype(int)

        word_images, word_labels_meta = [], []

        # Perform both alignments to get corresponding words from both models
        base_aligned = align_word_sequences(ref_words, base_words)
        visg_aligned = align_word_sequences(ref_words, visg_words)

        for idx in expanded_indices:
            ref_idx, hypo_idx, ref_word, hypo_word, is_match = aligned_words[idx]

            # Skip if reference word is missing (insertion in hypo)
            if ref_idx is None:
                continue

            is_wrong = not is_match

            # Get corresponding words from both models using ref_idx
            base_word = None
            visg_word = None
            
            for b_ref_idx, b_hypo_idx, _, b_hypo_word, _ in base_aligned:
                if b_ref_idx == ref_idx and b_hypo_idx is not None:
                    base_word = b_hypo_word
                    break
            
            for v_ref_idx, v_hypo_idx, _, v_hypo_word, _ in visg_aligned:
                if v_ref_idx == ref_idx and v_hypo_idx is not None:
                    visg_word = v_hypo_word
                    break

            # Expected phones
            expected_phones = get_pronunciation(ref_word)
            # Baseline phones
            avhubert_phones = infer_predicted_phones(base_word)
            # VisG phones
            visg_phones = infer_predicted_phones(visg_word)

            phone_frames = []
            if ref_idx in word_to_phones:
                for i, phone_idx in enumerate(word_to_phones[ref_idx]):
                    row = phones_df.loc[phone_idx]
                    start_f = max(0, row["begin_frame"])
                    end_f = min(row["end_frame"], total_frames)
                    mid_f = start_f if start_f >= end_f else (start_f + end_f) // 2
                    if mid_f >= total_frames:
                        continue
                    frame = frames[mid_f].copy()

                    # Frame-level red border for differing phonemes
                    diff_phone_indices = []
                    if avhubert_phones and visg_phones:
                        min_len = min(len(avhubert_phones), len(visg_phones))
                        diff_phone_indices = [j for j in range(min_len) if avhubert_phones[j] != visg_phones[j]]

                    if i in diff_phone_indices:
                        frame[:5, :] = [255, 0, 0]
                        frame[-5:, :] = [255, 0, 0]
                        frame[:, :5] = [255, 0, 0]
                        frame[:, -5:] = [255, 0, 0]

                    phone_frames.append(frame)

            if not phone_frames:
                continue

            word_strip = np.hstack(phone_frames)
            word_images.append(word_strip)
            word_labels_meta.append({
                "ref": ref_word,
                "expected_phones": expected_phones,
                "avhubert_phones": avhubert_phones,
                "visg_phones": visg_phones,
                "base_hypo": base_word if base_word else "",
                "visg_hypo": visg_word if visg_word else "",
                "is_wrong": is_wrong,
                "ref_idx": ref_idx
            })

    else:
        print("🖼️ No alignment file; cannot visualize.")
        return

    # --- Plot ---
    n_words = len(word_images)
    fig = plt.figure(figsize=(20, n_words * 3.5))
    gs = fig.add_gridspec(n_words, 1, hspace=0.5)

    for i, (img, label) in enumerate(zip(word_images, word_labels_meta)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        color = "red" if label["is_wrong"] else "blue"

        # Get both hypothesis texts for comparison
        base_text = label['base_hypo'] if label['base_hypo'] else "(deleted)"
        visg_text = label['visg_hypo'] if label['visg_hypo'] else "(deleted)"

        # Build title with both hypotheses
        title = f"REF='{label['ref']}' | AV-HuBERT='{base_text}' | VisG AV-HuBERT='{visg_text}'\n\n"
        title += f"Expected: [{' '.join(label['expected_phones']) if label['expected_phones'] else '(none)'}]\n"
        title += f"AV-HuBERT: [{' '.join(label['avhubert_phones']) if label['avhubert_phones'] else '(none)'}]\n"
        title += f"VisG AV-HuBERT: [{' '.join(label['visg_phones']) if label['visg_phones'] else '(none)'}]"

        ax.set_title(title, color=color, fontsize=11,
                     fontweight="bold" if label["is_wrong"] else "normal", pad=10)
        ax.axis("off")

    # --- Footer text ---
    fig.text(0.5, 0.08, f"REFERENCE: {ref}", fontsize=12, color="black", ha="center", weight="bold")
    fig.text(0.5, 0.05, f"AV-HuBERT Hypo: {avhubert_hypo}", fontsize=12, color="red", ha="center", weight="bold")
    fig.text(0.5, 0.02, f"VisG AV-HuBERT Hypo: {visg_hypo}", fontsize=12, color="green", ha="center", weight="bold")

    plt.subplots_adjust(bottom=0.1, top=0.98)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        print(f"Visualization saved to: {savepath}")
    else:
        plt.show()

