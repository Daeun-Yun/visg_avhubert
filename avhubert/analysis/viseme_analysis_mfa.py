import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import matplotlib.patches as mpatches
from g2p_en import G2p
import re
import jiwer
import editdistance
from plot_utils import phoneme_to_viseme

# ===================================================================
# 1. Load reference CSV alignment files (with viseme column)
# ===================================================================

def load_alignment_csv(csv_path):
    """
    Load CSV alignment file with columns: phoneme, start, end, viseme
    Returns list of dicts with phoneme, viseme, start, end timestamps
    """
    items = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "phoneme": row["Label"],
                "viseme": row["Viseme"],
                "start": float(row["Begin"]),
                "end": float(row["End"])
            })
    return items


# ===================================================================
# 2. Text to phoneme/viseme conversion
# ===================================================================

def normalize_phoneme(phoneme):
    """
    Normalize phoneme for consistent comparison.
    Removes stress markers and converts to uppercase.
    """
    return re.sub(r'[0-9]', '', phoneme.upper()).strip()


def text_to_phoneme_sequence(text, g2p):
    """
    Convert text to phoneme sequence using G2P.
    Returns list of phonemes (cleaned, no stress markers, uppercase).
    """
    phonemes = g2p(text)
    
    # Clean phonemes
    clean_phonemes = []
    for p in phonemes:
        # Skip non-phoneme characters
        if not any(c.isalpha() for c in p):
            continue
        
        # Normalize: uppercase, remove stress markers
        clean_p = re.sub(r'[0-9]', '', p.upper()).strip()
        
        if clean_p and clean_p not in ['', 'SIL', 'SP', 'SPN']:
            clean_phonemes.append(clean_p)
    
    return clean_phonemes


# ===================================================================
# 3. Align predicted phoneme sequence to reference using jiwer
# ===================================================================

def align_phoneme_sequences_with_jiwer(ref_phonemes, pred_phonemes, debug=False):
    """
    Align predicted phoneme sequence to reference using jiwer.
    Returns list of (ref_phoneme, pred_phoneme, error_type) tuples.
    error_type is one of: 'equal', 'substitute', 'delete', 'insert'
    """
    if not ref_phonemes and not pred_phonemes:
        return []
    
    if not ref_phonemes:
        return [(None, p, 'insert') for p in pred_phonemes]
    
    if not pred_phonemes:
        return [(r, None, 'delete') for r in ref_phonemes]
    
    # Convert to space-separated strings for jiwer
    ref_str = ' '.join(ref_phonemes)
    pred_str = ' '.join(pred_phonemes)
    
    if debug:
        print(f"\nDEBUG alignment:")
        print(f"  Ref phonemes ({len(ref_phonemes)}): {ref_str[:100]}...")
        print(f"  Pred phonemes ({len(pred_phonemes)}): {pred_str[:100]}...")
    
    # Use jiwer to get alignment
    output = jiwer.process_words(ref_str, pred_str)
    
    # Build alignment pairs from jiwer chunks
    aligned_pairs = []
    
    for chunk in output.alignments[0]:
        chunk_type = chunk.type
        
        if chunk_type == "equal":
            ref_indices = range(chunk.ref_start_idx, chunk.ref_end_idx)
            hyp_indices = range(chunk.hyp_start_idx, chunk.hyp_end_idx)
            
            for r_idx, h_idx in zip(ref_indices, hyp_indices):
                ref_ph = ref_phonemes[r_idx] if r_idx < len(ref_phonemes) else None
                pred_ph = pred_phonemes[h_idx] if h_idx < len(pred_phonemes) else None
                aligned_pairs.append((ref_ph, pred_ph, 'equal'))
        
        elif chunk_type == "substitute":
            ref_indices = range(chunk.ref_start_idx, chunk.ref_end_idx)
            hyp_indices = range(chunk.hyp_start_idx, chunk.hyp_end_idx)
            
            for r_idx, h_idx in zip(ref_indices, hyp_indices):
                ref_ph = ref_phonemes[r_idx] if r_idx < len(ref_phonemes) else None
                pred_ph = pred_phonemes[h_idx] if h_idx < len(pred_phonemes) else None
                aligned_pairs.append((ref_ph, pred_ph, 'substitute'))
        
        elif chunk_type == "delete":
            for r_idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ref_ph = ref_phonemes[r_idx] if r_idx < len(ref_phonemes) else None
                aligned_pairs.append((ref_ph, None, 'delete'))
        
        elif chunk_type == "insert":
            for h_idx in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                pred_ph = pred_phonemes[h_idx] if h_idx < len(pred_phonemes) else None
                aligned_pairs.append((None, pred_ph, 'insert'))
    
    if debug:
        print(f"  Aligned pairs: {len(aligned_pairs)}")
        print(f"  Sample pairs: {aligned_pairs[:10]}")
    
    return aligned_pairs


# ===================================================================
# 4. Compute error breakdown from aligned phoneme pairs
# ===================================================================

def compute_viseme_error_breakdown_from_phonemes(aligned_phoneme_pairs, ref_alignment_items):
    """
    Compute per-viseme error breakdown directly from aligned phoneme pairs.
    This properly handles insertions by tracking predicted visemes.
    
    Returns:
        breakdown: dict with 'sub', 'ins', 'del', 'counts' for each viseme
    """
    # Reference viseme counts
    ref_viseme_counts = defaultdict(int)
    
    # Track errors by reference viseme
    substitutions = defaultdict(int)
    deletions = defaultdict(int)
    
    # Track insertions by predicted viseme (since no reference exists)
    insertions_by_pred = defaultdict(int)
    
    # Build normalized reference phoneme list
    ref_phonemes_normalized = [normalize_phoneme(item["phoneme"]) for item in ref_alignment_items]
    ref_visemes = [item["viseme"] for item in ref_alignment_items]
    
    ref_idx = 0
    
    for aligned_ref_ph, aligned_pred_ph, error_type in aligned_phoneme_pairs:
        
        if error_type == 'insert':
            # Insertion: no reference, only prediction
            pred_viseme = phoneme_to_viseme(aligned_pred_ph) if aligned_pred_ph else None
            if pred_viseme and pred_viseme != 'sil':
                insertions_by_pred[pred_viseme] += 1
            continue
        
        if aligned_ref_ph is None:
            continue
        
        # Find matching reference phoneme
        aligned_ref_ph_norm = normalize_phoneme(aligned_ref_ph)
        
        found = False
        for i in range(ref_idx, len(ref_alignment_items)):
            ref_ph_norm = ref_phonemes_normalized[i]
            
            if ref_ph_norm == aligned_ref_ph_norm:
                ref_viseme = ref_visemes[i]
                
                # Skip silence
                if ref_viseme in ("", None, "sil", "/sil/"):
                    ref_idx = i + 1
                    found = True
                    break
                
                # Count reference viseme
                ref_viseme_counts[ref_viseme] += 1
                
                # Track error type
                if error_type == 'substitute':
                    substitutions[ref_viseme] += 1
                elif error_type == 'delete':
                    deletions[ref_viseme] += 1
                # 'equal' -> no error, just counted
                
                ref_idx = i + 1
                found = True
                break
    
    # For insertions, we need to attribute them to viseme classes
    # We'll track insertions per predicted viseme class
    insertions = dict(insertions_by_pred)
    
    # Prepare breakdown dictionary
    breakdown = {
        'sub': dict(substitutions),
        'ins': dict(insertions),
        'del': dict(deletions),
        'counts': dict(ref_viseme_counts)
    }
    
    return breakdown


def compute_overall_error_rates(breakdown):
    """
    Compute overall error rates from breakdown.
    
    Returns:
        error_rates: dict of viseme -> overall error rate
        stats: dict of viseme -> {correct, total}
    """
    stats = {}
    error_rates = {}
    
    for viseme in breakdown['counts']:
        total = breakdown['counts'][viseme]
        errors = (breakdown['sub'].get(viseme, 0) + 
                  breakdown['ins'].get(viseme, 0) + 
                 breakdown['del'].get(viseme, 0))
        correct = total - (breakdown['sub'].get(viseme, 0) + breakdown['del'].get(viseme, 0))
        
        stats[viseme] = {
            'correct': correct,
            'total': total
        }
        
        error_rates[viseme] = errors / total if total > 0 else 0
    
    return error_rates, stats


# ===================================================================
# 5. Process multiple test files
# ===================================================================

def load_predictions_json(json_path):
    """
    Load model predictions from JSON file.
    Returns dict mapping utt_id to prediction text.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    for i, utt_id in enumerate(data['utt_id']):
        predictions[utt_id] = data['hypo'][i]
    
    return predictions


def find_csv_for_utt_id(ref_csv_base_dir, utt_id):
    """
    Find CSV file for given utterance ID.
    """
    parts = utt_id.split('/')
    
    if len(parts) < 3:
        return None
    
    folder = parts[-2]
    file_id = parts[-1]
    
    # Try with _viseme_lee suffix
    csv_path = os.path.join(ref_csv_base_dir, folder, f"{file_id}_viseme_lee.csv")
    if os.path.exists(csv_path):
        return csv_path
    
    # Try without suffix
    csv_path = os.path.join(ref_csv_base_dir, folder, f"{file_id}.csv")
    if os.path.exists(csv_path):
        return csv_path
    
    return None


def process_test_set_per_utterance(ref_csv_dir, pred_json, g2p, debug=False):
    """
    Process all test set samples with PER-UTTERANCE averaging (matches original method).
    Also aggregates for breakdown plots.
    
    Returns:
        per_utt_vers: list of VER per utterance
        breakdown: aggregated breakdown for plotting
    """
    if not os.path.exists(ref_csv_dir):
        print(f"❌ ERROR: Reference CSV directory not found: {ref_csv_dir}")
        return None, None
    
    if not os.path.exists(pred_json):
        print(f"❌ ERROR: Prediction JSON file not found: {pred_json}")
        return None, None
    
    # Load predictions
    print(f"  Loading predictions from: {os.path.basename(pred_json)}")
    predictions = load_predictions_json(pred_json)
    print(f"  Found {len(predictions)} predictions in JSON")
    
    # Per-utterance VER list
    per_utt_vers = []
    
    # Aggregate breakdown across all utterances (for plotting)
    agg_breakdown = {
        'sub': defaultdict(int),
        'ins': defaultdict(int),
        'del': defaultdict(int),
        'counts': defaultdict(int)
    }
    
    processed = 0
    not_found = 0
    
    for idx, (utt_id, pred_text) in enumerate(predictions.items()):
        # Find corresponding CSV file
        csv_path = find_csv_for_utt_id(ref_csv_dir, utt_id)
        
        if csv_path is None:
            if debug and not_found < 3:
                print(f"  ⚠ CSV not found for: {utt_id}")
            not_found += 1
            continue
        
        # Load reference alignment
        try:
            ref_alignment = load_alignment_csv(csv_path)
            # Extract visemes (excluding silence)
            ref_visemes = [item["viseme"] for item in ref_alignment 
                          if item["viseme"] not in ("", None, "sil", "/sil/")]
        except Exception as e:
            print(f"  ⚠ Error loading CSV {csv_path}: {e}")
            continue
        
        # Convert prediction text to visemes
        pred_phonemes = text_to_phoneme_sequence(pred_text, g2p)
        pred_visemes = [phoneme_to_viseme(p) for p in pred_phonemes]
        
        # Debug first utterance
        if debug and processed == 0:
            print(f"\n  DEBUG first utterance:")
            print(f"    Utt ID: {utt_id}")
            print(f"    Pred text: {pred_text}")
            print(f"    Ref visemes: {ref_visemes[:20]}")
            print(f"    Pred visemes: {pred_visemes[:20]}")
        
        # Compute edit distance VER for this utterance (matching original method)
        dist = editdistance.eval(pred_visemes, ref_visemes)
        ver = dist / max(1, len(ref_visemes))
        per_utt_vers.append(ver)
        
        # Also compute breakdown for plots
        ref_phonemes = [normalize_phoneme(item["phoneme"]) 
                       for item in ref_alignment 
                       if item["viseme"] not in ("", None, "sil", "/sil/")]
        
        aligned_pairs = align_phoneme_sequences_with_jiwer(
            ref_phonemes, pred_phonemes, debug=(debug and processed == 0)
        )
        
        utt_breakdown = compute_viseme_error_breakdown_from_phonemes(
            aligned_pairs, ref_alignment
        )
        
        # Aggregate for plots
        for viseme, count in utt_breakdown['counts'].items():
            agg_breakdown['counts'][viseme] += count
        for viseme, count in utt_breakdown['sub'].items():
            agg_breakdown['sub'][viseme] += count
        for viseme, count in utt_breakdown['ins'].items():
            agg_breakdown['ins'][viseme] += count
        for viseme, count in utt_breakdown['del'].items():
            agg_breakdown['del'][viseme] += count
        
        processed += 1
    
    print(f"✓ Processed {processed} utterances ({not_found} not found)")
    
    # Convert defaultdicts to regular dicts
    breakdown = {
        'sub': dict(agg_breakdown['sub']),
        'ins': dict(agg_breakdown['ins']),
        'del': dict(agg_breakdown['del']),
        'counts': dict(agg_breakdown['counts'])
    }
    
    return per_utt_vers, breakdown


# ===================================================================
# 6. Plotting functions
# ===================================================================

def plot_viseme_error_rates_side_by_side(
        err_base, err_vsm, savepath=None, title_suffix="", model_name="AV-HuBERT"):
    """Plot per-viseme error rates for Baseline vs VisG side-by-side (excluding /sil/)."""
    # Filter out 'sil' viseme
    visemes = sorted([v for v in set(err_base.keys()) | set(err_vsm.keys()) if v != '/sil/'])
    
    if len(visemes) == 0:
        print("⚠ No visemes to plot - skipping side-by-side plot")
        return
    
    base_vals = [err_base.get(v, 0) * 100 for v in visemes]
    vsm_vals = [err_vsm.get(v, 0) * 100 for v in visemes]
    
    x = np.arange(len(visemes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(18, 6))
    base_bars = ax.bar(x - width/2, base_vals, width, label='AV-HuBERT', alpha=0.8)
    vsm_bars = ax.bar(x + width/2, vsm_vals, width, label='VisG AV-HuBERT', alpha=0.8)
    
    ymax = max(max(base_vals), max(vsm_vals)) * 1.15 if base_vals and vsm_vals else 100
    ax.set_ylim(0, ymax)
    
    for bars in (base_bars, vsm_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10
            )
    
    ax.set_xticks(x)
    ax.set_xticklabels(visemes, fontsize=14)
    ax.set_ylabel('Error Rate (%)', fontsize=16)
    ax.set_xlabel('Viseme Class', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_viseme_improvements(err_base, err_vsm, savepath=None, model_name="AV-HuBERT"):
    """Plot per-viseme error rate changes (VisG - Baseline) (excluding /sil/)."""
    # Filter out 'sil' viseme
    visemes = sorted([v for v in set(err_base.keys()) | set(err_vsm.keys()) if v != '/sil/'])
    
    if len(visemes) == 0:
        print("⚠ No visemes to plot - skipping improvement plot")
        return
    
    improvements = []
    for v in visemes:
        base_rate = err_base.get(v, 0) * 100
        vsm_rate = err_vsm.get(v, 0) * 100
        change = vsm_rate - base_rate
        improvements.append((v, change, base_rate, vsm_rate))
    
    improvements.sort(key=lambda x: x[1])
    
    visemes_sorted = [x[0] for x in improvements]
    changes = [x[1] for x in improvements]
    
    colors = ['green' if c < 0 else 'red' for c in changes]
    
    fig, ax = plt.subplots(figsize=(18, 6))
    bars = ax.bar(visemes_sorted, changes, color=colors, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ymin = min(changes) - 1 if changes else -1
    ymax = max(changes) + 1 if changes else 1
    ax.set_ylim(ymin, ymax)
    
    for bar, value in zip(bars, changes):
        height = bar.get_height()
        offset = (ymax - ymin) * 0.03
        ax.annotate(
            f"{value:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset if value >= 0 else -offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=14
        )
    
    ax.set_xlabel('Viseme Class', fontsize=18)
    ax.set_ylabel('Error Rate Change (%)', fontsize=18)
    ax.grid(axis='y', alpha=0.3)
    improved_patch = mpatches.Patch(color='green', label='Improved')
    degraded_patch = mpatches.Patch(color='red', label='Degraded')
    ax.legend(handles=[improved_patch, degraded_patch], fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    # plt.show()
    
    if savepath:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()
    
    print("\n" + "="*70)
    print("PER-VISEME ERROR ANALYSIS (MFA + Per-Utterance Averaging)")
    print("="*70)
    print("\nTop 10 Most Improved Visemes:")
    for viseme, diff, b_rate, v_rate in improvements[:10]:
        print(f"  {viseme:4s}: {diff:+6.2f}% | Baseline: {b_rate:5.2f}% → VisG: {v_rate:5.2f}%")
    
    if len(improvements) > 10:
        print("\nBottom 5 (Most Degraded):")
        for viseme, diff, b_rate, v_rate in improvements[-5:]:
            print(f"  {viseme:4s}: {diff:+6.2f}% | Baseline: {b_rate:5.2f}% → VisG: {v_rate:5.2f}%")
    
    print("="*70)


def plot_viseme_error_breakdown(errors, savepath, model_name='Model'):
    """
    Plot detailed S/I/D breakdown for each viseme (excluding /sil/).
    Insertions are shown as a separate component (not normalized by reference counts).
    """
    
    # Get all visemes from both reference counts and insertions, excluding 'sil'
    all_visemes = sorted([v for v in (set(errors['counts'].keys()) | set(errors['ins'].keys())) 
                          if v != '/sil/'])
    
    if not all_visemes:
        print("⚠ No visemes to plot in error breakdown!")
        return
    
    # Prepare data for each error type
    substitutions = [errors['sub'].get(v, 0) for v in all_visemes]
    insertions = [errors['ins'].get(v, 0) for v in all_visemes]
    deletions = [errors['del'].get(v, 0) for v in all_visemes]
    counts = [errors['counts'].get(v, 0) for v in all_visemes]
    
    # Convert to percentages (S and D normalized by reference count)
    sub_rates = [100 * s / c if c > 0 else 0 for s, c in zip(substitutions, counts)]
    del_rates = [100 * d / c if c > 0 else 0 for d, c in zip(deletions, counts)]
    
    # Insertions: normalize by reference count if exists, else show raw count scaled
    ins_rates = [100 * i / c if c > 0 else i * 10 for i, c in zip(insertions, counts)]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(all_visemes))
    width = 0.6
    
    ax.bar(x, sub_rates, width, label='Substitutions', color='#e74c3c')
    ax.bar(x, ins_rates, width, bottom=sub_rates, label='Insertions', color='#3498db')
    ax.bar(x, del_rates, width, 
           bottom=[s+i for s, i in zip(sub_rates, ins_rates)], 
           label='Deletions', color='#f39c12')
    
    ax.set_xlabel('Viseme Class', fontsize=16)
    ax.set_ylabel('Error Rate (%)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(all_visemes, fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


# ===================================================================
# 7. Main execution pipeline
# ===================================================================

if __name__ == "__main__":
    g2p = G2p()
    
    model = 'large'
    base_experiment_params = {'Name': 'Exp7'}
    large_experiment_params = {'Name': 'Exp5'}
    
    conditions = []
    for noise in ['babble', 'speech', 'music', 'noise']:
        for db in [-10, -5, 0, 5, 10]:
            conditions.append({'noise_type': noise, 'snr': db})
    conditions.append({'noise_type': None, 'snr': None})
    
    for cond in conditions:
        noise_type = cond['noise_type']
        snr = cond['snr']
        noise_path = f'{noise_type}_snr{snr}' if noise_type else None
        
        base_save_path = f'/home/aristosp/plots/selected_{model}/viseme_analysis/'
        experiment_params = base_experiment_params['Name'] if model == 'base' else large_experiment_params['Name']
        
        print("\n" + "="*70)
        print(f"ANALYZING {model.upper()} MODEL — {noise_path if noise_path else 'CLEAN'}")
        print("="*70)
        
        save_path = base_save_path + (noise_path + '/' if noise_path else 'clean/')
        os.makedirs(save_path, exist_ok=True)
        
        ref_csv_dir = '/home/aristosp/datasets/LRS3/audio/aligned/test/'
        
        if noise_type is not None:
            baseline_json = f'/home/aristosp/models/av_hubert/{model}_decode/{noise_path}/hypo-244018.json'
            vsm_json = f'/home/aristosp/viseme_{model}_models/{experiment_params}/decode/{noise_path}/hypo-244018.json'
        else:
            baseline_json = f'/home/aristosp/models/av_hubert/{model}_decode/hypo-244018.json'
            vsm_json = f'/home/aristosp/viseme_{model}_models/{experiment_params}/decode/hypo-244018.json'
        
        print("Processing baseline model (MFA + per-utterance averaging)...")
        per_utt_vers_baseline, breakdown_baseline = process_test_set_per_utterance(
            ref_csv_dir, baseline_json, g2p, debug=True
        )
        
        print("Processing VisG model (MFA + per-utterance averaging)...")
        per_utt_vers_vsm, breakdown_vsm = process_test_set_per_utterance(
            ref_csv_dir, vsm_json, g2p, debug=False
        )
        
        if per_utt_vers_baseline is None or per_utt_vers_vsm is None:
            print("⚠ WARNING: No data processed, skipping this condition")
            continue
        
        # Compute average VER using per-utterance method (matches original)
        avg_ver_baseline = 100 * np.mean(per_utt_vers_baseline)
        avg_ver_vsm = 100 * np.mean(per_utt_vers_vsm)
        
        print(f"\nOverall Viseme Error Rate (VER) - Per-Utterance Average:")
        print(f"  Baseline: {avg_ver_baseline:.2f}%")
        print(f"  VisG: {avg_ver_vsm:.2f}%")
        print(f"  Change: {avg_ver_vsm - avg_ver_baseline:+.2f}%")
        
        # Compute per-viseme error rates for plots
        print("\nComputing per-viseme error rates for plots...")
        err_baseline, stats_baseline = compute_overall_error_rates(breakdown_baseline)
        err_vsm, stats_vsm = compute_overall_error_rates(breakdown_vsm)
        
        # Print insertion stats
        total_base_ins = sum(breakdown_baseline['ins'].values())
        total_vsm_ins = sum(breakdown_vsm['ins'].values())
        print(f"\nTotal Insertions:")
        print(f"  Baseline: {total_base_ins}")
        print(f"  VisG: {total_vsm_ins}")
        
        condition_label = noise_path if noise_path else "Clean"
        
        print("\nGenerating plots...")
        
        # Original comparison plots
        plot_viseme_error_rates_side_by_side(
            err_baseline, err_vsm,
            savepath=save_path + f'{model}_viseme_error_rates_{condition_label}.pdf',
            title_suffix=f"({condition_label})",
            model_name=f"AV-HuBERT {model}"
        )
        
        plot_viseme_improvements(
            err_baseline, err_vsm,
            savepath=save_path + f'{model}_viseme_improvements_{condition_label}.pdf',
            model_name=f"AV-HuBERT {model}"
        )
        
        # New detailed breakdown plots (now includes insertions)
        plot_viseme_error_breakdown(
            breakdown_baseline,
            savepath=save_path + f'{model}_baseline_error_breakdown_{condition_label}.pdf',
            model_name=f"AV-HuBERT {model} Baseline ({condition_label})"
        )
        
        plot_viseme_error_breakdown(
            breakdown_vsm,
            savepath=save_path + f'{model}_visg_error_breakdown_{condition_label}.pdf',
            model_name=f"AV-HuBERT {model} VisG ({condition_label})"
        )
        
        print(f"✓ Saved all plots to {save_path}")
