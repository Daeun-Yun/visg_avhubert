import json
import matplotlib.pyplot as plt
from plot_utils import per_utterance_wer, phoneme_to_viseme
import jiwer
from collections import defaultdict
import numpy as np
import re
from g2p_en import G2p
import os



def compute_per_viseme_errors(ref_visemes, hyp_visemes, debug=True):
    """
    Compute substitution, insertion, deletion errors per viseme class.
    Uses jiwer alignment to track which visemes have errors.
    """
    if not ref_visemes and not hyp_visemes:
        return {}, {}, {}, {}
    
    # Convert lists to space-separated strings for jiwer
    ref_str = ' '.join(ref_visemes)
    hyp_str = ' '.join(hyp_visemes)
    
    output = jiwer.process_words(ref_str, hyp_str)
    
    # Extract errors by iterating through alignment chunks
    substitutions = defaultdict(int)
    insertions = defaultdict(int)
    deletions = defaultdict(int)
    # Count occurrences of each viseme in reference
    viseme_counts = defaultdict(int)
    for v in ref_visemes:
        viseme_counts[v] += 1
    
    # Debug first utterance
    if debug:
        print(f"\nDEBUG alignment info:")
        print(f"  Reference: {ref_str[:100]}")
        print(f"  Hypothesis: {hyp_str[:100]}")
        print(f"  Total S/I/D from jiwer: {output.substitutions}/{output.insertions}/{output.deletions}")
        print(f"  WER: {output.wer:.2%}")
        print(f"  Number of alignment chunks: {len(output.alignments[0])}")
        
        # Print first few chunks to see the structure
        for i, chunk in enumerate(output.alignments[0][:3]):
            print(f"    Chunk {i}: {chunk}")
    
    # Iterate through alignment chunks
    for chunk in output.alignments[0]:
        # Access chunk type
        chunk_type = chunk.type
        
        if chunk_type == 'substitute':
            # In older jiwer, use ref_start_idx and ref_end_idx
            for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if idx < len(ref_visemes):
                    substitutions[ref_visemes[idx]] += 1
                
        elif chunk_type == 'delete':
            for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if idx < len(ref_visemes):
                    deletions[ref_visemes[idx]] += 1
                
        elif chunk_type == 'insert':
            for idx in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if idx < len(hyp_visemes):
                    insertions[hyp_visemes[idx]] += 1
    
    if debug and output.substitutions + output.insertions + output.deletions > 0:
        print(f"  Extracted substitutions: {dict(substitutions)}")
        print(f"  Extracted insertions: {dict(insertions)}")
        print(f"  Extracted deletions: {dict(deletions)}")
    
    return substitutions, insertions, deletions, viseme_counts

def analyze_viseme_errors_from_json(baseline_file, vsm_file):
    """
    Analyze viseme-level errors from JSON prediction files.
    Returns per-viseme error statistics for baseline and VisG models.
    """
    # Load JSON files
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    with open(vsm_file, 'r') as f:
        vsm_data = json.load(f)

    # utt_ids = baseline_data['utt_id'] # same if read from vsm data
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

    # Initialize G2p
    g2p = G2p()
    
    # Initialize error accumulators
    baseline_subs = defaultdict(int)
    baseline_ins = defaultdict(int)
    baseline_dels = defaultdict(int)
    baseline_counts = defaultdict(int)
    
    vsm_subs = defaultdict(int)
    vsm_ins = defaultdict(int)
    vsm_dels = defaultdict(int)
    vsm_counts = defaultdict(int)
    
    # Get lists from JSON (parallel arrays)
    num_utterances = len(baseline_data['utt_id'])
    
    print(f"Processing {num_utterances} utterances...")
    
    # Process each utterance by index
    for idx in range(num_utterances):
        utt_id = baseline_data['utt_id'][idx]
        
        # Get reference and hypotheses for this index
        ref_text = baseline_data['ref'][idx]
        baseline_hyp = baseline_data['hypo'][idx]
        vsm_hyp = vsm_data['hypo'][idx]
        # Convert to phonemes
        ref_phons = [re.sub(r'\d', '', p) for p in g2p(ref_text) if p.isalpha() or p.isupper()]
        baseline_hyp_phons = [re.sub(r'\d', '', p) for p in g2p(baseline_hyp) if p.isalpha() or p.isupper()]
        vsm_hyp_phons = [re.sub(r'\d', '', p) for p in g2p(vsm_hyp) if p.isalpha() or p.isupper()]
        # Convert to visemes
        ref_visemes = [phoneme_to_viseme(x) for x in ref_phons]
        baseline_vis = [phoneme_to_viseme(x) for x in baseline_hyp_phons]
        vsm_vis = [phoneme_to_viseme(x) for x in vsm_hyp_phons]
        
        # Debug first utterance
        if idx == 0:
            print(f"\nFirst utterance '{utt_id}':")
            print(f"  Reference text: '{ref_text}'")
            print(f"  Baseline text: '{baseline_hyp}'")
            print(f"  VisG text: '{vsm_hyp}'")
            print(f"  Ref visemes ({len(ref_visemes)}): {' '.join(ref_visemes[:20])}")
            print(f"  Baseline visemes ({len(baseline_vis)}): {' '.join(baseline_vis[:20])}")
            print(f"  VisG visemes ({len(vsm_vis)}): {' '.join(vsm_vis[:20])}")
        
        # Compute errors for baseline
        debug_this = (idx == 0)
        b_sub, b_ins, b_del, b_cnt = compute_per_viseme_errors(ref_visemes, baseline_vis, debug=debug_this)
        
        for viseme, count in b_sub.items():
            baseline_subs[viseme] += count
        for viseme, count in b_ins.items():
            baseline_ins[viseme] += count
        for viseme, count in b_del.items():
            baseline_dels[viseme] += count
        for viseme, count in b_cnt.items():
            baseline_counts[viseme] += count
        
        # Compute errors for VisG model
        v_sub, v_ins, v_del, v_cnt = compute_per_viseme_errors(ref_visemes, vsm_vis, debug=False)
        for viseme, count in v_sub.items():
            vsm_subs[viseme] += count
        for viseme, count in v_ins.items():
            vsm_ins[viseme] += count
        for viseme, count in v_del.items():
            vsm_dels[viseme] += count
        for viseme, count in v_cnt.items():
            vsm_counts[viseme] += count
    
    print(f"\nProcessed {num_utterances} utterances")
    print(f"Baseline total visemes: {sum(baseline_counts.values())}")
    print(f"VisG total visemes: {sum(vsm_counts.values())}")
    print(f"Baseline total errors (S+I+D): {sum(baseline_subs.values()) + sum(baseline_ins.values()) + sum(baseline_dels.values())}")
    print(f"VisG total errors (S+I+D): {sum(vsm_subs.values()) + sum(vsm_ins.values()) + sum(vsm_dels.values())}")
    
    return {
        'baseline': {
            'sub': baseline_subs, 
            'ins': baseline_ins, 
            'del': baseline_dels, 
            'counts': baseline_counts
        },
        'vsm': {
            'sub': vsm_subs, 
            'ins': vsm_ins, 
            'del': vsm_dels, 
            'counts': vsm_counts
        }
    }



def plot_viseme_error_breakdown(errors, savepath, model_name='Model'):
    """Plot detailed S/I/D breakdown for each viseme."""
    
    # Get all visemes
    all_visemes = sorted(errors['counts'].keys())
    
    if not all_visemes:
        print("No visemes to plot!")
        return
    
    # Prepare data for each error type
    substitutions = [errors['sub'].get(v, 0) for v in all_visemes]
    insertions = [errors['ins'].get(v, 0) for v in all_visemes]
    deletions = [errors['del'].get(v, 0) for v in all_visemes]
    counts = [errors['counts'].get(v, 1) for v in all_visemes]
    
    # Convert to percentages
    sub_rates = [100 * s / c for s, c in zip(substitutions, counts)]
    ins_rates = [100 * i / c for i, c in zip(insertions, counts)]
    del_rates = [100 * d / c for d, c in zip(deletions, counts)]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(13, 10))
    x = np.arange(len(all_visemes))
    width = 0.6
    
    ax.bar(x, sub_rates, width, label='Substitutions', color='#e74c3c')
    ax.bar(x, ins_rates, width, bottom=sub_rates, label='Insertions', color='#3498db')
    ax.bar(x, del_rates, width, 
           bottom=[s+i for s, i in zip(sub_rates, ins_rates)], 
           label='Deletions', color='#f39c12')
    
    ax.set_xlabel('Viseme Class', fontsize=18)
    ax.set_ylabel('Error Rate (%)', fontsize=18)
    ax.set_title(f'Per-Viseme Error Breakdown - {model_name}', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(all_visemes)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=18)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path + f'{model_name}_per_viseme_errors.pdf', dpi=300)
    # plt.show()


def plot_viseme_improvements(errors_baseline, savepath, errors_vsm):
    """Plot which visemes improved between baseline and VisG models."""
    
    # Get all visemes
    all_visemes = set()
    all_visemes.update(errors_baseline['counts'].keys())
    all_visemes.update(errors_vsm['counts'].keys())
    all_visemes = sorted(all_visemes)
    
    if not all_visemes:
        print("No visemes found to plot!")
        return
    
    # Calculate error rates and improvements
    improvements = []
    for viseme in all_visemes:
        # Baseline error rate
        b_total = errors_baseline['counts'].get(viseme, 0)
        if b_total == 0:
            continue
        b_errors = (errors_baseline['sub'].get(viseme, 0) + 
                errors_baseline['ins'].get(viseme, 0) + 
                errors_baseline['del'].get(viseme, 0))
        b_rate = b_errors / b_total
        
        # VisG error rate
        v_total = errors_vsm['counts'].get(viseme, 0)
        if v_total == 0:
            v_total = b_total  # Use baseline count if VisG missing
        v_errors = (errors_vsm['sub'].get(viseme, 0) + 
                errors_vsm['ins'].get(viseme, 0) + 
                errors_vsm['del'].get(viseme, 0))
        v_rate = v_errors / v_total if v_total > 0 else 0
        
        # CHANGED: difference = VisG - Baseline (negative = improvement)
        difference = (v_rate - b_rate) * 100  # % change
        improvements.append((viseme, difference, b_rate * 100, v_rate * 100, b_total))

    # Sort by difference (most improved = most negative)
    improvements.sort(key=lambda x: x[1])

    # Plot
    fig, ax = plt.subplots(figsize=(13, 5))
    visemes = [x[0] for x in improvements]
    impr = [x[1] for x in improvements]

    # CHANGED: Red for negative (improvement), green for positive (degradation)
    colors = ['red' if i < 0 else 'green' for i in impr]
    bars = ax.bar(visemes, impr, color=colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    # --- FIX: Add margin above/below tallest bars ---
    ymin = min(impr) - 0.5
    ymax = max(impr) + 0.8
    ax.set_ylim(ymin, ymax)

    # --- FIX: Dynamic annotation offset ---
    for bar, value in zip(bars, impr):
        height = bar.get_height()

        # offset = 3% of y-range
        offset = (ymax - ymin) * 0.03 

        ax.annotate(
            f"{value:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset if value >= 0 else -offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=16
        )

    ax.set_xlabel('Viseme Class', fontsize=16)
    ax.set_ylabel('Error Rate Change (%)', fontsize=16)
    ax.set_title('Per-Viseme Error Rate Change (VisG AV-HuBERT - AV-HuBERT)', fontsize=16)
    ax.grid(axis='y', alpha=0.3)
    # plt.xticks(rotation=45)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(save_path + f'{model}_viseme_change_rate.pdf', dpi=300)
    # plt.show()

    # Print statistics
    print("\n" + "="*70)
    print("PER-VISEME ERROR ANALYSIS")
    print("="*70)
    print("\nTop 10 Most Improved Visemes (Most Negative Change):")
    for viseme, diff, b_rate, v_rate, count in improvements[:10]:
        print(f"  {viseme:4s}: {diff:+6.2f}% | Baseline: {b_rate:5.2f}% → VisG: {v_rate:5.2f}% | n={count}")

    if len(improvements) > 10:
        print("\nBottom 5 (Most Degraded/Positive Change):")
        for viseme, diff, b_rate, v_rate, count in improvements[-5:]:
            print(f"  {viseme:4s}: {diff:+6.2f}% | Baseline: {b_rate:5.2f}% → VisG: {v_rate:5.2f}% | n={count}")

    # Overall statistics
    total_baseline_errors = sum(errors_baseline['sub'].values()) + sum(errors_baseline['ins'].values()) + sum(errors_baseline['del'].values())
    total_baseline_visemes = sum(errors_baseline['counts'].values())

    total_vsm_errors = sum(errors_vsm['sub'].values()) + sum(errors_vsm['ins'].values()) + sum(errors_vsm['del'].values())
    total_vsm_visemes = sum(errors_vsm['counts'].values())

    baseline_ver = 100 * total_baseline_errors / total_baseline_visemes if total_baseline_visemes > 0 else 0
    vsm_ver = 100 * total_vsm_errors / total_vsm_visemes if total_vsm_visemes > 0 else 0

    # CHANGED: difference = VisG - Baseline
    difference = vsm_ver - baseline_ver

    print(f"\nOverall Viseme Error Rate (VER):")
    print(f"  Baseline: {baseline_ver:.2f}%")
    print(f"  VisG:     {vsm_ver:.2f}%")
    print(f"  Change (VisG - Baseline): {difference:+.2f}%")
    print("="*70)

def plot_viseme_error_breakdown_side_by_side(errors_baseline, errors_vsm, savepath, model_name="Comparison"):
    """
    Plot per-viseme S/I/D error rates for two models side by side.
    """
    # Union of visemes
    all_visemes = sorted(set(errors_baseline['counts'].keys()) | set(errors_vsm['counts'].keys()))
    x = np.arange(len(all_visemes))
    width = 0.12  # width for each bar

    def extract_rates(errors):
        counts = [errors['counts'].get(v, 1) for v in all_visemes]
        subs = [100 * errors['sub'].get(v, 0) / c for v, c in zip(all_visemes, counts)]
        ins  = [100 * errors['ins'].get(v, 0) / c for v, c in zip(all_visemes, counts)]
        dels = [100 * errors['del'].get(v, 0) / c for v, c in zip(all_visemes, counts)]
        return subs, ins, dels

    b_subs, b_ins, b_dels = extract_rates(errors_baseline)
    v_subs, v_ins, v_dels = extract_rates(errors_vsm)

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot Baseline (left cluster)
    ax.bar(x - width,      b_subs, width, label="Baseline Sub", color="#e74c3c")
    ax.bar(x - width/3,    b_ins,  width, label="Baseline Ins", color="#3498db")
    ax.bar(x + width/3,    b_dels, width, label="Baseline Del", color="#f39c12")

    # Plot VisG (right cluster)
    ax.bar(x + width,      v_subs, width, label="VisG Sub", color="#e74c3c", alpha=0.55)
    ax.bar(x + width*5/3,  v_ins,  width, label="VisG Ins", color="#3498db", alpha=0.55)
    ax.bar(x + width*7/3,  v_dels, width, label="VisG Del", color="#f39c12", alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(all_visemes, fontsize=15)
    ax.set_xlabel("Viseme Class", fontsize=18)
    ax.set_ylabel("Error Rate (%)", fontsize=18)
    ax.set_title(f"Per-Viseme Error Breakdown — AV-HuBERT vs VisG AV-HuBERT({model_name})", fontsize=20)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=13, ncol=2)
    plt.tight_layout()

    plt.savefig(savepath + f"{model_name}_viseme_error_comparison.pdf", dpi=300)
    plt.close()
    # plt.show()


def plot_viseme_error_breakdown_two_subplots(errors_baseline, errors_vsm, savepath, model_name="AV-HuBERT"):
    """
    Make a two-subplot figure comparing Baseline vs VisG viseme error breakdowns.
    Each subplot uses the original stacked-bar visualization.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    all_visemes = sorted(set(errors_baseline['counts'].keys()) |
                         set(errors_vsm['counts'].keys()))

    if not all_visemes:
        print("No visemes found!")
        return

    def extract_rates(errors):
        counts = [errors['counts'].get(v, 1) for v in all_visemes]
        subs = [100 * errors['sub'].get(v, 0) / c for v, c in zip(all_visemes, counts)]
        ins  = [100 * errors['ins'].get(v, 0) / c for v, c in zip(all_visemes, counts)]
        dels = [100 * errors['del'].get(v, 0) / c for v, c in zip(all_visemes, counts)]
        return subs, ins, dels

    b_subs, b_ins, b_dels = extract_rates(errors_baseline)
    v_subs, v_ins, v_dels = extract_rates(errors_vsm)

    x = np.arange(len(all_visemes))
    width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    # ==== LEFT SUBPLOT (Baseline) ====
    ax = axes[0]
    ax.bar(x, b_subs, width, label='Sub', color='#e74c3c')
    ax.bar(x, b_ins, width, bottom=b_subs, label='Ins', color='#3498db')
    ax.bar(x, b_dels, width, bottom=[s+i for s, i in zip(b_subs, b_ins)],
           label='Del', color='#f39c12')

    ax.set_title("Baseline", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_visemes, fontsize=18)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel("Error Rate (%)", fontsize=18)
    ax.legend(fontsize=14)

    # ==== RIGHT SUBPLOT (VisG) ====
    ax = axes[1]
    ax.bar(x, v_subs, width, label='Sub', color='#e74c3c')
    ax.bar(x, v_ins, width, bottom=v_subs, label='Ins', color='#3498db')
    ax.bar(x, v_dels, width, bottom=[s+i for s, i in zip(v_subs, v_ins)],
           label='Del', color='#f39c12')

    ax.set_title("VisG", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_visemes, fontsize=18)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=14)

    fig.suptitle(f"Per-Viseme Error Breakdown — {model_name}", fontsize=24)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    # plt.show()
    plt.savefig(savepath + f"{model_name}_baseline_vs_visg_two_subplots.pdf", dpi=300)
    plt.close()

def plot_per_viseme_total_error_rates_side_by_side(errors_baseline, errors_vsm, savepath, model_name="AV-HuBERT"):
    """
    Plot total per-viseme error rates (S+I+D) for Baseline vs VisG side-by-side.
    Fully compatible with analyze_viseme_errors_from_json() output.
    """

    # Union of all visemes
    visemes = sorted(set(errors_baseline['counts'].keys()) | 
                     set(errors_vsm['counts'].keys()))

    # Compute total error rates
    baseline_rates = []
    vsm_rates = []

    for v in visemes:
        # baseline
        b_count = errors_baseline['counts'].get(v, 0)
        b_err = (errors_baseline['sub'].get(v, 0) +
                 errors_baseline['ins'].get(v, 0) +
                 errors_baseline['del'].get(v, 0))
        b_rate = 100 * b_err / b_count if b_count > 0 else 0
        baseline_rates.append(b_rate)

        # vsm
        v_count = errors_vsm['counts'].get(v, 0)
        v_err = (errors_vsm['sub'].get(v, 0) +
                 errors_vsm['ins'].get(v, 0) +
                 errors_vsm['del'].get(v, 0))
        v_rate = 100 * v_err / v_count if v_count > 0 else 0
        vsm_rates.append(v_rate)

    # ---- Plotting ----
    x = np.arange(len(visemes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 8))

    bars1 = ax.bar(x - width/2, baseline_rates, width, label="AV-HuBERT")
    bars2 = ax.bar(x + width/2, vsm_rates, width, label="VisG AV-HuBERT")

    # Pad the y-axis
    ymax = max(max(baseline_rates), max(vsm_rates)) * 1.15
    ax.set_ylim(0, ymax)

    # ---- Annotate bars ----
    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10
            )

    annotate(bars1)
    annotate(bars2)

    # ---- Labels ----
    ax.set_xticks(x)
    ax.set_xticklabels(visemes, fontsize=14)
    ax.set_ylabel("Error Rate (%)", fontsize=18)
    ax.set_xlabel("Viseme Class", fontsize=18)
    ax.set_title(f"Total Per-Viseme Error Rates — Baseline vs VisG ({model_name})", fontsize=18)
    ax.legend(fontsize=14)

    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path + f'{model_name}_error_rates.pdf', dpi=300)


# Main execution
if __name__ == "__main__":
    conditions= []
    model = 'base'
    base_experiment_params = {'Name': 'Exp7'}
    large_experiment_params = {'Name': 'Exp5'}
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
        # Example for BASE model
        print("\n" + "="*70)
        print(f"ANALYZING {model} MODEL")
        print("="*70)
        
        if noise_type is not None:
            save_path = base_save_path + noise_path + '/'
            os.makedirs(save_path, exist_ok=True)
            baseline_file = f'/home/aristosp/models/av_hubert/{model}_decode/{noise_path}/hypo-244018.json'
            vsm_file = f'/home/aristosp/viseme_{model}_models/{experiment_params}/decode/{noise_path}/hypo-244018.json'
        else:
            baseline_file = f'/home/aristosp/models/av_hubert/{model}_decode/hypo-244018.json'
            vsm_file = f'/home/aristosp/viseme_{model}_models/{experiment_params}/decode/hypo-244018.json'
            save_path = base_save_path + 'clean/'
            os.makedirs(save_path, exist_ok=True)
        
        errors_dict = analyze_viseme_errors_from_json(baseline_file, vsm_file)
        
        print("\n Error rates per viseme...")
        
        plot_per_viseme_total_error_rates_side_by_side(
            errors_dict['baseline'],
            errors_dict['vsm'],
            save_path,
            model_name=f"AV-HuBERT {model}"
        )


        # Plot detailed error breakdown for each model
        print("\nPlotting baseline error breakdown...")
        plot_viseme_error_breakdown(errors_dict['baseline'], save_path, f'Baseline AV-HuBERT {model}')
        
        print("\nPlotting VisG error breakdown...")
        plot_viseme_error_breakdown(errors_dict['vsm'], save_path, f'VisG AV-HuBERT {model}')

        # plot_viseme_error_breakdown_side_by_side(
        #     errors_dict['baseline'], 
        #     errors_dict['vsm'], 
        #     save_path, 
        #     model_name=f"AV-HuBERT {model}"
        # )
        plot_viseme_error_breakdown_two_subplots( errors_dict['baseline'], 
            errors_dict['vsm'], 
            save_path, 
            model_name=f"AV-HuBERT {model}")
        
        
        # Plot improvements
        print("\nPlotting improvements...")
        plot_viseme_improvements(errors_dict['baseline'], save_path, errors_dict['vsm'])
