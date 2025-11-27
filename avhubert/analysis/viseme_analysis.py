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
    plt.savefig(save_path + f'{model_name}_per_viseme_errors2.pdf', dpi=300)
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
    ax.set_xlabel('Viseme Class', fontsize=16)
    ax.set_ylabel('Error Rate Change (VisG - Baseline) (%)', fontsize=16)
    ax.set_title('Per-Viseme Error Rate Change (VisG - Baseline)', fontsize=16)
    ax.grid(axis='y', alpha=0.3)
    # plt.xticks(rotation=45)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(save_path + f'{model}_viseme_change_rate2.pdf', dpi=300)
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
        
        # Plot detailed error breakdown for each model
        print("\nPlotting baseline error breakdown...")
        plot_viseme_error_breakdown(errors_dict['baseline'], save_path, f'Baseline AV-HuBERT {model}')
        
        print("\nPlotting VisG error breakdown...")
        plot_viseme_error_breakdown(errors_dict['vsm'], save_path, f'VisG AV-HuBERT {model}')
        
        # Plot improvements
        print("\nPlotting improvements...")
        plot_viseme_improvements(errors_dict['baseline'], save_path, errors_dict['vsm'])
