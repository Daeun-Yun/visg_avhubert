import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

lrs2 = '/home/aristosp/datasets/lrs2_rf/lrs2/lrs2_video_seg16s/'


# Input TSV files to read
tsv_fns = [
    os.path.join(lrs2, 'train.tsv'),
    os.path.join(lrs2, 'valid.tsv'),
    os.path.join(lrs2, 'test.tsv')
]

# Output TSV files (will overwrite originals with normalized versions)
output_tsvs = tsv_fns

sample_rate = 16_000
min_len = 1 * sample_rate
print("Normalizing speech noise wav files...")

for tsv_fn in tsv_fns:
    if not os.path.exists(tsv_fn):
        print(f"Warning: {tsv_fn} not found, skipping...")
        continue
    
    print(f"\nProcessing {tsv_fn}")
    
    # Read wav file paths from TSV
    lns = open(tsv_fn).readlines()[1:]
    wav_fns = [(ln.strip().split('\t')[2], int(ln.strip().split('\t')[-1])) for ln in lns]
    wav_fns = list(filter(lambda x: x[1] > min_len, wav_fns))
    wav_fns = [x[0] for x in wav_fns]
    
    print(f"Found {len(wav_fns)} wav files")
    
    # Process each wav file
    for wav_fn in tqdm(wav_fns):
        if not os.path.exists(wav_fn):
            print(f"Warning: {wav_fn} not found, skipping...")
            continue
        
        # Read wav file
        sr, wav_data = wavfile.read(wav_fn)
        
        # Normalize and convert to int16
        if wav_data.dtype in [np.float32, np.float64]:
            # For float data: normalize to [-1, 1] then scale to int16
            max_val = np.abs(wav_data).max()
            if max_val > 0:
                wav_data = (wav_data / max_val) * 32767.0
            wav_data = wav_data.astype(np.int16)
        else:
            # For int data: normalize then scale
            wav_data_float = wav_data.astype(np.float32)
            max_val = np.abs(wav_data_float).max()
            if max_val > 0:
                wav_data = (wav_data_float / max_val * 32767.0).astype(np.int16)
            else:
                wav_data = wav_data.astype(np.int16)
        
        # Write back normalized wav file
        wavfile.write(wav_fn, sr, wav_data)

print("\nNormalization complete!")
