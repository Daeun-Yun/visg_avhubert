import os
import random
import pandas as pd
import numpy as np
from scipy.io import wavfile

np.random.seed()

def load_wav_mono(path):
    """Load WAV and return (sr, mono ndarray)."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data

def make_noise_segment(noise, target_len):
    """Return noise segment of length target_len (tile or random crop)."""
    n = len(noise)
    if n == 0:
        return np.zeros(target_len, dtype=noise.dtype)
    if n < target_len:
        reps = int(np.ceil(target_len / n))
        return np.tile(noise, reps)[:target_len]
    if n == target_len:
        return noise.copy()
    start = random.randint(0, n - target_len)
    return noise[start:start + target_len]

def to_float32_norm(x):
    """Convert int16/float audio to float32 in -1..1 range."""
    x = x.astype(np.float32)
    if np.issubdtype(x.dtype, np.integer):
        # assume int16
        x = x / 32768.0
    return x

def add_noise_to_array(clean_wav, noise_wav, snr_db):
    """
    Mix noise_wav into clean_wav at desired SNR (dB).
    Returns int16 mixed audio.
    """
    clean_f = to_float32_norm(clean_wav)
    noise_f = to_float32_norm(noise_wav)
    if len(noise_f) != len(clean_f):
        noise_f = make_noise_segment(noise_f, len(clean_f))

    clean_rms = np.sqrt(np.mean(clean_f ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise_f ** 2) + 1e-12)

    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + 1e-12)
    noise_adj = noise_f * scale

    mixed = clean_f + noise_adj
    max_abs = np.max(np.abs(mixed))
    if max_abs > 1.0:
        mixed = mixed / max_abs * 0.999

    return (mixed * 32767.0).astype(np.int16)

def write_wav(path, sr, arr):
    """Write int16 numpy array to WAV, creating parent dir if needed."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    wavfile.write(path, sr, arr)




clean_fl = '/home/aristosp/datasets/LRS3/audio/test/0Fi83BHQsMA/00002.wav'
babble_noise_fl = '/home/aristosp/datasets/LRS3/noise/babble/noise.wav'

speech_tsv = pd.read_csv('/home/aristosp/datasets/LRS3/noise/speech/test.tsv', delimiter='\t')
speech_noise_fl = speech_tsv.iloc[:, 0].sample(1).iloc[0]

music_tsv = pd.read_csv('/home/aristosp/datasets/musan/tsv/music/test.tsv', delimiter='\t')
music_fl = music_tsv.iloc[:, 0].sample(1).iloc[0]

noise_tsv = pd.read_csv('/home/aristosp/datasets/musan/tsv/noise/test.tsv', delimiter='\t')
noise_fl = noise_tsv.iloc[:, 0].sample(1).iloc[0]

for noise_file, noise_type in zip([babble_noise_fl, speech_noise_fl, music_fl, noise_fl], ['babble', 'speech', 'music', 'noise']):
    for snr in [-10, -5, 0, 5, 10]:
        out = f'/home/aristosp/audio_noise/0Fi83BHQsMA_00002_{noise_type}_{snr}.wav'
        sr_c, clean = load_wav_mono(clean_fl)
        sr_n, noise = load_wav_mono(noise_file)
        if sr_c != sr_n:
            raise RuntimeError(f"Sample rates differ: clean {sr_c} Hz, noise {sr_n} Hz")
        mixed = add_noise_to_array(clean, noise, snr)
        write_wav(out, sr_c, mixed)
        print(f"Wrote {out}")