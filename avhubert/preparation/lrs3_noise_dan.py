import os
import wave
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generating speech noise from LRS3 pretrain',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    args = parser.parse_args()

    sample_rate = 16_000
    min_len = 20 * sample_rate  # 320000 samples

    pretrain_dir = os.path.join(args.lrs3, 'pretrain')
    speech_tsv_dir = os.path.join(args.lrs3, 'noise', 'speech')
    speech_wav_dir = os.path.join(args.lrs3, 'noise', 'speech', 'wav')
    os.makedirs(speech_tsv_dir, exist_ok=True)
    os.makedirs(speech_wav_dir, exist_ok=True)

    print(f'Scanning pretrain directory: {pretrain_dir}')
    wav_fns = []
    for vid_id in sorted(os.listdir(pretrain_dir)):
        vid_path = os.path.join(pretrain_dir, vid_id)
        if not os.path.isdir(vid_path):
            continue
        for f in sorted(os.listdir(vid_path)):
            if not f.endswith('.wav'):
                continue
            wav_path = os.path.join(vid_path, f)
            try:
                with wave.open(wav_path, 'r') as wf:
                    if wf.getnframes() > min_len:
                        wav_fns.append(wav_path)
            except Exception:
                pass

    print(f'# speech noise candidates (>= 20s): {len(wav_fns)}')
    print(f'Generating speech noise -> {speech_tsv_dir}')

    noise_fns = []
    for wav_fn in tqdm(wav_fns):
        sr, wav_data = wavfile.read(wav_fn)
        wav_data = wav_data[:min_len]
        filename = '_'.join(wav_fn.split('/')[-2:])
        noise_fn = os.path.join(speech_wav_dir, filename)
        noise_fns.append(noise_fn)
        wavfile.write(noise_fn, sr, wav_data.astype(np.int16))

    num_train = int(len(noise_fns) * 0.6)
    num_valid = int(len(noise_fns) * 0.2)
    num_test  = len(noise_fns) - num_train - num_valid

    prev = 0
    for split, num_x in [('train', num_train), ('valid', num_valid), ('test', num_test)]:
        split_fns = [os.path.abspath(fn) for fn in noise_fns[prev: prev + num_x]]
        tsv_path = os.path.join(speech_tsv_dir, f'{split}.tsv')
        with open(tsv_path, 'w') as fo:
            fo.write('\n'.join(split_fns) + '\n')
        prev += num_x
        print(f'{split}.tsv: {len(split_fns)} files -> {tsv_path}')


if __name__ == '__main__':
    main()
