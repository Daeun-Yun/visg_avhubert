from g2p_en import G2p
from tqdm import tqdm
import re

def format_phoneme_sequence(phonemes):
    # Join phonemes and replace <UNK> with spaces
    formatted = ' '.join(phonemes).replace(' S ', ' ')
    # Remove any trailing/leading spaces
    return formatted.strip()

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



def label_map(path):
  g2p = G2p()

  for file in ['train', 'valid', 'test']:
    print("Processing file: {}.wrd".format(file))
    with open(path + "{}.wrd".format(file)) as f:
        file_lines = f.read().splitlines()
    new_lines = []
    for line in tqdm(file_lines):
      phonemic_transcription = g2p(line)
      phonemic_trans = [re.sub(r'\d', '', x) for x in phonemic_transcription]
      viseme_trans = [phoneme_to_viseme(x) for x in phonemic_trans]
      # ints = label_encoder.transform(viseme_trans)
      new_lines.append(viseme_trans)
    with open(path + "{}.vsm".format(file), 'w') as f2:
      for string_list in tqdm(new_lines):
        formatted_string = format_phoneme_sequence(string_list)
        f2.write(formatted_string + "\n")
    print("Finished processing file: {}.wrd. Saved viseme file at {}".format(file, path))

def create_dict(path):
    dict_path = path + "dict.vsm.txt"
    viseme_labels = [
    'P', 'T', 'K', 'CH', 'F', 'W', 'IY', 'EH', 'AA', 'AH', 'AO', 'UH', 'ER',  'S',]
    with open(dict_path, 'w') as f:
        for label in viseme_labels:
            f.write(f"{label} 1\n")
    print(f"Viseme dictionary saved at {dict_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate viseme labels from word transcripts."
    )
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the directory containing .wrd files.")
    args = parser.parse_args()
    path = args.path
    label_map(path)
