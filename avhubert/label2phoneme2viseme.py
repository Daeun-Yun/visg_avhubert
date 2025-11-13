from g2p_en import G2p
from sklearn.preprocessing import LabelEncoder
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

viseme_labels = [
    'P', 'T', 'K', 'CH', 'F', 'W', 'IY', 'EH', 'AA', 'AH', 'AO', 'UH', 'ER',  'S',
]
# Create a LabelEncoder object
label_encoder = LabelEncoder()
# Fit the encoder to the viseme labels
label_encoder.fit(viseme_labels)
g2p = G2p()
for file in ['train', 'valid', 'test']:
  with open("/home/aristosp/datasets/LRS3/433h_data/{}.wrd".format(file)) as f:
      file_lines = f.read().splitlines()
  new_lines = []
  for line in tqdm(file_lines):
    phonemic_transcription = g2p(line)
    phonemic_trans = [re.sub(r'\d', '', x) for x in phonemic_transcription]
    viseme_trans = [phoneme_to_viseme(x) for x in phonemic_trans]
    # ints = label_encoder.transform(viseme_trans)
    new_lines.append(viseme_trans)
  with open("/home/aristosp/datasets/LRS3/433h_data/{}.vsm".format(file), 'w') as f2:
    for string_list in tqdm(new_lines):
      formatted_string = format_phoneme_sequence(string_list)
      f2.write(formatted_string + "\n")
