import numpy as np
import torch
from utils.text_processing import CharTokenizer

tokenizer = CharTokenizer()

def remove_duplicates(s):
    chars = []
    prev = None

    for c in s:
        if prev !=c:
            chars.append(c)
            prev = c
    return ''.join(chars)

class GreedyDecoder:
    def __init__(self):
        super().__init__()

    def __call__(self, matrix, labels):
        #Get max classes
        arg_maxes = torch.argmax(matrix, axis=-1)

        decodes = []
        targets = []

        for i in range(arg_maxes.shape[1]):
            max_id_units = arg_maxes[:,i]

            decode = tokenizer.ids_to_text(max_id_units.detach().cpu().numpy())
            decode = remove_duplicates(decode)
            decode = decode.replace("_", "")

            decodes.append(decode)

            target = tokenizer.ids_to_text((labels[i].detach().cpu().numpy()))
            target = remove_duplicates(target)
            target = target.replace("_", "")
            targets.append(target)

        return decodes, targets

def main():
    matrix = np.loadtxt('../test_path.txt')
    matrix = torch.Tensor(matrix).unsqueeze(0).transpose(1, 2)

    labels_indices = tokenizer.text_to_ids('there seems')
    len_sequence = len(labels_indices)

    sequence = tokenizer.add_blank(labels_indices)
    sequence = torch.Tensor(sequence).unsqueeze(0)






