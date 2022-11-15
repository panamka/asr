import torch
import torch.nn as nn
import pandas as pd

import glob
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Sampler

from utils.text_processing import CharTokenizer
from utils.audio_processing import StftHandler, compute_log_mel_spectrogram, SpectrogramTransform

path = ''
text_files_utts = glob.glob(f'{path}/**/*.txt', recursive=True)
audio_files_utts = glob.glob(f'{path}/**/*.txt', recursive=True)


def read_text(file):
    src_file = '-'.join(file.split('-')[:-1]) + '.trans.txt'
    idx = file.split('/')[-1].split('.')[0]
    with open(src_file, 'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ', 1)[1]


def collate_fn(data):
    data = [{
        'audio': audio,
        'utt_id': utt_id} for audio, utt_id in data
    ]

    audios = torch.nn.utils.rnn.pad_sequence(
        [obj['audio'] for obj in data], batch_first=True, padding_value=0.0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [obj['utt_id'] for obj in data], batch_first=True, padding_value=0.0
    )

    input_lengths = torch.tensor([obj['audio'].shape[0] for obj in data])
    label_lengths = torch.tensor([obj['utt_id'].shape[0] for obj in data])
    return audios, labels, input_lengths, label_lengths


class AsrDatasetV3(Dataset):
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.tokenizer = CharTokenizer()
        self.masking = SpectrogramTransform()

    def __len__(self):
        return len(self.data)

    def extractor(self, idx, tokenizer):
        signal_path = self.data.iloc[idx]['path']
        signal, sr = sf.read(signal_path, dtype='float32')

        signal = torch.from_numpy(signal)
        spec = compute_log_mel_spectrogram(signal)
        text_utt = self.data.iloc[idx]['text']

        text_utt = tokenizer.clean_text(text_utt)
        utt_ids = tokenizer.text_to_ids(text_utt)
        utt_ids = tokenizer.add_blanc(utt_ids)
        utt_ids = torch.IntTensor(utt_ids)

        return spec, utt_ids

    def __getitem__(self, idx):
        spec, utt_ids = self.extractor(idx, self.tokenizer)
        return spec, utt_ids


class LibriDatasetSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__(None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.cur_epoch = 0

        # SequentialSampler will return indices in range 0...size-1
        # BatchSampler wraps another sampler to yeild a mini-batch of indices
        # batches - list of batch indexes: at each batch  - index of dataset items

        self.batches = list(BatchSampler(SequentialSampler(self.dataset), batch_size=self.batch_size, drop_last=False))

    def __len__(self):
        return len(self.dataset)

    def __iter(self):
        if self.cur_epoch == 0:
            # select sequentional batches for the first epoch
            for batch_idx in SequentialSampler(self.batches):
                for idx in self.batches[batch_idx]:
                    yield idx
        else:
            for batch_idx in RandomSampler(self.batches):
                for idx in self.batches[batch_idx]:
                    yield idx
        self.cur_epoch += 1


def worker_init_fn():
    np.random.seed(torch.initial_seed() % (2 ** 32 - 1))


def main():
    np.random.seed(77)
    torch.manual_seed(42)

    root_train = 'libri_data_train.csv'
    train_dataset = AsrDatasetV3(root_train)

    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=LibriDatasetSampler(train_dataset, batch_size=batch_size),
                              collate_fn=collate_fn, shuffle=False, pin_memory=True,
                              num_workers=8, worker_init_fn=worker_init_fn())

    spec, label, input_length, label_length = next(iter(train_loader))


if __name__ == "__main__":
    main()
