import numpy as np
import torch
import torch.nn as nn

import torchaudio
import soundfile as sf
import glob

class StftHandler(nn.Module):
    def __init__(self, n_fft=512, win_len=480, hop_len=160):
        super().__init__()
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        window = torch.hann_window(self.win_len)
        self.register_buffer('stft_window', window)

    def wave_to_spec(self, x, seq_len):
        sps = torch.stft(
            x, n_fft=self.n_fft, win_length=self.win_len,
            hop_length=self.hop_len,
            window=self.stft_window
        )
        seq_len = ((seq_len + 2 * self.hop_len - self.win_len) // self.hop_len + 1).to(dtype=torch.long)
        return sps, seq_len

    def wave_to_spec_log(self, x):
        sps = torch.stft(
            x, n_fft=self.n_fft, win_length=self.win_len,
            hop_length=self.hop_len,
            window=self.stft_window,
            return_complex=True
        )
        return np.log1p(sps)


    def spec_to_mag(self, sps):
        mag = sps.abs() + 1e-8
        return mag

    def spec_to_wave(self, spec, lenght):
        wavform = torch.istft(
            spec, n_fft=self.n_fft, win_length=self.win_len,
            hop_length=self.hop_len,
            window=self.stft_window,
            lenght=lenght
        )
        return wavform

class SpectrogramTransform(torch.nn.Module):
    def __init__(self, freq_param=10, time_param=10):
        super(SpectrogramTransform, self).__init__()

        self.freq_param = freq_param
        self.time_param = time_param

    def forward(self, spectrogram):
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_param)
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param=self.time_param)
        return time_masking(freq_masking(spectrogram))

def compute_log_mel_spectrogram(
        audio, sequence_lengths,
        sample_rate=16000, window_size=0.02, window_step=0.01,
        f_min=20, f_max=3800, n_mels=64,
        window_fn=torch.hamming_window,
        power=1.0, eps=1e-6, spectrogram_transfrom=None):

    spectrogram_transfrom = (lambda x: x) if spectrogram_transfrom is None else spectrogram_transfrom

    win_lenth = int(window_size * sample_rate)
    hop_lenth = int(window_step * sample_rate)

    mel_transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, win_length=win_lenth, hop_length=hop_lenth,
        n_fft=win_lenth, f_min=f_min, pad=0, n_mels=n_mels, power=power,
        window_fn=window_fn).to(device=audio.device)

    log_mel_spectrogram = torch.log(mel_transformer(audio) + eps)
    log_mel_spectrogram = spectrogram_transfrom(log_mel_spectrogram)
    seq_len = ((sequence_lengths + 2 * hop_lenth - win_lenth) // hop_lenth + 1).to(dtype=torch.long)

    return log_mel_spectrogram, seq_len

def main():
    print('tyt')

    root_uts = ''
    audio_files_utts = glob.glob()
    signal_path, sr = sf.read()


