import torch
import torch.nn as nn


from utils.audio_processing import StftHandler

def _calc_func(seq_len):
    padding = 3 // 2
    dillation = 1
    kernel_size = 3
    stride = 2

    top = (seq_len + 2 * padding - dillation * (kernel_size - 1) -1)
    return torch.floor(top / stride + 1).to(dtype=torch.long)

class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()

class Residual(nn.Module):
    def __init__(self, n_channel, kernel_size, n_feats):
        super().__init__()
        self.layer_norm = CNNLayerNorm(n_feats//2+1)
        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=n_channel,out_channels=n_channel,
                                                   kernel_size=kernel_size, padding=1),
                                         self.layer_norm,
                                         nn.ReLU(),
                                         )
        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels=n_channel,out_channels=n_channel,
                                                   kernel_size=kernel_size, padding=1),
                                         self.layer_norm,
                                         nn.ReLU(),
                                         )
    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = residual + x
        return x

class ConvRes(nn.Module):
    def __init__(self, n_channels, kernel_size, stride, n_feats, out_dim):
        super().__init__()
        self.stft = StftHandler()
        self.conv = nn.Conv2d(1, n_channels, 3, stride=2, padding=3//2)
        self.res_conv = Residual(n_channels, kernel_size, n_feats)
        self.n_feats = n_feats
        self.in_dim = (n_feats//2+1) * n_channels
        self.fc = nn.Linear(self.in_dim, out_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, seq_len):
        x, seq_len = self.stft.wave_to_spec(x, seq_len)
        x = self.stft.spec_to_mag(x)

        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.res_conv(x)
        sizes = x.sizes()

        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])

        x = x.permute(2, 0, 1)
        x = self.fc(x)

        log = self.log_softmax(x)
        prob = self.softmax(x)

        seq_len = _calc_func(seq_len)
        return log, prob, seq_len


class ConvResRnn(nn.Module):
    def __init__(self, n_channels, kernel_size, hidden_size, num_layers, n_feats, out_dim):
        super().__init__()
        self.stft = StftHandler()
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, stride=2, padding=3 //2),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.res_conv = Residual(n_channels, kernel_size, n_feats)

        self.n_feats = n_feats
        self.in_dim = (n_feats//2+1) * n_channels
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=hidden_size,
                            bias=True, batch_first=True, bidirectional=True)

        self.fc_bi = nn.Linear(2 * hidden_size, out_dim)

        self.fc = nn.Linear(hidden_size, out_dim)

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, seq_len):
        x, seq_len = self.stft.wave_to_spec(x, seq_len)
        x = self.sftf.spec_to_max(x)

        x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.res_conv(x)
        sizes = x.size()

        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1,2)

        output, (h_n, c_n) = self.lstm_bi(x)
        output = self.fc_bi(output)

        output = output.transpose(1,2)

        log = self.log_softmax(output)
        prob = self.softmax(output)

        seq_len = _calc_func(seq_len)

        return log, prob, seq_len

