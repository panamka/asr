import torch
import torch.nn as nn

from utils.audio_processing import StftHandler

def _calc_func(seq_len, axis):
    padding = 0
    dillation = 1
    kernel_size_params = [11, 21]
    stride_params = [1, 1]

    kernel_size = kernel_size_params[axis]
    stride = stride_params[axis]

    top = (seq_len + 2 * padding - dillation * (kernel_size -1) - 1)
    return torch.floor(top / stride + 1).to(dtype=torch.long)


class DeepSpeech(nn.Module):
    def __init__(self, n_feats, n_channels, hidden_size, num_layers, out_dim):
        super(DeepSpeech, self).__init__()
        self.stft = StftHandler()
        self.n_feats = n_feats
        self.conv_block = nn.Sequential(nn.Conv2d(1, out_channels=n_channels, kernel_size=(11,21), stride=(1,1)),
                                        nn.BatchNorm2d(num_features=n_channels),
                                        nn.ReLU(inplace=True),

                                        nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(11, 21), stride=(1, 1)),
                                        nn.BatchNorm2d(num_features=n_channels),
                                        nn.ReLU(inplace=True)
                                        )
        self.in_dim = _calc_func((torch.tensor(n_feats), 0), 0).item() * n_channels

        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=hidden_size,
                            bidirectional=False, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, out_dim)
        #self.fc_bi = nn.Linear(hidden_size * 2, out_dim)

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, seq_len):
        """


        :param x: signal tensor with shape (batch_size, num_timesteps)
        :param seq_len: 1D tensor with shape (batch_size
        :return:

                3D tensor with shape (new_num_timessteps, batch_size, vocab_size)
                1D tensor with shape (batch_size)
        """
        #[B, features, time]

        x, seq_len = self.stft.wave_to_spec(x, seq_len)
        x = self.stft.spec_to_mag(x)
        #[Batch, 1, n_fft, time]
        x = x.unsqueeze(1)

        x = self.conv_block(x)
        sizes = x.size()

        #[Batch. n_channels, freatures, num_timesteps] -> [Batch, features, num_timesteps]
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])

        #[Batch, features, num_timesteps] ->  [Batch, num_timesteps, features]
        x = x.transpose(1, 2)
        output, (h, c) = self.lstm(x)
        #output, (h, c) = self.bi_lstm(x)

        output = self.fc(output)
        #output = self.fc_bi(output)

        #[Batch, num_timesteps, features] -> [Batch, features, num_timesteps]
        output = output.transpose(0, 1)

        log = self.log_softmax(output)
        prob = self.softmax(output)

        new_seq_len = _calc_func(_calc_func(torch.tensor(seq_len), 1), 1)

        return log, prob, new_seq_len



