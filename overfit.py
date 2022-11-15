from TensoboardLogger import TensorboardLogger
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import os
from shutil import rmtree
from torch.nn.utils import clip_grad_norm_
# import soundfile as sf
# import matplotlib.pyplot as plt

from dataset import AsrDatasetV3, DataLoader, collate_fn
from utils.text_processing import CharTokenizer

from utils.decoder import GreedyDecoder
from utils.metrics import CER, WER


from model.rnn_simple.conv import ConvRes, ConvResRnn
from model.deepspeech.model import DeepSpeech


tokenizer = CharTokenizer()
blank_id = tokenizer.text_to_ids('_')[0]

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32 - 1))

def to_numpy(x):
    return x.detach().cpu().numpy()




def train_epoch(model, audio, label, input_length, target_length, optimizer, decoder, metric_dict, criterion, max_grad_norm, device):
    log = {}
    losses = []
    audio = audio.to(device, dtype=torch.float)
    label = label.to(device, dtype=torch.float)
    input_length = input_length.to(device, dtype=torch.int)
    target_length = target_length.to(device, dtype=torch.int)

    optimizer.zero_grad()
    for param_group in optimizer.param_groups:
        log['lr_train'] = param_group['lr']

    log_prob, prob, input_length = model(audio, input_length)

    loss = criterion(log_prob, label, input_length, target_length)
    loss.backward()
    grad_norm = clip_grad_norm_(
        model.parameters(), max_grad_norm).item()
    log['grad_norm'] = grad_norm
    optimizer.step()
    decoded_seq, decoded_target = decoder(prob, label)
    print(decoded_seq)
    print(decoded_target)
    losses.append(loss.item())
    for name, metric in metric_dict.items():
        values = []
        for i in range(len(decoded_seq)):
            values_tmp = metric(decoded_target[i], decoded_seq[i])
            values.append(values_tmp)

        log[name] = np.mean(values)
        print(name, np.mean(values))

    return {'train_loss': np.mean(losses)} | log




def main():
    device = 'cuda:7'
    logdir = 'tensorboard_logging' # this is root for tensorboard
    model_name = 'conv_rnn_overfit' # this is name of results we want to track
    logger_tb = TensorboardLogger(os.path.join(logdir, model_name))
    decoder = GreedyDecoder()
    metric_dict = {
    'cer': CER(),
    'wer': WER(),
    }

    tokenizer = CharTokenizer()
    blank_id = tokenizer.text_to_ids('_')[0]
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=False)

    kernel_size = 3
    n_channels = 32
    n_feats = 257
    stride = 2
    out_dim = blank_id + 1
    hidden_size = 512
    num_layers = 2

    model = DeepSpeech(n_feats, n_channels, hidden_size, num_layers, out_dim)
    # model = ConvRes(n_channels, kernel_size, stride, n_feats, out_dim)
    # model = ConvResRnn(n_channels, kernel_size, hidden_size, num_layers, n_feats, out_dim)
    model.to(device)
    train_history = defaultdict(list)
    max_grad_norm = 5
    device = 'cuda:7'
    n_epochs = 700
    batch_size = 2
    lr = 5e-4
    start_epoch = 0



    optimizer = torch.optim.Adam(model.parameters(), lr)


    #Check if folder exists
    saveroot = '/home/eva/TrainResults/conv_rnn_overfit'
    if os.path.exists(saveroot):
        ans = input(f'{saveroot} is already existed. Do you want to rewrite it? Y/n: ').lower()
        if ans == 'y':
            rmtree(saveroot)
            os.makedirs(saveroot)
            if os.path.exists(os.path.join(logdir, model_name)):
                print('del TB folder')
                rmtree(os.path.join(logdir, model_name))
                print('create new TB folder')
                logger_tb = TensorboardLogger(os.path.join(logdir, model_name))
        else:
            # exit(1)
            print('Continue to train')
            checkpoint_path = os.path.join(saveroot, 'last_snapshot.tar')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['opt'])
            train_history = checkpoint['train_history']
            start_epoch = len(train_history['train_loss'])
            print(f'Train loss: {train_history["train_loss"][-1]}')
            model.train()
    else:
        os.makedirs(saveroot)


    root_train = '/home/Datasets/LibriSpeechWav/train-clean-360'
    train_dataset = AsrDatasetV3(root_train)


    # train_loader = DataLoader(
    # train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True,
    # num_workers=8, worker_init_fn=worker_init_fn
    # )

    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
    # spec, label, input_length, target_length = next(iter(train_loader))

    # root_train = 'libri_data_train.csv'
    # root_test = 'libri_data_test.csv'
    #
    # train_dataset = AsrDatasetV3(root_train)
    #
    # train_loader = DataLoader(
    # train_dataset, batch_size=batch_size, sampler = LibriDatasetSampler(train_dataset, batch_size=batch_size),
    # collate_fn=collate_fn, shuffle=False, pin_memory=True,
    # num_workers=8, worker_init_fn=worker_init_fn
    # )

    audio, label, input_length, target_length = next(iter(train_loader))

    save_audio_path = '/home/eva/tests_output/ars_lstm_overfit'
    # os.makedirs(save_audio_path)
    if os.path.exists(save_audio_path):
        rmtree(save_audio_path)
        os.makedirs(save_audio_path)
    else:
        os.makedirs(save_audio_path)



    for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
        train_dict = train_epoch(model, audio, label, input_length, target_length, optimizer, decoder, metric_dict, criterion, max_grad_norm, device)

        print('train metrics')
        for key, value in train_dict.items():
            logger_tb.log(epoch, value, key, 'train')
            print(f'{key}: {value}')
            train_history[key].append(value)





if __name__ == '__main__':
    main()