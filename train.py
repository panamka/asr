import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel as DP
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import os
from shutil import rmtree
import math
from TensoboardLogger import TensorboardLogger

from dataset import AsrDatasetV3, DataLoader, collate_fn
from utils.text_processing import CharTokenizer
from utils.decoder import GreedyDecoder
from utils.metrics import CER, WER

from model.rnn_simple.conv import ConvResRnn
from model.deepspeech.model import DeepSpeach


tokenizer = CharTokenizer()
blank_id = tokenizer.text_to_ids('_')[0]

def to_numpy(x):
    return x.detach().cpu().numpy()

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32 -1))


def train_epoch(model, loader, optimizer, scheduler, criterion,  max_norm_grad, device):
    log = {}
    losses = []
    for audio, label, input_length, target_length in tqdm(loader):
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
            model.parametrs(), max_norm_grad).item()
        log['grad_norm'] = grad_norm
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return {'train_loss' : np.mean(losses)} | log


@torch.inference_mode()
def val_epoch(model, loader, criterion, decoder, metric_dict, device):
    losses = []
    log = {}
    for audio, label, input_length, target_length in tqdm(loader):
        audio = audio.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)
        input_length = input_length.to(device, dtype=torch.int)
        target_length = target_length.to(device, dtype=torch.int)

        #output shape = TxBxN_classes
        log_prob, prob, input_length = model(audio, input_length)
        loss = criterion(log_prob, label, input_length, target_length)
        losses.append(loss.item())

        decoded_seq, decoded_target = decoder(prob, label)


        for name, metrics in metric_dict.items():
            values = []
            for i in range(len(decoded_seq)):
                values_tmp = metrics(decoded_target[i], decoded_seq[i])
                values.append(values_tmp)

            log[name] = np.mean(values)

    return {'val_loss' : np.mean(losses)} | log


def build_model():
    n_feats = 257
    n_channels = 32
    kernel_size = 3
    #stride = 2
    out_dim = blank_id + 1

    hidden_size = 512
    num_layers = 2

    #model = DeepSpeach(n_feats, n_channels, hidden_size, num_layers, out_dim)
    model = ConvResRnn(n_channels, kernel_size, hidden_size, num_layers, n_feats, out_dim)
    return model

def build_loader(batch_size):
    root_train = '/home/Datasets/LibriSpeech/train-clean-360'
    root_test = '/home/Datasets/LibriSpeech/train-clean-100'

    train_dataset = AsrDatasetV3(root_train)
    test_dataset = AsrDatasetV3(root_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True,
        num_workers=8, worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True,
        num_workers=8, worker_init_fn=worker_init_fn
    )
    return train_loader, test_loader

def main():
    logdir = 'tensorboard_logging'
    model_name = 'asr_conv_rnn'

    logger_tb = TensorboardLogger(os.path.join(logdir,model_name))

    train_history = defaultdict(list)
    val_history = defaultdict(list)

    max_grad_norm = 5
    cuda_visible_devices = '0,1,2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    devices = [1]
    device = 'cuda:1'
    n_epochs = 600
    batch_size = 32 * len(devices)
    lr =1e-4
    start_epoch = 0
    best_loss = float('inf')


    decoder = GreedyDecoder()
    metric_dict = {
                'cer': CER(),
                'wer': WER(),
                }

    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=False)

    model = build_model()
    model.to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr)
    lr_scheduler = build_sch(optimizer)

    train_loader, test_loader = build_loader(batch_size)

    saveroot = 'path'
    if os.path.exists(saveroot):
        ans = input(f'{saveroot} already exists. Do you want to rewrite it? Y/n: ').lower()
        if ans == 'y':
            rmtree(saveroot)
            os.makedirs(saveroot)
            if os.path.exists(os.path.join(logdir, model_name)):
                print('del TB folder')
                rmtree(os.path.join(logdir, model_name))
                print('create new TB folder')
                logger_tb = TensorboardLogger(os.path.join(logdir, model_name))
        else:
            print('Continue to train')
            checkpoint_path = os.path.join(saveroot, 'last_snapshot.tar')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['opt'])
            lr_scheduler.load_state_dict(checkpoint['sch'])
            train_history = checkpoint['train_history']
            val_history = checkpoint['val_history']
            best_loss = min(val_history['val_loss'])
            start_epoch = len(train_history['train_loss'])
            print(f'Train loss: {train_history["train_loss"][-1]}')
            print(f'Val loss: {val_history["val_loss"][-1]}')
            model.train()
    else:
        os.makedirs(saveroot)


    for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
        train_dict = train_epoch(model, train_loader, optimizer, lr_scheduler, criterion,
                                 max_grad_norm, device)
        val_dict = val_epoch(model, test_loader, criterion, decoder, metric_dict, device)
        print('train metrics')
        for key, value in train_dict.items():
            logger_tb.log(epoch, value, key, 'train')
            print(f'{key}: {value}')
            train_history[key].append(value)
        print('val metrics')
        for key, value in val_dict.items():
            logger_tb.log(epoch, value, key, 'val')
            print(f'{key}: {value}')
            val_history[key].append(value)
        snapshot = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'sch': lr_scheduler.state_dict(),
            'train_history': train_history,
            'val_history': val_history,
        }

        last_snapshot_path = os.path.join(saveroot, 'last_snapshot.tar')
        torch.save(snapshot, last_snapshot_path)

        cur_loss = val_dict['val_loss']
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_snapshot_path = os.path.join(saveroot, 'best_snapshot.tar')
            torch.save(snapshot, best_snapshot_path)

start_lr = 1e-5
main_lr = 1e-5

# main_lr = 4e-3

def build_sch(optimizer):
    def schedule(step):
        warm_up_steps = 1000

        if step >= warm_up_steps:
            return 1

        start_mult = start_lr / main_lr
        fraction = step / warm_up_steps * math.pi
        stair = (1 - math.cos(fraction)) / 2
        stair = start_mult + (1 - start_mult) * stair

        return stair

    sch = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=schedule
    )
    return sch

if __name__ == '__main__':
        main()