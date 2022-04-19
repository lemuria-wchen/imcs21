import os
import numpy as np
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from tqdm import tqdm

from utils import CustomDataset, collate_fn, BERTClass, load_json

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cuda_num", default='0', required=True, type=str)
args = parser.parse_args()


prefix = 'data'
model_prefix = 'saved'
os.makedirs(model_prefix, exist_ok=True)

train = load_json(os.path.join(prefix, 'train.json'))
dev = load_json(os.path.join(prefix, 'dev.json'))

device = 'cuda:{}'.format(args.cuda_num) if torch.cuda.is_available() else 'cpu'

num_digits = 4

enc_dim = 768
num_bio = 4
num_sn = 331
num_sl = 3
model_name = 'bert-base-chinese'

TRAIN_BATCH_SIZE = 128
DEV_BATCH_SIZE = 128
LEARNING_RATE = 1e-5
EPOCHS = 50

MAX_LEN = 128

tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_set = CustomDataset(train, tokenizer, MAX_LEN)
dev_set = CustomDataset(dev, tokenizer, MAX_LEN)

weights = [sample['weight'] for sample in train]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'num_workers': 0}
dev_params = {'batch_size': DEV_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}

train_loader = DataLoader(train_set, sampler=sampler, collate_fn=collate_fn, **train_params)
# train_loader = DataLoader(train_set, collate_fn=collate_fn, **train_params)
dev_loader = DataLoader(dev_set, collate_fn=collate_fn, **dev_params)


model = BERTClass(model_name, enc_dim, num_bio, num_sn, num_sl)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

criterion_bio = torch.nn.CrossEntropyLoss(ignore_index=0)
criterion_sn = torch.nn.CrossEntropyLoss()
criterion_sl = torch.nn.CrossEntropyLoss()


def train_epoch(data_loader, epoch, mode='train'):
    loss_list, bio_loss_list, sn_loss_list, sl_loss_list = [], [], [], []
    for _, batch in tqdm(enumerate(data_loader)):
        outputs = model(batch['ids'].to(device), batch['mask'].to(device), batch['token_type_ids'].to(device))
        features, sn_labels, sl_labels = [], [], []
        for i, chunks in enumerate(batch['chunks']):
            for chunk in chunks:
                features.append(torch.mean(outputs[chunk[0]: chunk[1] + 1, i, :], dim=0))
                sn_labels.append(chunk[2])
                sl_labels.append(chunk[3])
        sn_labels = torch.tensor(sn_labels, dtype=torch.long, device=device)
        sl_labels = torch.tensor(sl_labels, dtype=torch.long, device=device)
        bio_outputs = model.fc_bio(outputs)
        sn_outputs = model.fc_sn(torch.stack(features))
        sl_outputs = model.fc_sl(torch.stack(features))
        bio_loss = criterion_bio(bio_outputs.reshape(-1, num_bio), batch['bio_ids'].to(device).reshape(-1))
        sn_loss = criterion_sn(sn_outputs, sn_labels)
        sl_loss = criterion_sl(sl_outputs, sl_labels)
        if epoch < 5:
            loss = bio_loss
        elif epoch < 10:
            loss = bio_loss + sn_loss
        else:
            loss = bio_loss + sn_loss + sl_loss
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
        bio_loss_list.append(bio_loss.item())
        sn_loss_list.append(sn_loss.item())
        sl_loss_list.append(sl_loss.item())
    print('{} epoch: {},  avg loss: {}, avg bio_loss: {}, avg sn_loss: {}, avg sl_loss: {}'.format(
        mode, epoch + 1, np.round(np.mean(loss_list), num_digits), np.round(np.mean(bio_loss_list), num_digits),
        np.round(np.mean(sn_loss_list), num_digits), np.round(np.mean(sl_loss_list), num_digits),
    ))
    return np.mean(loss_list)


print('total steps: {}'.format(len(train_loader) * EPOCHS))

best_loss = 1e4

for _epoch in range(EPOCHS):
    # train
    model.train()
    train_epoch(train_loader, _epoch, mode='train')
    # evaluation
    model.eval()
    with torch.no_grad():
        dev_loss = train_epoch(dev_loader, _epoch, mode='dev')

    if _epoch > 10 and dev_loss < best_loss:
        print('saving model to {}'.format(os.path.join(model_prefix, 'model.pkl')))
        torch.save(model, os.path.join(model_prefix, 'model.pkl'))
        best_loss = dev_loss
