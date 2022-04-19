import os
import numpy as np
import pandas as pd
from utils import load_json, write_json
import argparse


def make_label_exp(symptoms):
    label = [0] * num_labels
    for symptom in symptoms:
        if sym2id.get(symptom) is not None:
            label[sym2id.get(symptom)] = 1
        else:
            raise Exception
    return label


def make_dataset_exp(samples, weight=5, mode='train', add_imp=False):
    instances, input_lens = [], []
    for pid, sample in samples.items():
        y = make_label_exp(sample['explicit_info']['Symptom']) if 'diagnosis' in sample else []
        if mode == 'train' and add_imp:
            for utt in sample['dialogue']:
                if 'symptom_norm' in utt and len(utt['symptom_norm']) > 0:
                    instances.append((utt['sentence'], y, pid, 1))
        input_lens.append(len(sample['self_report']))
        instances.append((sample['self_report'], y, pid, weight))
    return instances, input_lens


def make_label_imp(symptom_norm, symptom_type):
    assert len(symptom_norm) == len(symptom_type)
    label = [0] * (num_labels * 3)
    for sx_norm, sx_type in zip(symptom_norm, symptom_type):
        if sym2id.get(sx_norm) is not None:
            label[sym2id.get(sx_norm) * 3 + int(sx_type)] = 1
    return label


def make_dataset_imp(samples, num_contexts=1, weight=3, mode='train'):
    instances, input_lens = [], []
    n1, n2 = 1, 1
    for pid, sample in samples.items():
        diag = sample['dialogue']
        num_utts = len(diag)
        for i in range(num_utts):
            x = []
            start = max(i - num_contexts, 0)
            end = min(i + num_contexts + 1, num_utts)
            for j in range(start, end):
                x.append(diag[j]['speaker'] + '：' + diag[j]['sentence'])
            x = '\t'.join(x)
            y = [] if mode == 'test' else make_label_imp(diag[i]['symptom_norm'], diag[i]['symptom_type'])
            _weight = weight if 'symptom_norm' in diag[i] and len(diag[i]['symptom_norm']) > 0 else 1
            if 'symptom_norm' in diag[i] and len(diag[i]['symptom_norm']) > 0:
                n1 += 1
            else:
                n2 += 1
            input_lens.append(len(x))
            instances.append((x, y, pid, _weight))
    return instances, input_lens, n2 / n1


if __name__ == '__main__':

    data_dir = '../../../dataset'
    # data_dir = 'dataset'

    train_set = load_json(os.path.join(data_dir, 'train.json'))
    dev_set = load_json(os.path.join(data_dir, 'dev.json'))
    test_set = load_json(os.path.join(data_dir, 'test_input.json'))

    # load normalized symptom
    sym2id = {value: key for key, value in pd.read_csv(os.path.join(data_dir, 'symptom_norm.csv'))['norm'].items()}
    num_labels = len(sym2id)

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default='exp', required=True, type=str)
    args = parser.parse_args()

    saved_path = 'data'
    suffix = args.target
    os.makedirs(os.path.join(saved_path, suffix), exist_ok=True)

    if args.target == 'exp':
        train, lens = make_dataset_exp(train_set, mode='train')
        dev, _ = make_dataset_exp(dev_set, mode='dev')
        test, _ = make_dataset_exp(test_set, mode='test')
        print('train/dev/test size: {}/{}/{}'.format(len(train), len(dev), len(test)))
        print('In train set, 95 percent of input length is shorter than {}'.format(np.round(np.quantile(lens, 0.95), 4)))
    else:
        train, lens, ratio = make_dataset_imp(train_set, mode='train')
        dev, _, _ = make_dataset_imp(dev_set, mode='dev')
        test, _, _ = make_dataset_imp(test_set, mode='test')
        print('train/dev/test size: {}/{}/{}'.format(len(train), len(dev), len(test)))
        print('In train set, ratio of utterance containing symptoms are {}'.format(np.round(ratio, 4)))
        print('In train set, 95 percent of input length is shorter than {}'.format(np.round(np.quantile(lens, 0.95), 4)))

    write_json(train, os.path.join(saved_path, suffix, 'train.json'))
    write_json(dev, os.path.join(saved_path, suffix, 'dev.json'))
    write_json(test, os.path.join(saved_path, suffix, 'test.json'))
