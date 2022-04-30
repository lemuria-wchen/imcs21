import json
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import argparse

# load normalized symptom
# prefix = 'dataset'
prefix = '../../dataset'
sym2id = {value: key for key, value in pd.read_csv(os.path.join(prefix, 'symptom_norm.csv'))['norm'].items()}
num_labels = len(sym2id)


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def make_label(symptoms, target):
    if target == 'exp':
        label = [0] * num_labels
        for sx in symptoms:
            if sym2id.get(sx) is not None:
                label[sym2id.get(sx)] = 1
    else:
        label = [0] * (num_labels * 3)
        for sx_norm, sx_type in symptoms.items():
            if sym2id.get(sx_norm) is not None:
                label[sym2id.get(sx_norm) * 3 + int(sx_type)] = 1
    return label


def hamming_score(golds, preds):
    assert len(golds) == len(preds)
    out = np.ones(len(golds))
    n = np.logical_and(golds, preds).sum(axis=1)
    d = np.logical_or(golds, preds).sum(axis=1)
    return np.mean(np.divide(n, d, out=out, where=d != 0))


def multi_label_metric(golds, preds):
    # Example-based Metrics
    print('Exact Match Ratio: {}'.format(accuracy_score(golds, preds, normalize=True, sample_weight=None)))
    print('Hamming loss: {}'.format(hamming_loss(golds, preds)))
    print('Hamming score: {}'.format(hamming_score(golds, preds)))
    print('Sample Precision: {}'.format(recall_score(y_true=golds, y_pred=preds, average='samples', zero_division=0)))
    print('Sample Recall: {}'.format(precision_score(y_true=golds, y_pred=preds, average='samples', zero_division=0)))
    print('Sample F1: {}'.format(f1_score(y_true=golds, y_pred=preds, average='samples', zero_division=0)))
    # Label-based Metrics
    print('micro Precision: {}'.format(recall_score(y_true=golds, y_pred=preds, average='micro', zero_division=0)))
    print('micro Recall: {}'.format(precision_score(y_true=golds, y_pred=preds, average='micro', zero_division=0)))
    print('micro F1: {}'.format(f1_score(y_true=golds, y_pred=preds, average='micro', zero_division=0)))


def labels_metric(golds, preds):
    f1 = f1_score(golds, preds, average='macro')
    acc = accuracy_score(golds, preds)
    labels = classification_report(golds, preds, output_dict=True)
    print('F1-score -> positive: {}, negative: {}, uncertain: {}, overall: {}, acc: {}'.format(
        labels['1']['f1-score'], labels['0']['f1-score'], labels['2']['f1-score'], f1, acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--gold", required=True, type=str)
    parser.add_argument("--pred", required=True, type=str)
    args = parser.parse_args()

    # gold_data = load_json('dataset/test.json')
    # pred_data = load_json('task/SLI/MTL-SLI-SLI-back/mtl_imp_pred.json')

    gold_data = load_json(args.gold)
    pred_data = load_json(args.pred)

    if args.target == 'exp':
        golds_sx, preds_sx = [], []
        for pid, sample in gold_data.items():
            gold = sample['explicit_info']['Symptom']
            pred = pred_data.get(pid)
            golds_sx.append(make_label(gold, target='exp'))
            preds_sx.append(make_label(pred, target='exp'))
        golds_sx, preds_sx = np.array(golds_sx), np.array(preds_sx)
        print('-- symptom metric --')
        multi_label_metric(golds_sx, preds_sx)
    else:
        golds_sx, preds_sx, golds_full, preds_full = [], [], [], []
        gold_labels, pred_labels = [], []
        for pid, sample in gold_data.items():
            gold = sample['implicit_info']['Symptom']
            pred = pred_data.get(pid)
            golds_sx.append(make_label(gold, target='exp'))
            preds_sx.append(make_label(pred, target='exp'))
            golds_full.append(make_label(gold, target='imp'))
            preds_full.append(make_label(pred, target='imp'))
            for sx in gold:
                if sx in pred:
                    gold_labels.append(gold.get(sx))
                    pred_labels.append(pred.get(sx))
        golds_sx, preds_sx, golds_full, preds_full = \
            np.array(golds_sx), np.array(preds_sx), np.array(golds_full), np.array(preds_full)
        print('-- symptom metric --')
        multi_label_metric(golds_sx, preds_sx)
        print('-- symptom with label metric --')
        multi_label_metric(golds_full, preds_full)
        print('-- label metric --')
        labels_metric(gold_labels, pred_labels)
