import json
import os
import re
import sys

# prefix = 'dataset'
prefix = '../../dataset'


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def process(title, delimiter='\t'):
    x = []
    for key, value in title.items():
        x.append(key + '：' + value)
    return delimiter.join(x)


mrs = {}

gold_data = load_json(os.path.join(prefix, 'test.json'))
golds = []
for pid, sample in gold_data.items():
    title1, title2 = sample['report'][0], sample['report'][1]
    golds.append([process(title1), process(title2)])

mrs.update({'gold': golds})

preds = []
with open('t5/data/predict_result.tsv', 'r') as f:
    for line in f.readlines():
        pred, _ = line.split('\t')
        preds.append(pred)
mrs.update({'t5': preds})

preds = []
with open('prophetnet/sorted_outputs.txt', 'r') as f:
    for line in f.readlines():
        preds.append(''.join(line.strip().split()))
mrs.update({'prophetnet': preds})


for file in ['lstm', 'pg', 'tf']:
    preds = []
    with open('opennmt/data/pred_{}.txt'.format(file), 'r') as f:
        for line in f.readlines():
            preds.append(''.join(line.strip().split()))
    mrs.update({file: preds})


keys = ['gold', 'lstm', 'pg', 'tf', 'prophetnet', 't5']

with open('../NER/LeBERT/datasets/msra/test.json', 'w', encoding='utf-8') as f:
    for key in keys:
        for value in mrs.get(key):
            if key == 'gold':
                for v in value:
                    text = [_ for _ in v.replace(' ', '')]
                    label = ['O'] * len(text)
                    f.write(json.dumps({'text': text, 'label': label}, ensure_ascii=False) + '\n')
            else:
                text = [_ for _ in value.replace(' ', '')]
                label = ['O'] * len(text)
                f.write(json.dumps({'text': text, 'label': label}, ensure_ascii=False) + '\n')


import numpy as np

xxx = np.load('../NER/BERT-NER/reports.npy', allow_pickle=True)

ref1, ref2, lstm, pg, tf, prophetnet, t5 = np.split(xxx, 7)


def get_entity_bio(seq, raw):
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    entities = []
    for chunk in chunks:
        entities.append(raw[chunk[1]: chunk[2]+1])
    return entities


def compute(origin, found, right):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1


ref = list(ref1) + list(ref2)

origins = []
founds = []
rights = []
a, b, c = 0, 0, 0
for i in range(len(lstm)):
    label_path1 = ref[2 * i]
    label_path2 = ref[2 * i + 1]
    ref_text1 = mrs.get('gold')[i][0].replace(' ', '')
    ref_text2 = mrs.get('gold')[i][1].replace(' ', '')
    a += len(ref_text1) + len(ref_text2)
    pre_path = prophetnet[i]
    pred_text = mrs['prophetnet'][i].replace(' ', '')
    assert len(label_path1) == len(ref_text1)
    assert len(label_path2) == len(ref_text2)
    assert len(pre_path) == len(pred_text)
    label_entities1 = set(get_entity_bio(label_path1, ref_text1))
    label_entities2 = set(get_entity_bio(label_path2, ref_text2))
    pre_entities = set(get_entity_bio(pre_path, pred_text))
    a += len(label_entities1.intersection(pre_entities))
    a += len(label_entities2.intersection(pre_entities))
    b += len(label_entities1) + len(label_entities2)
    c += 2 * len(pre_entities)
r = a/b
p = a/c
print('p/r/f1: {}/{}/{}'.format(round(p, 4), round(r, 4), round(2*r*p/(p+r), 4)))


#       p/r/f1
# lstm  0.4242/0.3239/0.3673
# pg    0.5586/0.3962/0.4636
# tf    0.4622/0.3626/0.4064
# pro   0.5642/0.4427/0.4961
# t5    0.577/0.3902/0.4655

# {'gold': 87.9778, 't5': 53.5709, 'prophetnet': 65.1714, 'lstm': 65.1467, 'pg': 62.5055, 'tf': 66.709}
# {'t5': 0.476, 'prophetnet': 0.5536, 'lstm': 0.4834, 'pg': 0.566, 'tf': 0.545}
