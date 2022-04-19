import os
import pandas as pd
from utils import load_json, write_json, get_entity_bio


def make_dataset(samples, num_contexts=1, weight=3, mode='train'):
    instances = []
    for pid, sample in samples.items():
        # break
        diag = sample['dialogue']
        num_utts = len(diag)
        for i in range(num_utts):
            # 话语和其上下文作为输入
            utts, bio_tags, bio_ids, chunks = [], [], [], []
            start_pos, end_pos = 0, 0
            # task 1: predict bio labels
            start = max(i - num_contexts, 0)
            end = min(i + num_contexts + 1, num_utts)
            for j in range(start, end):
                _utt = diag[j]['speaker'] + '：' + diag[j]['sentence']
                utts.append(_utt)
                if mode != 'test':
                    if j != i:
                        _bio_tag = ['PAD'] * len(_utt)
                    else:
                        _bio_tag = ['O'] * 3 + [tag if 'Symptom' in tag else 'O' for tag in diag[j]['BIO_label'].split()]
                    assert len(_utt) == len(_bio_tag)
                else:
                    _bio_tag = []
                bio_tags.append(_bio_tag)
                if j == i:
                    start_pos = sum([len(utt) + 1 for utt in utts[:-1]])
                    end_pos = sum([len(utt) + 1 for utt in utts]) - 2
                    if mode != 'test':
                        _chunks = get_entity_bio(_bio_tag)
                        assert len(_chunks) == len(diag[i]['symptom_norm']) == len(diag[i]['symptom_type'])
                        for chunk, sn, st in zip(_chunks, diag[i]['symptom_norm'], diag[i]['symptom_type']):
                            assert sn in sym2id
                            assert st in sl2id
                            # task 2/3: predict normalized name and symptom label
                            chunks.append([start_pos + chunk[1], start_pos + chunk[2], sym2id.get(sn), sl2id.get(st)])
            utts = '\t'.join(utts)
            if mode != 'test':
                for tags in bio_tags:
                    for tag in tags:
                        bio_ids.append(bio2id.get(tag))
                    bio_ids.append(0)
                bio_ids.pop(-1)
                assert len(utts) == len(bio_ids)
                _weight = weight if len(diag[i]['symptom_norm']) > 0 else 1
            else:
                _weight = 1
            instance = {
                'utts': utts,
                'bio_ids': bio_ids,
                'chunks': chunks,
                'bounds': [start_pos, end_pos],
                'weight': _weight,
                'pid': pid
            }
            instances.append(instance)
    return instances


if __name__ == '__main__':

    data_dir = '../../../dataset'
    # data_dir = 'dataset'
    train_set = load_json(os.path.join(data_dir, 'train.json'))
    dev_set = load_json(os.path.join(data_dir, 'dev.json'))
    test_set = load_json(os.path.join(data_dir, 'test_input.json'))

    saved_path = 'data'
    os.makedirs(os.path.join(saved_path), exist_ok=True)

    sym2id = {value: key for key, value in pd.read_csv(os.path.join(data_dir, 'symptom_norm.csv'))['norm'].items()}
    bio2id = {item: idx for idx, item in enumerate(['PAD', 'O', 'B-Symptom', 'I-Symptom'])}
    sl2id = {'0': 0, '1': 1, '2': 2}

    train = make_dataset(train_set, mode='train')
    dev = make_dataset(dev_set, mode='dev')
    test = make_dataset(test_set, mode='test')

    print('train/dev/test size: {}/{}/{}'.format(len(train), len(dev), len(test)))

    write_json(train, os.path.join(saved_path, 'train.json'))
    write_json(dev, os.path.join(saved_path, 'dev.json'))
    write_json(test, os.path.join(saved_path, 'test.json'))
