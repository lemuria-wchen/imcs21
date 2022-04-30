import json
import os
import argparse

from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu


# prefix = 'dataset'
prefix = '../../dataset'


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def process(title, delimiter=''):
    x = []
    for key, value in title.items():
        x.append(key + '：' + value)
    return delimiter.join(x)


def compute_rouge(source, targets):
    try:
        r1, r2, rl = 0, 0, 0
        n = len(targets)
        for target in targets:
            source, target = ' '.join(source), ' '.join(target)
            scores = Rouge().get_scores(hyps=source, refs=target)
            r1 += scores[0]['rouge-1']['f']
            r2 += scores[0]['rouge-2']['f']
            rl += scores[0]['rouge-l']['f']
        return {
            'rouge-1': r1 / n,
            'rouge-2': r2 / n,
            'rouge-l': rl / n,
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]
    print({k: v / len(targets) for k, v in scores.items()})


def bleu(refs, hyp):
    b1 = corpus_bleu(refs, hyp, weights=(1, 0, 0, 0))
    b2 = corpus_bleu(refs, hyp, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(refs, hyp, weights=(1/3, 1/3, 1/3, 0))
    b4 = corpus_bleu(refs, hyp, weights=(0.25, 0.25, 0.25, 0.25))
    print('bleu-1/2/3/4: {} / {} / {} / {}'.format(b1, b2, b3, b4))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, default='../../dataset/test.json', help='gold file path')
    parser.add_argument('--pred_path', type=str, help='pred file path')
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--target', default='non_t5', type=str)

    args = parser.parse_args()

    gold_data = load_json(os.path.join(prefix, '{}.json'.format(args.dataset)))

    golds = []
    for pid, sample in gold_data.items():
        title1, title2 = sample['report'][0], sample['report'][1]
        golds.append([process(title1), process(title2)])

    preds = []
    if args.target == 't5':
        with open(args.pred_path, 'r') as f:
            for line in f.readlines():
                pred, _ = line.split('\t')
                preds.append(pred)
    else:
        with open(args.pred_path, 'r') as f:
            for line in f.readlines():
                preds.append(''.join(line.strip().split()))

    compute_rouges(preds, golds)

    _golds = [[' '.join(g).split() for g in gold] for gold in golds]
    _preds = [' '.join(pred).split() for pred in preds]
    bleu(_golds, _preds)
