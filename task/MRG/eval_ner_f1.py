import json
import os
import numpy as np
import pickle

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


def load_preds(pred_path, target='no_t5'):
    preds = []
    if target == 't5':
        with open(pred_path, 'r') as f:
            for line in f.readlines():
                pred, _ = line.split('\t')
                preds.append(pred)
    else:
        with open(pred_path, 'r') as f:
            for line in f.readlines():
                preds.append(''.join(line.strip().split()))
    return preds


def load_gold(dataset='test'):
    gold_data = load_json(os.path.join(prefix, '{}.json'.format(dataset)))
    golds = []
    for pid, sample in gold_data.items():
        title1, title2 = sample['report'][0], sample['report'][1]
        golds.append([process(title1), process(title2)])
    return golds


def get_entity_bio(raw, seq):
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


def compute_metric(origin, found, right, digits=4):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    print('Knowledge P: {}, R: {}, F1: {}'.format(round(recall, digits), round(precision, digits), round(f1, digits)))


def entity_f1(refs_1, refs_2, refs_1_tags, refs_2_tags, hyps, hyps_tags):
    num_origins = 0
    num_founds = 0
    num_rights = 0
    for ref_1, ref_2, ref_1_tag, ref_2_tag, hyp, hyp_tag in zip(
            refs_1, refs_2, refs_1_tags, refs_2_tags, hyps, hyps_tags):
        es1 = set(get_entity_bio(ref_1[: len(ref_1_tag)], ref_1_tag))
        es2 = set(get_entity_bio(ref_2[: len(ref_2_tag)], ref_2_tag))
        es3 = set(get_entity_bio(hyp[: len(hyp_tag)], hyp_tag))
        num_origins += len(es1) + len(es2)
        num_founds += 2 * len(es3)
        num_rights += len(es1.intersection(es3)) + len(es2.intersection(es3))
    compute_metric(num_origins, num_founds, num_rights)


if __name__ == '__main__':

    golds_test = load_gold(dataset='test')
    golds_dev = load_gold(dataset='dev')

    lstm_preds_test = load_preds('opennmt/data/pred_lstm.txt')
    pg_preds_test = load_preds('opennmt/data/pred_pg.txt')
    tf_preds_test = load_preds('opennmt/data/pred_tf.txt')
    t5_preds_test = load_preds('t5/data/predict_result.tsv', target='t5')
    prophetnet_preds_test = load_preds('prophetnet/test_pred.txt')
    assert len(golds_test) == len(lstm_preds_test) == len(pg_preds_test) == len(tf_preds_test) == len(
        t5_preds_test) == len(prophetnet_preds_test)

    lstm_preds_dev = load_preds('opennmt/data/pred_lstm_dev.txt')[::2]
    pg_preds_dev = load_preds('opennmt/data/pred_pg_dev.txt')[::2]
    tf_preds_dev = load_preds('opennmt/data/pred_tf_dev.txt')[::2]
    t5_preds_dev = load_preds('t5/data/predict_result_dev.tsv', target='t5')
    prophetnet_preds_dev = load_preds('prophetnet/dev_pred.txt')
    assert len(golds_dev) == len(lstm_preds_dev) == len(pg_preds_dev) == len(tf_preds_dev) == len(t5_preds_dev) == len(
        prophetnet_preds_dev)

    mrs_for_ner_test = [
        [g[0] for g in golds_test], [g[1] for g in golds_test], lstm_preds_test, pg_preds_test, tf_preds_test, t5_preds_test, prophetnet_preds_test
    ]

    mrs_for_ner_dev = [
        [g[0] for g in golds_dev], [g[1] for g in golds_dev], lstm_preds_dev, pg_preds_dev, tf_preds_dev, t5_preds_dev, prophetnet_preds_dev
    ]

    print([round(sum([len(mr) for mr in mrs]) / len(mrs), 4) for mrs in mrs_for_ner_test[2:7]])
    print([round(sum([len(mr) for mr in mrs]) / len(mrs), 4) for mrs in mrs_for_ner_dev[2:7]])

    """
    # ----------------------------------------------------------------------------------------------------------------------
    # run BERT-NER inference script to recognize entities for all medical reports
    saved_path = 'mrs_for_ner_test.txt'

    with open(saved_path, 'w', encoding='utf-8') as f:
        for i in range(len(mrs_for_ner_dev)):
            for mr in mrs_for_ner_dev[i]:
                f.write(mr + '\n')
                
    # ----------------------------------------------------------------------------------------------------------------------
    """

    # test set
    with open('mrs_entities_test.pkl', 'rb') as f:
        bio_tags_test = pickle.load(f)

    all_bio_tags = np.split(np.array(bio_tags_test), 7)

    refs_test_1 = [gold[0] for gold in golds_test]
    refs_test_2 = [gold[1] for gold in golds_test]

    print('-' * 25 + ' Test Set ' + '-' * 25)
    entity_f1(refs_test_1, refs_test_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_test[2], all_bio_tags[2])
    entity_f1(refs_test_1, refs_test_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_test[3], all_bio_tags[3])
    entity_f1(refs_test_1, refs_test_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_test[4], all_bio_tags[4])
    entity_f1(refs_test_1, refs_test_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_test[5], all_bio_tags[5])
    entity_f1(refs_test_1, refs_test_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_test[6], all_bio_tags[6])

    # dev set
    with open('mrs_entities_dev.pkl', 'rb') as f:
        bio_tags_dev = pickle.load(f)

    all_bio_tags = np.split(np.array(bio_tags_dev), 7)

    refs_dev_1 = [gold[0] for gold in golds_dev]
    refs_dev_2 = [gold[1] for gold in golds_dev]

    print('-' * 25 + ' Dev Set ' + '-' * 25)
    entity_f1(refs_dev_1, refs_dev_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_dev[2], all_bio_tags[2])
    entity_f1(refs_dev_1, refs_dev_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_dev[3], all_bio_tags[3])
    entity_f1(refs_dev_1, refs_dev_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_dev[4], all_bio_tags[4])
    entity_f1(refs_dev_1, refs_dev_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_dev[5], all_bio_tags[5])
    entity_f1(refs_dev_1, refs_dev_2, all_bio_tags[0], all_bio_tags[1], mrs_for_ner_dev[6], all_bio_tags[6])

