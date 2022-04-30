import json
import os
import re

import numpy as np

prefix = '../../dataset'


# 计算其中的诊断字段对疾病的判断准确度，只要在诊断字段中包含真实疾病对应的关键词，即认为诊断成功
diseases_map = {
    '上呼吸道感染': ['呼吸道'],
    '小儿便秘': ['便秘'],
    '小儿发热': ['发热'],
    '小儿咳嗽': ['咳嗽'],
    '小儿感冒': ['感冒'],
    '小儿支气管炎': ['支气管', '肺', '支气'],
    '小儿支气管肺炎': ['肺', '支气管', '支气'],
    '小儿消化不良': ['消化', '腹'],
    '小儿腹泻': ['腹', '消化'],
    '新生儿黄疸': ['黄疸']
}


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_gold(dataset='test'):
    gold_data = load_json(os.path.join(prefix, '{}.json'.format(dataset)))
    diseases = []
    for pid, sample in gold_data.items():
        diseases.append(sample['diagnosis'])
    return diseases


def regex_acc(diseases, reports):
    n = 0
    for disease, report in zip(diseases, reports):
        res = re.findall(r'.*诊断：(.*)?[建议]+', report)
        if len(res) > 0:
            assert len(res) == 1
            _in = False
            for j in diseases_map[disease]:
                if j in res[0]:
                    _in = True
            if _in:
                n += 1
    print('regex-based diagnostic accuracy: {}'.format(round(n / len(reports), 4)))


if __name__ == '__main__':

    # test set
    gold_diseases = load_gold(dataset='test')

    saved_path = 'mrs_for_ner_test.txt'
    with open(saved_path, 'r', encoding='utf-8') as f:
        mrs_for_ner_test = []
        for line in f.readlines():
            mrs_for_ner_test.append(line.strip())

    all_reports = np.split(np.array(mrs_for_ner_test), 7)

    print('-' * 25 + ' Test Set ' + '-' * 25)
    regex_acc(gold_diseases, all_reports[2])
    regex_acc(gold_diseases, all_reports[3])
    regex_acc(gold_diseases, all_reports[4])
    regex_acc(gold_diseases, all_reports[5])
    regex_acc(gold_diseases, all_reports[6])

    # dev set
    gold_diseases = load_gold(dataset='dev')

    saved_path = 'mrs_for_ner_dev.txt'
    with open(saved_path, 'r', encoding='utf-8') as f:
        mrs_for_ner_dev = []
        for line in f.readlines():
            mrs_for_ner_dev.append(line.strip())

    all_reports = np.split(np.array(mrs_for_ner_dev), 7)

    print('-' * 25 + ' Dev Set ' + '-' * 25)
    regex_acc(gold_diseases, all_reports[2])
    regex_acc(gold_diseases, all_reports[3])
    regex_acc(gold_diseases, all_reports[4])
    regex_acc(gold_diseases, all_reports[5])
    regex_acc(gold_diseases, all_reports[6])

