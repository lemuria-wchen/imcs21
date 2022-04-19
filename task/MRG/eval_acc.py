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


def process(title):
    r = []
    for part in title.split('\n'):
        xx = re.sub(r'[(]\d[)]', '', re.sub('\n', '', part.strip())).strip()
        if xx[-1] != '。':
            xx += '。'
        r.append(xx)
    return ''.join(r)


mrs = {}

gold_data = load_json(os.path.join(prefix, 'test.json'))
golds = []
gold_disease = []
for pid, sample in gold_data.items():
    title1, title2 = sample['report'][0], sample['report'][1]
    golds.append([process(title1), process(title2)])
    gold_disease.append(sample['diagnosis'])

mrs.update({'gold': golds})
mrs.update({'gold_dis': gold_disease})

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

# 计算平均长度
ss = {}
for key, items in mrs.items():
    n = 0
    for item in items:
        if isinstance(item, list):
            n += (len(item[0]) + len(item[1])) / 2
        else:
            n += len(item)
    ss.update({
        key: round(n / len(items), 4)
    })
print(ss)

# 计算其中的诊断字段
field = '诊断'

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

xxx = {}
for key, items in mrs.items():
    if key != 'gold' and key != 'gold_dis':
        n = 0
        for i, item in enumerate(items):
            dis = mrs['gold_dis'][i]
            res = re.findall(r'.*诊断：(.*)?[建议]+', item)
            if len(res) > 0:
                assert len(res) == 1
                _in = False
                for j in diseases_map[dis]:
                    if j in res[0]:
                        _in = True
                if _in:
                    n += 1
        xxx.update({key: round(n / len(items), 4)})
print(xxx)


# 长度
# {'gold': 87.9778, 't5': 53.5709, 'prophetnet': 65.1714, 'lstm': 65.1467, 'pg': 62.5055, 'tf': 66.709}
# {'t5': 0.476, 'prophetnet': 0.5536, 'lstm': 0.4834, 'pg': 0.566, 'tf': 0.545}


if __name__ == "__main__":
    # grounds = load_json(sys.argv[1])
    # predictions = load_json(sys.argv[2])
    pass
