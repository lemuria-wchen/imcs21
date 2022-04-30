import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
from collections import Counter
import re

from utils import load_data, Stats


# 导入数据
data_path = 'dataset/raw/corpus.json'
# data_path = 'dataset/train.json'
# data_path = 'dataset/test.json'
data = load_data(data_path)

# 可视化参数设置
font_scale = 1.8


from collections import defaultdict
# 实体总数量
entity_set = defaultdict(set)

b_prefixs = ['B-Symptom', 'B-Medical_Examination', 'B-Drug', 'B-Drug_Category', 'B-Operation']
i_prefixs = ['I-Symptom', 'I-Medical_Examination', 'I-Drug', 'I-Drug_Category', 'I-Operation']


for _, value in data.items():
    for item in value['dialogue']:
        bio_labels = [_.strip() for _ in item['BIO_label'].split(' ')]
        n = len(bio_labels)
        for b_prefix, i_prefix in zip(b_prefixs, i_prefixs):
            start_idx, end_idx = 0, 0
            while start_idx < n and end_idx < n:
                while start_idx < n and not bio_labels[start_idx].startswith(b_prefix):
                    start_idx += 1
                end_idx = start_idx + 1
                while end_idx < n and bio_labels[end_idx].startswith(i_prefix):
                    end_idx += 1
                # 将标注的实体添加到集合中
                entity_set[b_prefix].add(item['sentence'][start_idx: end_idx])
                start_idx = end_idx

a = {key: len(value) for key, value in entity_set.items()}

aa = set()
for key, val in data.items():
    for w in val['self_report']:
        aa.add(w)
    for sent in val['dialogue']:
        for w in sent['sentence']:
            aa.add(w)
    for w in val['report'][0]:
        aa.add(w)
    for w in val['report'][1]:
        aa.add(w)

# 基本数据统计
stats = Stats(samples=data)
stats.num_samples()
stats.dis_stats()
stats.sx_stats()



# ----------------------------------------------------------------------------------------------------------------------
# 话语和词分布
num_utterances = np.array([len(value['dialogue']) for _, value in data.items()])
num_words = np.array([sum([len(item['sentence']) for item in value['dialogue']]) for _, value in data.items()])

with PdfPages('utter_dist.pdf') as pdf:
    plt.clf()
    _, ax = plt.subplots()
    sns.set_context('paper', font_scale=font_scale)
    ax.hist(x=num_utterances, bins=25, color='#607c8e', alpha=0.7, rwidth=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    plt.xlabel('Number of utterances')
    plt.ylabel('Frequency')
    plt.grid(True, color="grey", linewidth="1.0", linestyle="-.", axis='y', alpha=0.2)
    plt.tight_layout()
    # plt.show()
    pdf.savefig()

with PdfPages('words_dist.pdf') as pdf:
    plt.clf()
    sns.set_context('paper', font_scale=font_scale)
    _, ax = plt.subplots()
    ax.hist(x=num_words, bins=25, color='#607c8e', alpha=0.7, rwidth=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(True, color="grey", linewidth="1.0", linestyle="-.", axis='y', alpha=0.2)
    plt.tight_layout()
    # plt.show()
    pdf.savefig()

# ----------------------------------------------------------------------------------------------------------------------
# 疾病分布
dis_types = Counter([value['diagnosis'] for _, value in data.items()])
print(dis_types)

with PdfPages('dis_dist.pdf') as pdf:
    # 定义字体大小
    sns.set_context('paper', font_scale=font_scale)
    group_names = ['小儿支气管炎', '小儿发热', '小儿腹泻', '上呼吸道感染', '小儿消化不良',
                   '小儿感冒', '小儿咳嗽', '新生儿黄疸', '小儿便秘', '小儿支气管肺炎']
    # URI is short for Upper Respiratory Infection
    # BPN is short for Bronchopneumonia
    group_names_en = ['Bronchitis', 'Fever', 'Diarrhea', 'URI', 'Dyspepsia',
                      'Cold', 'Cough', 'Jaundice', 'Constipation', 'BPN']
    group_names_en.reverse()
    group_size = [543, 542, 534, 486, 475, 472, 344, 294, 221, 205]
    group_size.reverse()
    group_size_ratio = np.array(group_size) / np.sum(group_size)
    _, ax = plt.subplots()
    width = 0.6
    ax.barh(group_names_en, group_size, width, color='#607c8e', alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.xaxis.set_label_position('top')
    plt.xlim(0, 700)
    # 显示数字百分比
    for i, p in enumerate(ax.patches[-len(group_names):]):
        x, y = p.get_xy()
        ax.annotate('{:.1%}'.format(group_size_ratio[i]), (group_size[i] + 10, y))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    plt.grid(True, color="grey", linewidth="1.0", linestyle="-.", axis='x', alpha=0.2)
    plt.tight_layout()
    # plt.show()
    pdf.savefig()

# ----------------------------------------------------------------------------------------------------------------------
# 实体类型分布
entity_types = defaultdict(int)
for key, value in data.items():
    for item in value['dialogue']:
        for et in item['BIO_label'].split(' '):
            if et.strip().startswith('B'):
                if item['speaker'] == '医生':
                    entity_types['d_' + et.strip()] += 1
                else:
                    entity_types['p_' + et.strip()] += 1
print(entity_types)

with PdfPages('et_dist.pdf') as pdf:
    plt.clf()
    # 数据
    group_names = ['Symptom', 'Examination', 'Drug Name', 'Drug Category', 'Operation']
    # group_names = ['SX', 'EX', 'DN', 'DC', 'OP']
    group_size = [62802, 18223, 12435, 9990, 4282]
    # 定义字体大小
    sns.set_context('paper', font_scale=font_scale)
    fig, ax = plt.subplots()
    ax.axis('equal')
    colors = [plt.cm.Reds, plt.cm.Blues, plt.cm.Purples, plt.cm.Greens, plt.cm.Oranges]
    pie, _, _ = ax.pie(group_size, labels=group_names, autopct='%1.1f%%',
                       colors=[color(0.5) for color in colors],
                       radius=1.3, pctdistance=0.6, labeldistance=1.06, startangle=-30)
    # plt.tight_layout()
    plt.legend(loc=(-0.15, 0.70))
    # plt.show()
    pdf.savefig()

# ----------------------------------------------------------------------------------------------------------------------
# 实体位置分布计算
split = 5
entity_pos = np.zeros((5, split))
entity_list = ['B-Symptom', 'B-Medical_Examination', 'B-Drug', 'B-Drug_Category', 'B-Operation']

for key, value in data.items():
    _num = len(value['dialogue'])
    for i, item in enumerate(value['dialogue']):
        for et in item['BIO_label'].split(' '):
            if et.strip().startswith('B'):
                entity_pos[entity_list.index(et.strip())][int(i // (_num / split))] += 1
# 按列归一化
entity_prop = entity_pos / entity_pos.sum(axis=1).reshape(-1, 1)

with PdfPages('et_pos.pdf') as pdf:
    plt.clf()
    sns.set_context('paper', font_scale=font_scale)
    cmap = sns.color_palette("Spectral", as_cmap=True)
    x_label = ['0%-20%', '20%-40%', '40%-60%', '60%-80%', '80%-100%']
    y_label = ['Symptom', 'Examination', 'Drug', 'Drug Category', 'Operation']
    ax = sns.heatmap(entity_prop, center=0.8, square=False, linewidths=0.1,
                     cbar_kws={"shrink": 1.0}, xticklabels=x_label, yticklabels=y_label)
    ax.xaxis.tick_top()
    plt.xticks(rotation=20)
    plt.tight_layout()
    # plt.show()
    pdf.savefig()

# ----------------------------------------------------------------------------------------------------------------------
# 对话行为统计
action_types = defaultdict(int)
for key, value in data.items():
    for item in value['dialogue']:
        if item['speaker'] == '医生':
            action_types['D_' + item['dialogue_act']] += 1
        else:
            action_types['P_' + item['dialogue_act']] += 1
print(action_types)

with PdfPages('da_dist.pdf') as pdf:
    plt.clf()
    # 简写详情
    # Physical Characteristic(PC); Symptom (SX); Etiology (ETIOL); Existing Examination and Treatment (EET);
    # Make Diagnose (MD); Drug Recommendation (DR); Medical Advice (MA); Precautions (PRCTN)
    # 定义标签名称
    group_names = ['R-ETIOL', 'R-PRCTN', 'R-MA', 'I-ETIOL', 'DIAG', 'R-BI', 'R-DR', 'I-MA', 'R-EET', 'I-BI',
                   'I-PRCTN', 'I-EET', 'I-DR', 'R-SX', 'I-SX']
    group_size = [1504, 1630, 2017, 2533, 3744, 5054, 5642, 6247, 6567, 7974, 8282, 9878, 11196, 14848, 21830]
    doctor_group_size = [157, 4, 18, 2395, 3701, 5032, 31, 6230, 6529, 34, 8243, 38, 11149, 14741, 251]
    patient_group_size = [1347, 1626, 1999, 138, 43, 22, 5611, 17, 38, 7940, 39, 9840, 47, 107, 21579]
    group_size_ratio = np.array(group_size) / np.sum(group_size)
    # 定义字体大小
    sns.set_context('paper', font_scale=font_scale)
    _, ax = plt.subplots()
    width = 0.65
    ax.barh(group_names, doctor_group_size, width,
            label='Doctor', edgecolor='black', color='cornflowerblue')
    ax.barh(group_names, patient_group_size, width, left=doctor_group_size,
            label='Patient', edgecolor='black', color='sandybrown')
    ax.set_xlabel('Frequency')
    ax.xaxis.set_label_position('top')
    # 显示数字百分比
    for i, p in enumerate(ax.patches[-len(group_names):]):
        x, y = p.get_xy()
        ax.annotate('{:.1%}'.format(group_size_ratio[i]), (group_size[i] + 200, y), size=14)
    # 淡化坐标轴框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    # 横坐标简化表示
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: '{:,.0f}'.format(v / 1000) + 'K'))
    plt.grid(True, color="grey", linewidth="1.0", linestyle="-.", axis='x', alpha=0.2)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    pdf.savefig()

# ----------------------------------------------------------------------------------------------------------------------
# 对话行为位置分布计算
split = 5
action_list = [
    'Request-Basic_Information', 'Inform-Basic_Information', 'Request-Symptom', 'Inform-Symptom', 'Request-Etiology',
    'Inform-Etiology', 'Request-Existing_Examination_and_Treatment', 'Inform-Existing_Examination_and_Treatment',
    'Diagnose', 'Request-Drug_Recommendation', 'Inform-Drug_Recommendation', 'Request-Medical_Advice',
    'Inform-Medical_Advice', 'Request-Precautions', 'Inform-Precautions']
action_pos = np.zeros((len(action_list), split))
for key, value in data.items():
    _num = len(value['dialogue'])
    for i, item in enumerate(value['dialogue']):
        if item['dialogue_act'] in action_list:
            action_pos[action_list.index(item['dialogue_act'])][int(i // (_num / split))] += 1
# 按列归一化
action_prop = action_pos / action_pos.sum(axis=1).reshape(-1, 1)

# 对话行为位置热图统计
# 观察：
with PdfPages('da_pos.pdf') as pdf:
    plt.clf()
    # 设置字体
    sns.set_context('paper', font_scale=font_scale)
    # 设置调色板
    # cmap = sns.color_palette("Spectral", as_cmap=True)
    # cmap = sns.color_palette("pastel", as_cmap=True)
    # 图中的标签
    x_label = ['0%-20%', '20%-40%', '40%-60%', '60%-80%', '80%-100%']
    y_label = ['R-BI', 'I-BI', 'R-SX', 'I-SX', 'R-ETIOL', 'I-ETIOL', 'R-EET', 'I-EET',
               'DIAG', 'R-DR', 'I-DR', 'R-MA', 'I-MA', 'R-PRCTN', 'I-PRCTN']
    # 热图
    ax = sns.heatmap(action_prop, center=0.8, square=False, linewidths=0.1,
                     cbar_kws={"shrink": 1.0}, xticklabels=x_label, yticklabels=y_label)
    ax.xaxis.tick_top()
    plt.xticks(rotation=20)
    plt.tight_layout()
    # plt.show()
    pdf.savefig()

# # ----------------------------------------------------------------------------------------------------------------------
# # 查看症状
# speaker = '患者'
# dialogue_act = 'Request-Symptom'
# for _, value in data.items():
#     for item in value['dialogue']:
#         if item['speaker'] == speaker and item['dialogue_act'] == dialogue_act:
#             print(item)
#
# # 医生 Inform-Symptom
# # 1. 体温比较高，是发热的
# # 2. 精神可以，说明病情不严重，不要担心
#
# # 病人 Request-Symptom
# # 1. 奶瓣是什么形状？
# # 2. 脱水后有什么症状么？
#
# # ----------------------------------------------------------------------------------------------------------------------
# 症状和疾病的共现信息
diseases, symptoms = [], []
for _, value in data.items():
    diseases.append(value['diagnosis'])
    for item in value['dialogue']:
        for symptom in item['symptom_norm']:
            symptoms.append(symptom)
dis_counter = Counter(diseases).most_common(n=10)
sx_counter = Counter(symptoms).most_common(n=50)
print(sx_counter)
# use translate api
dis_labels = ['Bronchitis', 'Fever', 'Diarrhea', 'URI', 'Dyspepsia', 'Cold', 'Cough', 'Jaundice', 'Constipation', 'BPN']
sx_labels = ['Cough', 'Fever', 'Cold', 'Phlegm', 'Diarrhea', 'Runny nose', 'Vomit', 'Moderately hot', 'Jaundice',
             'Pneumonia', 'Indigestion', 'High fever', 'Loose stool', 'Viral infection', 'Stuffy nose', 'Bronchitis',
             'Low heat', 'Bacterial infections', 'Crying', 'Constipate']
dis2id = {item[0]: idx for idx, item in enumerate(dis_counter)}
sx2id = {item[0]: idx for idx, item in enumerate(sx_counter)}
com = np.zeros((len(dis2id), len(sx2id)))
for _, value in data.items():
    dis_id = dis2id.get(value['diagnosis'])
    sx_ids = set()
    for item in value['dialogue']:
        for sx in item['symptom_norm']:
            if sx in sx2id:
                sx_ids.add(sx2id.get(sx))
    for sx_id in sx_ids:
        com[dis_id][sx_id] += 1
com_norm = com / com.sum(axis=0)

with PdfPages('dis_sx.pdf') as pdf:
    plt.clf()
    sns.set(rc={'figure.figsize': (16, 9)})
    # ax = sns.heatmap(com_norm, xticklabels=sx_labels, yticklabels=dis_labels, annot=False, linewidths=0.01, cbar=True, cbar_kws={"shrink": 1.0})
    ax = sns.heatmap(com_norm, xticklabels=False, yticklabels=dis_labels, annot=False, linewidths=0.01, cbar=True, cbar_kws={"shrink": 1.0})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('Top-50 most common symptoms', labelpad=20, fontsize=32)
    # ax.xaxis.set_tick_params(labelsize=28)
    ax.yaxis.set_tick_params(labelsize=28)
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    # plt.show()
    pdf.savefig()


symptom_labels = defaultdict(int)
for key, value in data.items():
    for item in list(value['implicit_info']['Symptom'].values()):
        symptom_labels[item] += 1
    symptom_labels['explicit_symptom'] += len(value['explicit_info']['Symptom'])

group_size = [16524, 6365, 4275]
group_size_ratio = np.array(group_size) / np.sum(group_size)
print(np.array(group_size) / len(data))
print(np.sum(group_size) / len(data))


#
from nltk.translate.bleu_score import sentence_bleu

b4 = []
for key, value in data.items():
    r1 = value['report'][0]
    r2 = value['report'][1]
    # b4.append(sentence_bleu([r1], r2))
    b4.append(sentence_bleu([r2], r1))

print(np.mean(np.array(b4)))

# ----------------------------------------------------------------------------------------------------------------------
# 0.84979424 0.95475113 0.79243542 0.72674419 0.65572034
# 0.83793738 0.82195122 0.90947368 0.81928839 0.98129252

# 0.84979424 0.95475113 0.79243542 0.72674419 0.65572034
# 0.84346225 0.88292683 0.95052632 0.93071161 0.98129252

fields = ['主诉', '现病史', '辅助检查', '既往史', '诊断', '建议']
field = '既往史'

n = 0
for _, value in data.items():
    report1, report2 = value['report'][0], value['report'][1]
    disease = value['diagnosis']
    dis_id = diseases.index(disease)
    res1 = re.findall(r'.*{}：(.*)?'.format(field), report1)
    res2 = re.findall(r'.*{}：(.*)?'.format(field), report2)
    if res1 and ('不详' in res1[0] or '无' in res1[0] or '暂缺' in res1[0]):
        n += 1
    if res2 and ('不详' in res2[0] or '无' in res2[0] or '暂缺' in res2[0]):
        n += 1

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
diseases = list(diseases_map.keys())

fields = ['主诉', '现病史', '辅助检查', '既往史', '诊断', '建议']
field = '诊断'

n, N = np.zeros(len(diseases)), np.zeros(len(diseases))

for _, value in data.items():
    report1, report2 = value['report'][0], value['report'][1]
    disease = value['diagnosis']
    dis_id = diseases.index(disease)
    res1 = re.findall(r'.*{}：(.*)?'.format(field), report1)
    res2 = re.findall(r'.*{}：(.*)?'.format(field), report2)
    if res1:
        _in = False
        for item in diseases_map[disease]:
            if item in res1[0]:
                _in = True
        if _in:
            n[dis_id] += 1
    if res2:
        _in = False
        for item in diseases_map[disease]:
            if item in res2[0]:
                _in = True
        if _in:
            n[dis_id] += 1
    N[dis_id] += 2

print(n/N)
print(sum(n)/sum(N))


# number of entities
n = 0
for _, value in data.items():
    for item in value['dialogue']:
        bio_labels = [_.strip() for _ in item['BIO_label'].split(' ')]
        for bio_label in bio_labels:
            if bio_label.startswith('B'):
                n += 1

print(n / len(data))


# len of self-report
n = 0
for _, value in data.items():
    report1 = value['report'][0]
    report2 = value['report'][1]
    n += len(report1) + len(report2)

print(n / len(data) / 2)


# Medical Report 平均字数统计
print(np.mean([len(''.join(value['report'][0].split('\n'))) + len(''.join(value['report'][1].split('\n')))
               for _, value in data.items()]) / 2 - 40)

# 分部分统计
report_parts = ['主诉', '现病史', '辅助检查', '既往史', '诊断', '建议']
report_part = report_parts[0]

for report_part in report_parts:
    s, n = 0, 0
    for _, value in data.items():
        res1 = re.findall(r'.*{}：(.*)?'.format(report_part), value['report'][0])
        res2 = re.findall(r'.*{}：(.*)?'.format(report_part), value['report'][1])
        if res1 and res2:
            s += len(res1[0]) + len(res2[0])
            n += 1
    print('{}: {}'.format(report_part, s / n / 2))
