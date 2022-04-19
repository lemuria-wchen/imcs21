import pickle
import json
import os


def load_json(fn):
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data


data_dir = '../../../dataset'
# data_dir = 'dataset'
train_dia = load_json(os.path.join(data_dir, 'train.json'))
dev_dia = load_json(os.path.join(data_dir, 'dev.json'))
test_dia = load_json(os.path.join(data_dir, 'test.json'))


goal_train_dev = {'train': []}
goal_test = {'test': []}

all_symptom = []


def generate_data(ff):
    count = 0
    total = list()
    for all_value in ff:
        record = dict()
        count += 1
        record['consult_id'] = count
        record['disease_tag'] = all_value['label']
        record['goal'] = dict()
        record['goal']['explicit_inform_slots'] = dict()
        record['goal']['implicit_inform_slots'] = dict()

        for key, values in all_value['exp_sxs'].items():
            record['goal']['explicit_inform_slots'][key] = '1'

        for keys, values in all_value['imp_sxs'].items():
            if keys not in record['goal']['explicit_inform_slots'].keys():
                record['goal']['implicit_inform_slots'][keys] = values
        total.append(record)
    return total


train_data = generate_data(train_dia)
dev_data = generate_data(dev_dia)
test_data = generate_data(test_dia)

goal_train_dev['train'] = train_data
goal_train_dev['dev'] = dev_data
goal_test['test'] = test_data

slot_set_train = []
slot_set_dev = []
slot_set_test = []

for all_value in goal_train_dev['train']:
    for key, values in all_value['goal']['explicit_inform_slots'].items():
        slot_set_train.append(key)
    for key, values in all_value['goal']['implicit_inform_slots'].items():
        slot_set_train.append(key)
slot_set_train = set(slot_set_train)

for all_value in goal_train_dev['dev']:
    for key,values in all_value['goal']['explicit_inform_slots'].items():
        slot_set_dev.append(key)
    for key, values in all_value['goal']['implicit_inform_slots'].items():
        slot_set_dev.append(key)
slot_set_dev = set(slot_set_dev)

for all_value in goal_test['test']:
    for key, values in all_value['goal']['explicit_inform_slots'].items():
        slot_set_test.append(key)
    for key, values in all_value['goal']['implicit_inform_slots'].items():
        slot_set_test.append(key)
slot_set_test = set(slot_set_test)

pickle.dump(goal_train_dev, open('./goal_set.p', 'wb'))
pickle.dump(goal_test, open('./goal_test_set.p', 'wb'))

print('finish!')
