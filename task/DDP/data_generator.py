import pickle


goal_set = pickle.load(file=open('./goal_set.p', "rb"))
goal_test_set = pickle.load(file=open('./goal_test_set.p', "rb"))
output_files = './allsymptoms/'
slot_set = dict()
disease_set = list()
disease_set_dict = dict()
disease_symptom = dict()
slot_set_dict = dict()

slot_set_test = dict()
count = 0
goal_train_set = goal_set['train']

for info in goal_train_set:
    goal = info['goal']
    disease_tag = info['disease_tag']
    for symptom, state in goal['explicit_inform_slots'].items():
        slot_set[symptom] = slot_set.get(symptom, 0) + 1
    for symptom, state in goal['implicit_inform_slots'].items():
        slot_set[symptom] = slot_set.get(symptom, 0) + 1

# slot_set = dict(sorted(slot_set.items(),key=lambda x:x[1],reverse=True)[:100:])
slot_set = dict(sorted(slot_set.items(), key=lambda x: x[1], reverse=True))
slot_set_key = set(slot_set.keys())

for info in goal_set['train']:
    disease_tag = info['disease_tag']
    if disease_tag not in disease_set:
        disease_symptom[disease_tag] = dict()
        disease_symptom[disease_tag]['index'] = count
        disease_set_dict[disease_tag] = count
        count += 1
        disease_symptom[disease_tag]['Symptom'] = dict()
    disease_set.append(info['disease_tag'])
    goal = info['goal']
    disease_set_dict[disease_tag] = disease_set_dict.get(disease_tag, 0) + 1
    goal['explicit_inform_slots'] = {k: v for k, v in goal['explicit_inform_slots'].items() if k in slot_set_key}
    for symptom, state in goal['explicit_inform_slots'].items():
        disease_symptom[disease_tag]['Symptom'][symptom] = disease_symptom[disease_tag]['Symptom'].get(symptom, 0) + 1
    goal['implicit_inform_slots'] = {k: v for k, v in goal['implicit_inform_slots'].items() if k in slot_set_key}
    for symptom, state in goal['implicit_inform_slots'].items():
        disease_symptom[disease_tag]['Symptom'][symptom] = disease_symptom[disease_tag]['Symptom'].get(symptom, 0) + 1
        if symptom in goal['explicit_inform_slots'].keys():
            print(1)

for info in goal_set['dev']:
    disease_tag = info['disease_tag']

    if disease_tag not in disease_set:
        disease_symptom[disease_tag] = dict()
        disease_symptom[disease_tag]['index'] = count
        disease_set_dict[disease_tag] = count
        count += 1
        disease_symptom[disease_tag]['Symptom'] = dict()
    disease_set.append(info['disease_tag'])
    goal = info['goal']

    goal['explicit_inform_slots'] = {k: v for k, v in goal['explicit_inform_slots'].items() if k in slot_set_key}
    for symptom, state in goal['explicit_inform_slots'].items():
        disease_symptom[disease_tag]['Symptom'][symptom] = disease_symptom[disease_tag]['Symptom'].get(symptom, 0) + 1
    goal['implicit_inform_slots'] = {k: v for k, v in goal['implicit_inform_slots'].items() if k in slot_set_key}
    for symptom, state in goal['implicit_inform_slots'].items():
        disease_symptom[disease_tag]['Symptom'][symptom] = disease_symptom[disease_tag]['Symptom'].get(symptom, 0) + 1


for info in goal_test_set['test']:
    disease_tag = info['disease_tag']
    if disease_tag not in disease_set:
        disease_symptom[disease_tag] = dict()
        disease_symptom[disease_tag]['index'] = count
        disease_set_dict[disease_tag] = count
        count += 1
        disease_symptom[disease_tag]['Symptom'] = dict()
    disease_set.append(info['disease_tag'])
    goal = info['goal']
    goal['explicit_inform_slots'] = {k: v for k, v in goal['explicit_inform_slots'].items() if k in slot_set_key}
    for symptom, state in goal['explicit_inform_slots'].items():
        disease_symptom[disease_tag]['Symptom'][symptom] = disease_symptom[disease_tag]['Symptom'].get(symptom, 0) + 1
    goal['implicit_inform_slots'] = {k: v for k, v in goal['implicit_inform_slots'].items() if k in slot_set_key}
    for symptom, state in goal['implicit_inform_slots'].items():
        disease_symptom[disease_tag]['Symptom'][symptom] = disease_symptom[disease_tag]['Symptom'].get(symptom, 0) + 1
        if symptom in goal['explicit_inform_slots'].keys():
            print(1)
for info in goal_test_set['test']:
    goal = info['goal']
    disease_tag = info['disease_tag']
    for symptom, state in goal['explicit_inform_slots'].items():
        slot_set_test[symptom] = slot_set.get(symptom, 0) + 1
    for symptom, state in goal['implicit_inform_slots'].items():
        slot_set_test[symptom] = slot_set.get(symptom, 0) + 1

disease_set_key = set(disease_set)

for key, value in enumerate(slot_set_key):
    slot_set_dict[value] = key

pickle.dump(slot_set_dict, open('.//allsymptoms//slot_set.p', 'wb'))
pickle.dump(disease_set_dict, open('.//allsymptoms//disease_set.p', 'wb'))
pickle.dump(disease_symptom, open('.//allsymptoms//disease_symptom.p', 'wb'))
pickle.dump(goal_set, open('.//allsymptoms//goal_set.p', 'wb'))
pickle.dump(goal_test_set, open('.//allsymptoms//goal_test_set.p', 'wb'))
