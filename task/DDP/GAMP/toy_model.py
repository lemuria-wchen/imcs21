from network import Discriminator, Generator, Inference, init_weights        
import utils
from dataloader import get_loader, LSTM_get_loader, Dis_get_loader, Inf_get_loader
import pickle
import sys, os
import torch
from torch import optim
import random
import numpy as np
def progress_bar_warm_gen(bar_len, SR, loss, best_SR, currentNumber, wholeNumber):
    # 20, success_rate, avg_turns, avg_object, success_rate_test, avg_turns_test, best_success_rate_test, best_avg_turns_test, i, simulate_epoch_number
    """
    bar_len 进度条长度
    currentNumber 当前迭代数
    wholeNumber 总迭代数
    """
    filled_len = int(round(bar_len * currentNumber / float(wholeNumber)))
    percents = round(100.0 * currentNumber / float(wholeNumber), 1)
    bar = '\033[32;1m%s\033[0m' % '>' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(\
        '[%d/%d][%s] %s%s \033[31;1mSR\033[0m = %3f \033[36;1mLoss\033[0m= %3f Best_SR\033[0m = %3f \033[36;1m \r' %\
         (int(currentNumber),int(wholeNumber), bar, '\033[32;1m%s\033[0m' % percents, '%', SR, loss, best_SR))
    sys.stdout.flush()


file0='./Data/mz10/allsymptoms/'
slot_set = pickle.load(file=open(file0+'/slot_set.p', "rb"))
disease_set = pickle.load(file=open(file0+'/disease_set.p', "rb"))
for idx, disease in enumerate(list(disease_set.keys())):
    disease_set[disease] =  idx
goal_set = pickle.load(file=open(file0+'/goal_set.p', "rb"))
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

best_correct_num_gen = 0
best_correct_num_dis = 0
best_correct_num_inf = 0
loss = 0
CELoss = torch.nn.CrossEntropyLoss()
BCELoss = torch.nn.BCELoss()

training_set = goal_set['train']
state, disease = utils.dataset_generate(training_set, slot_set, disease_set)
num_slot = len(slot_set)
num_disease = len(disease_set)
Inference_test = Inference(input_size= num_slot, output_size=num_disease)
best_model = Inference(input_size= num_slot, output_size=num_disease)

optimizer_inf = optim.Adam(Inference_test.parameters(), 0.01)
Inference_test = Inference_test.to(device)
best_model = best_model.to(device)

test_symptom = list()
test_disease = list()
test_set = goal_set['test']
for record in test_set:
    origin_state = dict()
    for key, values in record['goal']['explicit_inform_slots'].items():
        origin_state[slot_set[key]] = values
    data_length = len(record['goal']['explicit_inform_slots'])
    goal_symptom = record['goal']['explicit_inform_slots'].update(record['goal']['implicit_inform_slots'])
    origin_symptom = list(origin_state.keys())
    action_tensor = 0
    before_state, after_state = utils.seq_generate(origin_symptom, action_tensor, data_length, origin_state, num_slot)
    test_symptom.append(before_state)
    test_disease.append(disease_set[record['disease_tag']])


state_len = len(state)

list_index = list(range(state_len))
train_index = random.sample(list_index, int(state_len*0.8))
test_index =  [item for item in list_index if item not in set(train_index)]

state = np.array(state)
disease = np.array(disease)
Inf_dataset = Inf_get_loader(state[train_index], disease[train_index])
Inf_valid_dataset = Inf_get_loader(state[test_index], disease[test_index])


Inf_test_dataset = Inf_get_loader(test_symptom, test_disease)

print("#### Now training the inference ####")
for i in range(15):
    correct_num  = 0
    total_loss = 0
    length = 0
    max_prob = 0
    for j, (state, label) in enumerate(Inf_dataset):
        state = state.to(device)
        label = label.to(device)
        #length += state.size(0)
        output = Inference_test(state.float())
        loss = CELoss(output, label)
        optimizer_inf.zero_grad()
        loss.backward()
        optimizer_inf.step()
        total_loss += loss.item()

    for j, (state, label) in enumerate(Inf_valid_dataset):
        state = state.to(device)
        label = label.to(device)
        length += state.size(0)
        output = Inference_test(state.float())
        output_max = torch.argmax(output, dim = 1)

        for j in range(output_max.size()[0]):
            max_prob += output[j][output_max[j]]
            if output_max[j] == label[j]:
                correct_num += 1
    max_prob = max_prob / length
    if best_correct_num_inf < correct_num/length:
        best_correct_num_inf = correct_num/length
        best_model.load_state_dict(Inference_test.state_dict())
    progress_bar_warm_gen(20, correct_num/length, max_prob, best_correct_num_inf, i, 15)


for j, (state, label) in enumerate(Inf_test_dataset):
    state = state.to(device)
    label = label.to(device)
    length += state.size(0)
    output = best_model(state.float())