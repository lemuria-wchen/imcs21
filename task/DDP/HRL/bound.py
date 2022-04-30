import sys, os
import pickle
sys.path.append(os.getcwd().replace("HRL/dialogue_system/dialogue_manager",""))
from HRL.dialogue_system.disease_classifier import dl_classifier
import numpy as np
import random
import copy
import random
from collections import deque
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
#file0 = '/remote-home/czhong/RL/data/dxy_dataset/dxy_dataset/100symptoms/HRL-1label/'
#file0 = '/remote-home/czhong/RL/data/new_data/mz10/allsymptoms/HRL_4label/'
#file0 = '/remote-home/czhong/RL/data/data/dataset/label/allsymptoms/HRL/'
#file0='/remote-home/czhong/RL/data/dxy_dataset/dxy_dataset/100symptoms/HRL/'
file0='/remote-home/czhong/RL/data/Fudan-Medical-Dialogue2.0/synthetic_dataset'
#file0='/remote-home/czhong/RL/data/dxy_dataset/dxy_dataset/100symptoms/HRL/'
#file0 = '/remote-home/czhong/RL/data/data/dataset/label/100symptoms/HRL'
#file0 = '/remote-home/czhong/RL/data/data/dataset/label/allsymptoms/HRL-1label/'
action_set = file0 + '/action_set.p'
slot_set = file0 +'/slot_set.p' 
goal_set = file0 + '/goal_set.p' 
#goal_set = '/remote-home/czhong/RL/log/MeicalChatbot-HRL-master/MeicalChatbot-HRL-master/HRL/dialogue_system/records/1114090542.p'
disease_symptom = file0 + "/disease_symptom.p" 


class dataset(Dataset):
    def init(self, loader):
        self.loader = loader
    def __getitem__(self, index):
        seq = self.loader[0]
        label = self.loader[1]
        return seq, label
    def __len__(self):
        return len(self.loader)
    
def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

class Bound_cal(object):
    def __init__(self, parameter):
        self.slot_set = pickle.load(open(slot_set, 'rb'))
        self.disease_symptom = pickle.load(open(disease_symptom,'rb'))
        self.parameter = parameter
        try:
            self.slot_set.pop('disease')
        except:
            pass
        self.id2disease = {}
        self.disease2id = {}
        for disease, v in self.disease_symptom.items():
            self.id2disease[v['index']] = disease
            self.disease2id[disease] = v['index']
        self.train_disease_replay = list()
        self.test_disease_replay = list()

    def build_deep_learning_classifier(self):
        self.model = dl_classifier(input_size=len(self.slot_set), hidden_size=256,
                                    output_size=len(self.disease_symptom),
                                    parameter=self.parameter)


    def train_deep_learning_classifier(self, epochs):
        #print("############   the deep learning model is training over  ###########")
        best_test_acc = 0
        for iter in range(epochs):
            loss = 0
            try:
                random.shuffle(self.train_disease_replay)
                self.train_disease_replay_split = list_split(self.train_disease_replay, 32)
                for batch in self.train_disease_replay_split:
                    loss += self.model.train(batch=batch)['loss']
            except:
                pass
            self.model.eval_mode()
            test_batch = random.sample(self.test_disease_replay, len(self.test_disease_replay))
            test_acc = self.model.test(test_batch=test_batch)
            if iter % 10 == 0:
                print('Iteration:{},loss:{:.4f}, test_acc:{:.4f}'.format(iter, loss/len(self.train_disease_replay), test_acc))

            if test_acc > best_test_acc:
                best_test_acc = test_acc
        return best_test_acc

    def save_dl_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        temp_checkpoint_path = os.path.join(checkpoint_path, 'classifier/')
        self.model.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)
    
    def flush(self):
        self.train_disease_replay = list()
        self.test_disease_replay = list()
        self.build_deep_learning_classifier()
    
    def current_state_representation(self, state, imp = False):
        """
        The state representation for the input of disease classifier.
        :param state: the last dialogue state before fed into disease classifier.
        :return: a vector that has equal length with slot set.
        """
        assert 'disease' not in self.slot_set.keys()
        state_rep = [0]*len(self.slot_set)
        current_slots = copy.deepcopy(state['goal'])
        if imp:
            for slot, value in current_slots["implicit_inform_slots"].items():
                if value == '1' or value == True:
                    state_rep[self.slot_set[slot]] = 1
                elif value == '0' or value == False:
                    state_rep[self.slot_set[slot]] = -1
        for slot, value in current_slots["explicit_inform_slots"].items():
            if value == '1' or value == True:
                state_rep[self.slot_set[slot]] = 1
            elif value == '0'  or value == False:
                state_rep[self.slot_set[slot]] = -1
        return state_rep

    def build_dataset(self, goal_set, imp):
        self.flush()
        for record in goal_set['train']:
            state_rep = bound_cal.current_state_representation(record, imp)
            disease = record["disease_tag"]
            self.train_disease_replay.append((state_rep, self.disease2id[disease]))
            
        for record in goal_set['test']:
            state_rep = bound_cal.current_state_representation(record, imp)
            disease = record["disease_tag"]
            self.test_disease_replay.append((state_rep, self.disease2id[disease]))

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", dest="batch_size", type=int, default=100, help="the batch size when training.")
args = parser.parse_args()
parameter = vars(args)
torch.cuda.manual_seed(220)
torch.cuda.manual_seed(306)
torch.manual_seed(200)
torch.manual_seed(200)

bound_cal = Bound_cal(parameter)
bound_cal.build_deep_learning_classifier()
goal_set = pickle.load(open(goal_set,'rb'))

imp = True
bound_cal.build_dataset(goal_set, imp)
#bound_cal.model.restore_model('/remote-home/czhong/RL/log/MeicalChatbot-HRL-master/MeicalChatbot-HRL-master/HRL/dialogue_system/model/DQN/checkpoint/1108041451_agenthrljoint2_T22_ss100_lr0.0005_RFS6_RFF0_RFNCY0.0_RFIRS80_RFRA-4_RFRMT-100_mls0_gamma1_gammaW0.9_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs1_dtft0_ird0_ubc0.985_lbc1e-10_data_RID0/classifier/model_d10agenthrljoint2_s0.784_r152.859_t17.291_mr0.291_mr2-0.368_e-4656.pkl')
test_acc = bound_cal.model.test(test_batch=bound_cal.test_disease_replay)
upper_bound = bound_cal.train_deep_learning_classifier(500)


imp = False
bound_cal.build_dataset(goal_set, imp)
lower_bound = bound_cal.train_deep_learning_classifier(10)

print("Upper Bound = ", upper_bound)
print("Lower Bound = ", lower_bound)