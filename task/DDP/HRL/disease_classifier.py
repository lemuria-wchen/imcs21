import torch
import torch.nn.functional
import os
import numpy as np
from collections import namedtuple
import pickle
import copy
import random

class Model(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        # different layers. Two layers.
        self.policy_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )

        # one layer.
        #self.policy_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        '''
        if torch.cuda.is_available():
            x.cuda()
        '''
        q_values = self.policy_layer(x.float())
        return q_values

class dl_classifier(object):
    def __init__(self, input_size, hidden_size, output_size,  parameter):
        self.parameter = parameter
        self.device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
        self.model = Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(self.device)

        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)

        self.optimizer = torch.optim.Adam([
            {'params': weight_p, 'weight_decay': 0.001},  # with L2 regularization
            {'params': bias_p, 'weight_decay': 0}  # no L2 regularization.
        ], lr=0.01)
        #], lr=parameter.get("dqn_learning_rate"))

        self.criterion = torch.nn.CrossEntropyLoss()
        named_tuple = ("slot","disease")
        self.Transition = namedtuple('Transition', named_tuple)
        #self.test_batch = self.create_data(train_mode=False)

        #if self.params.get("train_mode") is False and self.params.get("agent_id").lower() == 'agentdqn':
        #    self.restore_model(self.params.get("saved_model"))

    def train(self, batch):
        if not batch:
            return {"loss": 'None'}
        batch = self.Transition(*zip(*batch))
        #print(batch.slot.shape)
        slot = torch.LongTensor(batch.slot).to(self.device)
        disease = torch.LongTensor(batch.disease).to(self.device)
        out = self.model.forward(slot)
        #print(disease.shape)
        #print(out.shape)
        #print(out.shape, disease)
        loss = self.criterion(out, disease)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, slots):
        self.model.eval()
        # print(batch.slot.shape)
        slots = torch.LongTensor(slots).to(self.device)
        Ys = self.model.forward(slots)
        max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        self.model.train()
        return Ys, max_index


    def train_dl_classifier(self, epochs):
        batch_size = self.parameter.get("batch_size")
        #print(batch_size)
        #print(self.total_batch[0])
        total_batch = self.create_data(train_mode=True)
        for iter in range(epochs):
            batch = random.sample(total_batch, batch_size)
            #print(batch[0][0].shape)
            loss = self.train(batch)
            if iter%100==0:
                print('epoch:{},loss:{:.4f}'.format(iter, loss["loss"]))

    def test_dl_classifier(self):
        self.model.eval()
        self.test_batch = self.create_data(train_mode=False)
        batch = self.Transition(*zip(*self.test_batch))
        slot = torch.LongTensor(batch.slot).to(self.device)
        #disease = torch.LongTensor(batch.disease).to(self.device)
        disease = batch.disease
        Ys, pred = self.predict(slot)
        #print(pred)
        num_correct = len([1 for i in range(len(disease)) if disease[i]==pred[i]])
        print("the test accuracy is %f", num_correct / len(self.test_batch))
        self.model.train()

    def test(self, test_batch):
        #self.model.eval()

        batch = self.Transition(*zip(*test_batch))
        slot = torch.LongTensor(batch.slot).to(self.device)
        #disease = torch.LongTensor(batch.disease).to(self.device)
        disease = batch.disease
        Ys, pred = self.predict(slot.cpu())
        #print(pred)
        num_correct = len([1 for i in range(len(disease)) if disease[i]==pred[i]])
        #print("the test accuracy is %f", num_correct / len(self.test_batch))
        test_acc = num_correct / len(test_batch)
        #self.model.train()
        return test_acc



    def create_data(self, train_mode):
        goal_set = pickle.load(open(self.parameter.get("goal_set"), 'rb'))
        self.slot_set = pickle.load(open(self.parameter.get("slot_set"), 'rb'))
        disease_symptom = pickle.load(open(self.parameter.get("disease_symptom"),'rb'))

        self.disease2id = {}
        for disease, v in disease_symptom.items():
            self.disease2id[disease] = v['index']
        self.slot_set.pop('disease')
        disease_y = []
        # total_set = random.sample(goal_set['train'], 10000)
        if train_mode==True:
            total_set = copy.deepcopy(goal_set["train"])
        else:
            total_set = copy.deepcopy(goal_set["test"])
        total_batch = []


        for i, dialogue in enumerate(total_set):
            slots_exp = [0] * len(self.slot_set)
            tag = dialogue['disease_tag']
            # tag_group=disease_symptom1[tag]['symptom']
            disease_y.append(tag)
            goal = dialogue['goal']
            explicit = goal['explicit_inform_slots']
            for exp_slot, value in explicit.items():
                #try:
                    slot_id = self.slot_set[exp_slot]
                    if value == True:
                        slots_exp[slot_id] = 1
                #except:
                #    pass
            if sum(slots_exp) == 0:
                print("############################")
            total_batch.append((slots_exp, self.disease2id[tag]))
        #print("the disease data creation is over")
        return total_batch

    def save_model(self,  model_performance, episodes_index, checkpoint_path):
        if os.path.isdir(checkpoint_path) == False:
            os.makedirs(checkpoint_path)
        agent_id = self.parameter.get("agent_id").lower()
        disease_number = self.parameter.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_match_rate = model_performance["avg_f1"]
        average_match_rate2 = model_performance["avg_recall"]
        model_file_name = os.path.join(checkpoint_path, "model_d" + str(disease_number) + str(agent_id) + "_s" + str(
            success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn) \
                                       + "_mr" + str(average_match_rate) + "_mr2-" + str(
            average_match_rate2) + "_e-" + str(episodes_index) + ".pkl")

        torch.save(self.model.state_dict(), model_file_name)

    def restore_model(self, saved_model):
        """
        Restoring the trained parameters for the model. Both current and target net are restored from the same parameter.

        Args:
            saved_model (str): the file name which is the trained model.
        """
        print("loading trained model", saved_model)
        if torch.cuda.is_available() is False:
            map_location = 'cpu'
        else:
            map_location = None
        self.model.load_state_dict(torch.load(saved_model,map_location='cpu'))

    def eval_mode(self):
        self.model.eval()


