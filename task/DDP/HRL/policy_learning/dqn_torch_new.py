# -*- coding: utf8 -*-

import torch
import torch.nn.functional
import os
import numpy as np
from collections import namedtuple


class DQNModel(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    """
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(DQNModel, self).__init__()
        self.params = parameter
        # different layers. Two layers.
        self.policy_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )

        # one layer.
        #self.policy_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        q_values = self.policy_layer(x)
        return q_values

class DQNModel2(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    """
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(DQNModel2, self).__init__()
        self.params = parameter
        # different layers. Two layers.


        self.policy_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.1),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Dropout(0.1),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        q_values = self.policy_layer(x)
        return q_values

class DuelingDQN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(DuelingDQN, self).__init__()
        self.params = parameter
        self.output_size = output_size
        # different layers. Two layers.
        self.fc_adv = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )
        self.fc_val = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, 1, bias=True)
        )

        '''
        self.fc1_adv = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc1_val = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.fc2_adv = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
        self.fc2_val = torch.nn.Linear(in_features=hidden_size, out_features=1)

        self.relu = torch.nn.LeakyReLU()
        '''
        # one layer.
        # self.policy_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        #adv = self.relu(self.fc1_adv(x))
        #val = self.relu(self.fc1_val(x))

        #adv = self.fc2_adv(adv)
        #val = self.fc2_val(val).expand(x.size(0), self.output_size)

        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), self.output_size)

        q_values = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.output_size)
        return q_values

class DQNModelWithRelational(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    """
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(DQNModelWithRelational, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = parameter
        # different layers. Two layers.
        self.policy_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )

        # Relational Refinement
        self.relational_weights = torch.Tensor(output_size, output_size).to(self.device)
        # one layer.
        # self.policy_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def get_q_values(self, x):
        q_1 = self.policy_layer(x)
        # print(q_1.size())
        q_2 = torch.mm(q_1, self.relational_weights)
        return q_1 + q_2

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        q_values = self.get_q_values(x)
        return q_values

class DQNEncoder(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    dont know whether the non-linear is right
    """
    def __init__(self, input_size):
        super(DQNEncoder, self).__init__()
        # different layers. Two layers.
        self.Encoder_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2048, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
        )
    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        embedding = self.Encoder_layer(x)
        return embedding

class DQNSymptomDecoder(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    dont know whether the non-linear is right
    """
    def __init__(self, output_size):
        super(DQNSymptomDecoder, self).__init__()
        # different layers. Two layers.???
        self.Encoder_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024, bias=True),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(),
        )
    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        embedding = self.Encoder_layer(x)
        return embedding
class DQN(object):
    def __init__(self, input_size, hidden_size, output_size, parameter, named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over')):
        self.params = parameter
        self.Transition = namedtuple('Transition', named_tuple)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.current_net = DQNModel(input_size, hidden_size, output_size, parameter).to(self.device)
        self.target_net = DQNModel(input_size, hidden_size, output_size, parameter).to(self.device)

        print(self.current_net)

        if torch.cuda.is_available():
            if parameter["multi_GPUs"] == True: # multi GPUs
                self.current_net = torch.nn.DataParallel(self.current_net)
                self.target_net = torch.nn.DataParallel(self.target_net)
            else:# Single GPU
                self.current_net.cuda(device=self.device)
                self.target_net.cuda(device=self.device)

        self.target_net.load_state_dict(self.current_net.state_dict()) # Copy paraameters from current networks.
        self.target_net.eval()  # set this model as evaluate mode. And it's parameters will not be updated.

        # Optimizer with L2 regularization
        weight_p, bias_p = [], []
        for name, p in self.current_net.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)

        self.optimizer = torch.optim.Adam([
            {'params': weight_p, 'weight_decay': 0.1}, # with L2 regularization
            {'params': bias_p, 'weight_decay': 0} # no L2 regularization.
        ], lr=self.params.get("dqn_learning_rate",0.0002))

        if self.params.get("train_mode") is False and self.params.get("agent_id").lower() == 'agentdqn':
            self.restore_model(self.params.get("saved_model"))
            self.current_net.eval()
            self.target_net.eval()

    def singleBatch(self, batch, params):
        """
         Training the model with the given batch of data.

        Args:
            batch (list): the batch of data, each data point in the list is a tuple: (state, agent_action, reward,
                next_state, episode_over).
            params (dict): dict like, the super-parameters.

        Returns:
            A scalar (float), the loss of this batch.

        """
        #print('batch_before',batch)
        gamma = params.get('gamma', 0.9)
        batch_size = len(batch)
        batch = self.Transition(*zip(*batch))
        #print('batch_after',batch)

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.LongTensor(list(batch.episode_over)).to(device=self.device)
        non_final_next_states = torch.Tensor([batch.next_state[i] for i in range(batch_size) if batch.episode_over[i] is False ]).to(device=self.device)
        state_batch = torch.Tensor(batch.state).to(device=self.device)
        #print(batch.agent_action)
        action_batch = torch.LongTensor(batch.agent_action).view(-1,1).to(device=self.device)
        reward_batch = torch.Tensor(batch.reward).to(device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.current_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN}) for all next states.
        if self.params.get("dqn_type") == "DQN" or self.params.get("dqn_type") == "DuelingDQN":
            next_state_values = self.next_state_values_DQN(batch_size=batch_size, non_final_mask=non_final_mask, non_final_next_states=non_final_next_states)
        elif self.params.get("dqn_type") == "DoubleDQN":
            next_state_values = self.next_state_values_double_DQN(batch_size=batch_size, non_final_mask=non_final_mask, non_final_next_states=non_final_next_states)
        else:
            raise ValueError("dqn_type should be one of ['DQN', 'DoubleDQN','DuelingDQN']")
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = torch.nn.functional.mse_loss(input=state_action_values,target=expected_state_action_values.view(-1, 1))

        # Optimize the model
        self.optimizer.zero_grad() # zero the gradients.
        loss.backward() # calculate the gradient.
        # for name, param in self.current_net.named_parameters():
        #     param.grad.data.clamp_(-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN, 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN) # gradient clipping
        self.optimizer.step()
        return {"loss":loss.item()}

    def next_state_values_DQN(self, batch_size, non_final_mask, non_final_next_states ):
        """
        Computate the values of all next states with DQN.
        `http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf`

        Args:
            batch_size (int): the size of given batch.
            non_final_mask (Tensor): shape: 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-D, [batch_size], 0: non-terminal state, 0: terminal state
            non_final_next_states (Tensor): 2-D, shape: [num_of_non_terminal_states, state_dim]

        Returns:
            A 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-D Tensor, shape:[batch_size]
        """
        # Compute V(s_{t+0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN}) for all next states.
        next_state_values = torch.zeros(batch_size).to(device=self.device)
        if non_final_next_states.size()[0] > 0: # All current states in this batch are the terminal states of their corresonpding sessions.
            next_state_values[non_final_mask==0] = self.target_net(non_final_next_states).max(1)[0].detach()
        return next_state_values

    def next_state_values_double_DQN(self,batch_size, non_final_mask, non_final_next_states):
        """
        Computate the values of all next states with Double DQN.
        `http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847`

        Args:
            batch_size (int): the size of given batch.
            non_final_mask (Tensor): shape: 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-D, [batch_size], 0: non-terminal state, 0: terminal state
            non_final_next_states (Tensor): 2-D, shape: [num_of_non_terminal_states, state_dim]

        Returns:
            A 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-D Tensor, shape:[batch_size]
        """
        next_state_values = torch.zeros(batch_size).to(device=self.device)
        if non_final_next_states.size()[0] > 0:
            next_action_batch_current = self.current_net(non_final_next_states).max(1)[1].view(-1,1).detach()
            next_state_values[non_final_mask==0] = self.target_net(non_final_next_states).gather(1, next_action_batch_current).detach().view(-1)
        return next_state_values

    def predict(self, Xs, **kwargs):
        # train_mode = kwargs.get("train_mode")
        # assert train_mode is not None
        # if train_mode is False:
        #     self.current_net.eval()
        Xs = torch.Tensor(Xs).to(device=self.device)
        Ys = self.current_net(Xs)
        #print(Ys.detach().numpy())
        # self.current_net.train()
        max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        return Ys, max_index[0]

    def predict_slot(self, Xs, **kwargs):
        # train_mode = kwargs.get("train_mode")
        # assert train_mode is not None
        # if train_mode is False:
        #     self.current_net.eval()
        slot_num = kwargs.get("slot_num")
        Xs = torch.Tensor(Xs).to(device=self.device)
        Ys = self.current_net(Xs)
        #slot_num =
        max_index = np.argmax(Ys.detach().cpu().numpy()[:,:slot_num], axis=1)
        return Ys, max_index[0]

    def predict_target(self, Xs, **kwargs):
        Xs = torch.Tensor(Xs).to(device=self.device)
        Ys = self.target_net(Xs)
        #max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        return Ys

    def save_model(self, model_performance,episodes_index, checkpoint_path):
        """
        Saving the trained model.

        Args:
            model_performance (dict): the test result of the model, which contains different metrics.
            episodes_index (int): the current step of training. And this will be appended to the model name at the end.
            checkpoint_path (str): the directory that the model is going to save to. Default None.
        """
        if os.path.isdir(checkpoint_path) == False:
            # os.mkdir(checkpoint_path)
            #print(os.getcwd())
            os.makedirs(checkpoint_path)
        agent_id = self.params.get("agent_id").lower()
        disease_number = self.params.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_match_rate = model_performance["average_match_rate"]
        average_match_rate2 = model_performance["average_match_rate2"]
        model_file_name = os.path.join(checkpoint_path, "model_d" + str(disease_number) +  str(agent_id) + "_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn)\
                          + "_mr" + str(average_match_rate) + "_mr2-" + str(average_match_rate2) + "_e-" + str(episodes_index) + ".pkl")

        torch.save(self.current_net.state_dict(), model_file_name)

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
        self.current_net.load_state_dict(torch.load(saved_model,map_location=map_location))
        self.target_net.load_state_dict(self.current_net.state_dict())

    def update_target_network(self):
        """
        Updating the target network with the parameters copyed from the current networks.
        """
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.current_net.named_parameters()

class DQN2(object):
    def __init__(self, input_size, hidden_size, output_size, parameter, named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over')):
        self.params = parameter
        self.Transition = namedtuple('Transition', named_tuple)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        if self.params.get("agent_id").lower() == 'agenthrljoint2':
            self.Transition = namedtuple('Transition',('state', 'agent_action', 'reward', 'next_state', 'episode_over', 'subtask_turn'))

        self.current_net = DQNModel2(input_size, hidden_size, output_size, parameter).to(self.device)
        self.target_net = DQNModel2(input_size, hidden_size, output_size, parameter).to(self.device)

        print(self.current_net)

        if torch.cuda.is_available():
            if parameter["multi_GPUs"] == True: # multi GPUs
                self.current_net = torch.nn.DataParallel(self.current_net)
                self.target_net = torch.nn.DataParallel(self.target_net)
            else:# Single GPU
                self.current_net.cuda(device=self.device)
                self.target_net.cuda(device=self.device)

        self.target_net.load_state_dict(self.current_net.state_dict()) # Copy paraameters from current networks.
        self.target_net.eval()  # set this model as evaluate mode. And it's parameters will not be updated.

        # Optimizer with L2 regularization
        weight_p, bias_p = [], []
        for name, p in self.current_net.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)

        self.optimizer = torch.optim.Adam([
            {'params': weight_p, 'weight_decay': 0.0}, # with L2 regularization
            {'params': bias_p, 'weight_decay': 0} # no L2 regularization.
        ], lr=self.params.get("dqn_learning_rate",0.0002))

        if self.params.get("train_mode") is False and self.params.get("agent_id").lower() == 'agentdqn':
            self.restore_model(self.params.get("saved_model"))
            self.current_net.eval()
            self.target_net.eval()

    def singleBatch(self, batch, params):
        """
         Training the model with the given batch of data.

        Args:
            batch (list): the batch of data, each data point in the list is a tuple: (state, agent_action, reward,
                next_state, episode_over).
            params (dict): dict like, the super-parameters.

        Returns:
            A scalar (float), the loss of this batch.

        """
        gamma = params.get('gamma', 0.9)
        batch_size = len(batch)
        batch = self.Transition(*zip(*batch))
        #print('batch_after',batch)

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.LongTensor(list(batch.episode_over)).to(device=self.device)
        non_final_next_states = torch.Tensor([batch.next_state[i] for i in range(batch_size) if batch.episode_over[i] is False ]).to(device=self.device)
        state_batch = torch.Tensor(batch.state).to(device=self.device)
        #print(batch.agent_action)
        action_batch = torch.LongTensor(batch.agent_action).view(-1,1).to(device=self.device)
        reward_batch = torch.Tensor(batch.reward).to(device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.current_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN}) for all next states.
        if self.params.get("dqn_type") == "DQN" or self.params.get("dqn_type") == "DuelingDQN":
            next_state_values = self.next_state_values_DQN(batch_size=batch_size, non_final_mask=non_final_mask, non_final_next_states=non_final_next_states)
        elif self.params.get("dqn_type") == "DoubleDQN":
            next_state_values = self.next_state_values_double_DQN(batch_size=batch_size, non_final_mask=non_final_mask, non_final_next_states=non_final_next_states)
        else:
            raise ValueError("dqn_type should be one of ['DQN', 'DoubleDQN','DuelingDQN']")
        # Compute the expected Q values
        '''
        if self.params.get("agent_id").lower() == 'agenthrljoint2':
            subtask_turn_batch = torch.Tensor(batch.subtask_turn).to(device=self.device)
            gamma_batch = gamma ** subtask_turn_batch
            #print(gamma_batch)
            expected_state_action_values = (next_state_values * gamma_batch) + reward_batch
        else:
        '''
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # weight_sampling, mainly for master agent in agent_hrl.

        # Compute Huber loss
        loss = torch.nn.functional.mse_loss(input=state_action_values,target=expected_state_action_values.view(-1, 1))

        # Optimize the model
        self.optimizer.zero_grad() # zero the gradients.
        loss.backward() # calculate the gradient.
        # for name, param in self.current_net.named_parameters():
        #     param.grad.data.clamp_(-0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN, 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN) # gradient clipping
        self.optimizer.step()
        return {"loss":loss.item()}

    def next_state_values_DQN(self, batch_size, non_final_mask, non_final_next_states ):
        """
        Computate the values of all next states with DQN.
        `http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf`

        Args:
            batch_size (int): the size of given batch.
            non_final_mask (Tensor): shape: 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-D, [batch_size], 0: non-terminal state, 0: terminal state
            non_final_next_states (Tensor): 2-D, shape: [num_of_non_terminal_states, state_dim]

        Returns:
            A 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-D Tensor, shape:[batch_size]
        """
        # Compute V(s_{t+0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN}) for all next states.
        next_state_values = torch.zeros(batch_size).to(device=self.device)
        if non_final_next_states.size()[0] > 0: # All current states in this batch are the terminal states of their corresonpding sessions.
            next_state_values[non_final_mask==0] = self.target_net(non_final_next_states).max(1)[0].detach()
        return next_state_values

    def next_state_values_double_DQN(self,batch_size, non_final_mask, non_final_next_states):
        """
        Computate the values of all next states with Double DQN.
        `http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847`

        Args:
            batch_size (int): the size of given batch.
            non_final_mask (Tensor): shape: 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-D, [batch_size], 0: non-terminal state, 0: terminal state
            non_final_next_states (Tensor): 2-D, shape: [num_of_non_terminal_states, state_dim]

        Returns:
            A 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-D Tensor, shape:[batch_size]
        """
        next_state_values = torch.zeros(batch_size).to(device=self.device)
        if non_final_next_states.size()[0] > 0:
            next_action_batch_current = self.current_net(non_final_next_states).max(1)[1].view(-1,1).detach()
            next_state_values[non_final_mask==0] = self.target_net(non_final_next_states).gather(1, next_action_batch_current).detach().view(-1)
        return next_state_values

    def predict(self, Xs, **kwargs):
        # train_mode = kwargs.get("train_mode")
        # assert train_mode is not None
        # if train_mode is False:
        #     self.current_net.eval()
        Xs = torch.Tensor(Xs).to(device=self.device)
        Ys = self.current_net(Xs)
        #print(Ys.detach().numpy())
        # self.current_net.train()
        max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        return Ys, max_index[0]

    def predict_target(self, Xs, **kwargs):
        Xs = torch.Tensor(Xs).to(device=self.device)
        Ys = self.target_net(Xs)
        #max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        return Ys

    def save_model(self, model_performance,episodes_index, checkpoint_path):
        """
        Saving the trained model.

        Args:
            model_performance (dict): the test result of the model, which contains different metrics.
            episodes_index (int): the current step of training. And this will be appended to the model name at the end.
            checkpoint_path (str): the directory that the model is going to save to. Default None.
        """
        if os.path.isdir(checkpoint_path) == False:
            # os.mkdir(checkpoint_path)
            #print(os.getcwd())
            os.makedirs(checkpoint_path)
        agent_id = self.params.get("agent_id").lower()
        disease_number = self.params.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_match_rate = model_performance["average_match_rate"]
        average_match_rate2 = model_performance["average_match_rate2"]
        model_file_name = os.path.join(checkpoint_path, "model_d" + str(disease_number) +  str(agent_id) + "_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn)\
                          + "_mr" + str(average_match_rate) + "_mr2-" + str(average_match_rate2) + "_e-" + str(episodes_index) + ".pkl")

        torch.save(self.current_net.state_dict(), model_file_name)

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
        self.current_net.load_state_dict(torch.load(saved_model,map_location=map_location))
        self.target_net.load_state_dict(self.current_net.state_dict())

    def update_target_network(self):
        """
        Updating the target network with the parameters copyed from the current networks.
        """
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.current_net.named_parameters()