# -*- coding: utf8 -*-
"""
The model in this file is reference to `Florensa, C., Duan, Y., & Abbeel, P. (2017). Stochastic neural networks for hierarchical reinforcement learning. arXiv preprint arXiv:1704.03012.`
https://arxiv.org/abs/1704.03012
"""

import torch
from collections import namedtuple
import os, sys
sys.path.append(os.getcwd().replace("Flat_DQN/dialogue_system/policy_learning",""))
from .dqn_torch import DQNModel


class DQNGoal(object):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.parameter = parameter
        self.Transition = namedtuple('Transition', ('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.current_net = DQNModel(input_size,hidden_size, output_size, parameter).to(self.device)
        self.target_net = DQNModel(input_size, hidden_size,output_size, parameter).to(self.device)

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

        self.optimizer = torch.optim.SGD([
            {'params': weight_p, 'weight_decay': 0.1}, # with L2 regularization
            {'params': bias_p, 'weight_decay': 0} # no L2 regularization.
        ], lr=self.parameter.get("dqn_learning_rate",0.001))

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

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.LongTensor(list(batch.episode_over)).to(device=self.device)
        non_final_next_states = torch.Tensor([batch.next_state[i] for i in range(batch_size) if batch.episode_over[i] is False ]).to(device=self.device)
        state_batch = torch.Tensor(batch.state).to(device=self.device)
        action_batch = torch.LongTensor(batch.agent_action).view(-1,1).to(device=self.device)
        reward_batch = torch.Tensor(batch.reward).to(device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.current_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        if self.parameter.get("dqn_type") == "DQN":
            next_state_values = self.next_state_values_DQN(batch_size=batch_size, non_final_mask=non_final_mask, non_final_next_states=non_final_next_states)
        elif self.parameter.get("dqn_type") == "DoubleDQN":
            next_state_values = self.next_state_values_double_DQN(batch_size=batch_size, non_final_mask=non_final_mask, non_final_next_states=non_final_next_states)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = torch.nn.functional.mse_loss(input=state_action_values,target=expected_state_action_values.view(-1, 1))

        # Optimize the model
        self.optimizer.zero_grad() # zero the gradients.
        loss.backward() # calculate the gradient.
        # for name, param in self.current_net.named_parameters():
        #     param.grad.data.clamp_(-1, 1) # gradient clipping
        self.optimizer.step()
        return {"loss":loss.item()}

    def next_state_values_DQN(self, batch_size, non_final_mask, non_final_next_states ):
        """
        Computate the values of all next states with DQN.
        `http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf`

        Args:
            batch_size (int): the size of given batch.
            non_final_mask (Tensor): shape: 1-D, [batch_size], 0: non-terminal state, 0: terminal state
            non_final_next_states (Tensor): 2-D, shape: [num_of_non_terminal_states, state_dim]

        Returns:
            A 1-D Tensor, shape:[batch_size]
        """
        # Compute V(s_{t+1}) for all next states.
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
            non_final_mask (Tensor): shape: 1-D, [batch_size], 0: non-terminal state, 0: terminal state
            non_final_next_states (Tensor): 2-D, shape: [num_of_non_terminal_states, state_dim]

        Returns:
            A 1-D Tensor, shape:[batch_size]
        """
        next_state_values = torch.zeros(batch_size).to(device=self.device)
        if non_final_next_states.size()[0] > 0:
            next_action_batch_current = self.current_net(non_final_next_states).max(1)[1].view(-1,1).detach()
            next_state_values[non_final_mask==0] = self.target_net(non_final_next_states).gather(1, next_action_batch_current).detach().view(-1)
        return next_state_values

    def predict(self, Xs, **kwargs):
        Xs = torch.Tensor(Xs).to(device=self.device)
        Ys = self.current_net(Xs)
        max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        return Ys, max_index[0]

    def _predict_target(self, Xs, params, **kwargs):
        Xs = torch.Tensor(Xs).to(device=self.device)
        Ys = self.current_net(Xs)
        max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        return Ys, max_index[0]

    def save_model(self, model_performance,episodes_index, checkpoint_path):
        """
        Saving the trained model.

        Args:
            model_performance (dict): the test result of the model, which contains different metrics.
            episodes_index (int): the current step of training. And this will be appended to the model name at the end.
            checkpoint_path (str): the directory that the model is going to save to. Default None.
        """
        if os.path.isdir(checkpoint_path) == False:
            os.mkdir(checkpoint_path)
        agent_id = self.parameter.get("agent_id")
        dqn_id = self.parameter.get("dqn_id")
        disease_number = self.parameter.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_wrong_disease = model_performance["average_wrong_disease"]
        model_file_name = os.path.join(checkpoint_path, "model_d" + str(disease_number) + "_agent" + str(agent_id) + "_dqn" + \
                          str(dqn_id) + "_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn)\
                          + "_wd" + str(average_wrong_disease) + "_e-" + str(episodes_index) + ".pkl")

        torch.save(self.current_net.state_dict(), model_file_name)

    def restore_model(self, saved_model):
        """
        Restoring the trained parameters for the model. Both current and target net are restored from the same parameter.

        Args:
            saved_model (str): the file name which is the trained model.
        """
        print("loading trained model")
        self.current_net.load_state_dict(torch.load(saved_model))
        self.target_net.load_state_dict(self.current_net.state_dict())

    def update_target_network(self):
        """
        Updating the target network with the parameters copyed from the current networks.
        """
        self.target_net.load_state_dict(self.current_net.state_dict())
