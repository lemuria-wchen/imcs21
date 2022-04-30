# -*- coding:utf-8 -*-

import torch
import random
from collections import deque
import sys, os
sys.path.append(os.getcwd().replace("HRL/dialogue_system/policy_learning",""))
from HRL.dialogue_system.policy_learning.dqn_torch import DQN
from collections import namedtuple


class DQNModelWithGoal(torch.nn.Module):
    """
    The model in this file is reference to `Florensa, C., Duan, Y., & Abbeel, P. (2017). Stochastic neural networks for
    hierarchical reinforcement learning. arXiv preprint arXiv:1704.03012.`
    https://arxiv.org/abs/1704.03012
    """
    def __init__(self, input_size, hidden_size, output_size, number_of_latent_variables, parameter):
        super(DQNModelWithGoal, self).__init__()
        self.params = parameter
        self.number_of_latent_variables = number_of_latent_variables
        # different layers
        self.goal_layer1 = torch.nn.Linear(input_size, number_of_latent_variables, bias=True)
        self.policy_layer1 = torch.nn.Linear(input_size + number_of_latent_variables, output_size, bias=True)

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        goal = self.goal_generator(x)
        # print(goal)
        q_values = self.compute_q_value(x,goal)
        return q_values

    def goal_generator(self, x):
        logits = self.goal_layer1(x)
        goal_rep = torch.nn.functional.gumbel_softmax(logits=logits, tau=self.tau, hard=False)
        # goal_rep = torch.nn.functional.softmax(input=logits)
        return goal_rep

    def compute_q_value(self, x, goal):
        temp = torch.cat((x, goal), dim=1)
        q_values = self.policy_layer1(temp)
        return q_values


class DQNModelWithGoal2(torch.nn.Module):
    """
    The model in this file is reference to `Florensa, C., Duan, Y., & Abbeel, P. (2017). Stochastic neural networks for
    hierarchical reinforcement learning. arXiv preprint arXiv:1704.03012.`
    https://arxiv.org/abs/1704.03012
    """
    def __init__(self, input_size, hidden_size, output_size, number_of_latent_variables, parameter):
        super(DQNModelWithGoal2, self).__init__()
        self.params = parameter
        self.number_of_latent_variables = number_of_latent_variables
        # different layers
        self.goal_input_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.goal_state_abstract_layer = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.goal_generate_layer = torch.nn.Linear(hidden_size, number_of_latent_variables, bias=True)

        self.policy_layer1 = torch.nn.Linear(hidden_size * number_of_latent_variables, output_size, bias=True)

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        batch_size = x.size()[0]
        goal, abstract_state = self.goal_generator(x)
        # print(goal)
        temp = torch.bmm(goal.unsqueeze(2), abstract_state.unsqueeze(1))
        temp = temp.view(batch_size, -1)
        q_values = self.compute_q_value(temp)
        return q_values

    def goal_generator(self, x):
        h1 = self.goal_input_layer(x)
        h_state = self.goal_state_abstract_layer(torch.nn.functional.relu(h1))
        goal_logits = self.goal_generate_layer(torch.nn.functional.relu(h1))
        goal_rep = torch.nn.functional.gumbel_softmax(logits=goal_logits, tau=self.tau, hard=True)
        # goal_rep = torch.nn.functional.softmax(input=logits)
        return goal_rep, h_state

    def compute_q_value(self, x):
        q_values = self.policy_layer1(x)
        return q_values


class DQNModelWithGoal3(torch.nn.Module):
    """
    The model in this file is reference to `Florensa, C., Duan, Y., & Abbeel, P. (2017). Stochastic neural networks for
    hierarchical reinforcement learning. arXiv preprint arXiv:1704.03012.`
    https://arxiv.org/abs/1704.03012
    """
    def __init__(self, input_size, hidden_size, output_size, number_of_latent_variables, parameter):
        super(DQNModelWithGoal3, self).__init__()
        self.params = parameter
        self.number_of_latent_variables = number_of_latent_variables
        # different layers
        self.goal_input_layer = torch.nn.Linear(input_size, number_of_latent_variables, bias=True)

        self.policy_layer1 = torch.nn.Linear(input_size * number_of_latent_variables, output_size, bias=True)

    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        batch_size = x.size()[0]
        goal = self.goal_generator(x)
        # print(goal)
        temp = torch.bmm(goal.unsqueeze(2), x.unsqueeze(1))
        temp = temp.view(batch_size, -1)
        q_values = self.compute_q_value(temp)
        return q_values

    def goal_generator(self, x):
        logits = self.goal_input_layer(x)
        goal_rep = torch.nn.functional.gumbel_softmax(logits=logits, tau=self.tau, hard=True)
        # goal_rep = torch.nn.functional.softmax(input=logits)
        return goal_rep

    def compute_q_value(self, x):
        q_values = self.policy_layer1(x)
        return q_values


class DQNWithGoal(DQN):
    def __init__(self, input_size, hidden_size, output_size, parameter):
        super(DQNWithGoal, self).__init__(input_size, hidden_size, output_size, parameter)
        del self.current_net
        del self.target_net

        self.params = parameter
        self.Transition = namedtuple('Transition', ('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.current_net = DQNModelWithGoal(input_size, hidden_size, output_size, 4, parameter).to(self.device)
        self.target_net = DQNModelWithGoal(input_size, hidden_size, output_size, 4, parameter).to(self.device)
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
        ], lr=self.params.get("dqn_learning_rate",0.001))

        if self.params.get("train_mode") is False:
            self.restore_model(self.params.get("saved_model"))