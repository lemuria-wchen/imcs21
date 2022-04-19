import numpy as np

import torch
import torch.optim as optim

from .BaseAgent import BaseAgent
from OpenMedicalChatBox.KRDS.qlearning.dqn_prior import KR_DQN
from OpenMedicalChatBox.KRDS.utils.ReplayMemory import ExperienceReplayMemory, MutExperienceReplayMemory, PrioritizedReplayMemory, MutPrioritizedReplayMemory
import OpenMedicalChatBox.KRDS.dialog_config as dialog_config
import copy


class AgentDQN(BaseAgent):
    def __init__(self, sym_dict=None, dise_dict=None, req_dise_sym_dict=None, dise_sym_num_dict=None,
                 tran_mat=None, dise_sym_pro=None, sym_dise_pro=None, sym_prio = None, act_set=None, slot_set=None, params=None, static_policy=False):
        super(AgentDQN, self).__init__()
        self.device = torch.device("cuda:" + params.get('device', 0) if torch.cuda.is_available() else "cpu")
        # parameters for DQN
        self.noisy = params.get('noisy', False)
        self.priority_replay = False
        self.gamma = params.get('gamma', 0.9)
        self.lr = params.get('lr', 0.01)
        self.batch_size = params.get('batch_size', 30)
        self.target_net_update_freq = params.get('target_net_update_freq', 1)
        self.experience_replay_size = params.get('experience_replay_size', 10000)
        # self.learn_start = params.get('learn_start', 1)
        self.sigma_init = params.get('sigma_init', 0.5)
        self.priority_beta_start = params.get('priority_beta_start', 0.4)
        self.priority_beta_frames = params.get('priority_beta_frames', 10000)
        self.priority_alpha = params.get('priority_alpha', 0.6)
        self.dqn_hidden_size = params.get('dqn_hidden_size', 128)
        self.static_policy = static_policy

        self.predict_mode = False
        self.warm_start = params['warm_start']
        self.max_turn = params['max_turn']
        self.epsilon = params.get('epsilon', 0.1)
        self.sym_dict = sym_dict  # all symptoms
        self.dise_dict = dise_dict
        self.req_dise_sym_dict = req_dise_sym_dict  # high freq dise sym relations
        self.dise_sym_num_dict = dise_sym_num_dict  # dise sym discrete
        self.tran_mat_flag = torch.from_numpy(np.where(tran_mat>0,1.0,0.0)).float().to(self.device)
        self.tran_mat = torch.from_numpy(tran_mat).float().to(self.device)
        self.dise_num = len(dise_dict.keys())
        self.sym_dise_pro = torch.from_numpy(sym_dise_pro).float().to(self.device)
        self.dise_sym_pro = torch.from_numpy(dise_sym_pro).float().to(self.device)
        self.sym_prio = torch.from_numpy(sym_prio).float().to(self.device)
        self.act_set = act_set  # all acts
        self.slot_set = slot_set  # slot for agent
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys()) # add five disease slot
        self.feasible_actions = params['feasible_actions']  # actions
        self.num_actions = len(self.feasible_actions)
        self.state_dimension = 2 * self.act_cardinality + 1 * self.slot_cardinality + self.max_turn
        self.dise_start = 2
        
        self.sym_start = self.dise_start + self.dise_num

        self.declare_networks(params['trained_model_path'], params['mode'])
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.001)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=0.001)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        '''
        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()
        '''
        self.model.train()
        self.target_model.eval()
        self.update_count = 0

        self.declare_memory()        
        self.nsteps = params.get('nsteps', 1)
        self.nstep_buffer = []

        self.request_set = copy.deepcopy(params['sys_request_slots_highfreq'])
        self.current_slots = {}

    def initialize_episode(self):
        self.current_slots = {}

    def declare_networks(self, path, mode):
        
        self.model = KR_DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions, self.tran_mat, self.dise_start, self.act_cardinality, self.slot_cardinality, self.sym_dise_pro, self.dise_sym_pro, self.sym_prio, self.device)
        self.target_model = KR_DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions, self.tran_mat, self.dise_start, self.act_cardinality, self.slot_cardinality, self.sym_dise_pro, self.dise_sym_pro, self.sym_prio, self.device)
        if path is not None and not(mode):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.predict_mode = True
            self.warm_start = 2

    def load_specific_state_dict(self,path):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def declare_memory(self):
        self.memory = MutExperienceReplayMemory(self.experience_replay_size)

    def prepare_state_representation(self, state):
        user_action = state['user_action']
        current_slots = state['current_slots']
        #print('current_slots', current_slots)
        agent_last = state['agent_action']
        # user action
        user_act_rep = torch.zeros(1, self.act_cardinality, device=self.device)
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0
        # user inform slots
        # user_inform_slots_rep = torch.zeros(1, self.slot_cardinality)
        # for slot in user_action['inform_slots'].keys():
        #     if slot not in self.slot_set:
        #         continue
        #     user_inform_slots_rep[0, self.slot_set[slot]] = user_action['inform_slots'][slot]
        # current slots
        current_slots_rep = torch.zeros(1, self.slot_cardinality, device=self.device)
        for slot in current_slots['inform_slots']:
            if slot not in self.slot_set:
                continue
            # if current_slots['inform_slots'][slot] == -2:
            #     current_slots_rep[0, self.slot_set[slot]] = 0.5
            # else:
            #     current_slots_rep[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]
           
            # if slot is disease, slot position =1, if slot is sym, slot position = slot val
            if slot == 'disease': 
                current_slots_rep[0, self.slot_set[slot]] = 1
            else:
                current_slots_rep[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]

        # agent action
        agent_act_rep = torch.zeros(1, self.act_cardinality, device=self.device)
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0
        # agent request slots
        # agent_request_slots_rep = torch.zeros(1, self.slot_cardinality).cuda()
        # if agent_last:
        #     for slot in agent_last['request_slots'].keys():
        #         if slot not in self.slot_set:
        #             continue
        #         agent_request_slots_rep[0, self.slot_set[slot]] = 1.0
        # turn info
        turn_onehot_rep = torch.zeros(1, self.max_turn, device=self.device)
        turn_onehot_rep[0, state['turn']] = 1.0
        final_representation = torch.cat((user_act_rep, agent_act_rep, current_slots_rep, turn_onehot_rep), 1)

        return final_representation

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)

        # only record experience of dqn train, and warm start
        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                # self.experience_replay_pool.append(training_example)
                self.memory.push(training_example)
        else:  # Prediction Mode
            # self.experience_replay_pool.append(training_example)
            self.memory.push(training_example)
    
        
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over = zip(*transitions)
        neg_batch_episode_over = []
        for i in range(self.batch_size):
            if batch_episode_over[i]:
                neg_batch_episode_over.append(0)
            else:
                neg_batch_episode_over.append(1)
        
        batch_state = torch.cat(batch_state, dim=0).view(-1, self.state_dimension)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.cat(batch_next_state, dim=0).view(-1, self.state_dimension)
        batch_episode_over = torch.tensor(neg_batch_episode_over, device=self.device, dtype=torch.float).squeeze().view(-1, 1)  # flag for existence of next state

        return batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights

    def get_sym_flag(self, batch_state):
        bs = batch_state.size(0)
        
        ones = torch.ones(batch_state.size()).to(self.device)
        zeros = torch.zeros(batch_state.size()).to(self.device)
        return torch.cat((torch.ones(bs, self.sym_start).to(self.device),torch.where(batch_state == 0, ones, zeros)),1)

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights = batch_vars

        # estimate
        # self.model.sample_noise()
        batch_state_flag = self.get_sym_flag(batch_state[:,(2*self.act_cardinality+self.dise_sym_pro.size(0)+1):(2*self.act_cardinality+self.slot_cardinality)])
        current_q_values = self.model(batch_state, batch_state_flag).gather(1, batch_action)   # get the output values of the groundtruth batch_action
        # target
        with torch.no_grad():
            # self.target_model.sample_noise()
            batch_next_state_flag = self.get_sym_flag(batch_next_state[:,(2*self.act_cardinality+self.dise_sym_pro.size(0)+1):(2*self.act_cardinality+self.slot_cardinality)])
            max_next_q_values = self.target_model(batch_next_state,batch_next_state_flag).max(dim=1)[0].view(-1, 1)  # max q value
            expected_q_values = batch_reward + batch_episode_over*(self.gamma ** self.nsteps)*max_next_q_values
        diff = (expected_q_values - current_q_values)
        loss = 0.5*torch.pow(diff,2) 
        loss = loss.mean()

        return loss


    def single_batch(self):
        batch_vars = self.prep_minibatch()
        loss = self.compute_loss(batch_vars)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.model.parameters():
        #    if param.requires_grad == True:    
        #        param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # self.update_target_model()
        return loss.item()

    def train(self):
        cur_bellman_err = 0.0
        for iter in range(int(len(self.memory)/self.batch_size)):
            cur_bellman_err += self.single_batch()
        print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) * self.batch_size / len(self.memory), len(self.memory)))
        #print(self.model.tran_mat)
        self.update_target_model()

    def run_policy(self, rep, state):
        act_ind = -1
        if np.random.random() < self.epsilon:
            act_ind = np.random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.memory.buffer) > self.experience_replay_size:
                    self.warm_start = 2
                act_ind = self.rule_policy(state)
            else:
                sym_flag = self.get_sym_flag(rep[:, (2*self.act_cardinality+self.dise_sym_pro.size(0)+1):(2*self.act_cardinality+self.slot_cardinality)])
                act_ind = self.model.predict(rep, sym_flag)
        repeat = self.detect_repeat(state, act_ind)
        return act_ind, repeat

    def detect_repeat(self, state, act_ind):
        current_inform_slot = state['current_slots']['inform_slots'] 
        #print(current_inform_slot)
        #current_inform_slot.update(state['current_slots']['agent_request_slots'] )
        repeat = 0
        action = self.feasible_actions[act_ind]
        #print(action)
        if action['diaact'] == 'request' and list(action['request_slots'].keys())[0] in current_inform_slot.keys():
            repeat = 1
        #print(repeat)
        return repeat
 
    def disease_from_dict(self, current_slots, sym_flag):

        if sym_flag == 0:
            dise = dialog_config.NO_MATCH
            for d in self.req_dise_sym_dict:
                dise = d
                for sym in self.req_dise_sym_dict[d]:
                    if sym not in current_slots['inform_slots'] or current_slots['inform_slots'][sym] != True:
                        dise = dialog_config.NO_MATCH
                if dise != dialog_config.NO_MATCH:
                    return dise
            return dise
        else:
            dise = dialog_config.NO_MATCH_BY_RATE
            max_sym_rate = 0.0
            for d in self.dise_sym_num_dict:
                tmp = [v for v in self.dise_sym_num_dict[d].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                cur_dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[d][sym]
                for sym in self.dise_sym_num_dict[d]:
                    cur_dise_sym_sum += self.dise_sym_num_dict[d][sym]
                # tmp_rate = float(len(tmp))/float(len(self.req_dise_sym_dict[dise]))
                tmp_rate = float(tmp_sum) / float(cur_dise_sym_sum)
                if tmp_rate > max_sym_rate:
                    max_sym_rate = tmp_rate
                    dise = d
            return dise

    def rule_policy(self, state):
        """ Rule Policy """
        current_slots = state['current_slots']
        act_slot_response = {}
        sym_flag = 1  # 1 for no left sym, 0 for still have
        for sym in self.request_set:
            if sym not in current_slots['inform_slots'].keys():
                sym_flag = 0
        dise = self.disease_from_dict(current_slots, sym_flag)
        if dise == dialog_config.NO_MATCH:  # no match but still have syms to ask
            cur_dise_sym_rate = {}
            for dise in self.dise_sym_num_dict:
                if dise not in cur_dise_sym_rate:
                    cur_dise_sym_rate[dise] = 0
                tmp = [v for v in self.dise_sym_num_dict[dise].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[dise][sym]
                for sym in self.dise_sym_num_dict[dise]:
                    dise_sym_sum += self.dise_sym_num_dict[dise][sym]
                # dise_sym_rate[dise] = float(len(tmp))/float(len(self.dise_sym_num_dict[dise]))
                    cur_dise_sym_rate[dise] = float(tmp_sum) / float(dise_sym_sum)

            sorted_dise = list(dict(sorted(cur_dise_sym_rate.items(), key=lambda d: d[1], reverse=True)).keys())
            left_set = []
            for i in range(len(sorted_dise)):
                max_dise = sorted_dise[i]
                left_set = [v for v in self.req_dise_sym_dict[max_dise] if v not in current_slots['inform_slots'].keys()]
                if len(left_set) > 0: break
            # if syms in request set of all disease have been asked, choose one sym in request set
            if len(left_set) == 0:
                print('this will not happen')
                left_set = [v for v in self.request_set if v not in current_slots['inform_slots'].keys()]
            slot = np.random.choice(left_set)
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}

        elif dise == dialog_config.NO_MATCH_BY_RATE: # no match and no sym to ask
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': 'UNK', 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {},

        else:  # match one dise by complete match or by rate
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': dise, 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}

        return self.action_index(act_slot_response)

    def state_to_action(self, state):
        self.representation = self.prepare_state_representation(state)
        self.action, repeat = self.run_policy(self.representation, state)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}, repeat

    def action_index(self, act_slot_response):
        """ Return the index of action """
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception("action index not found")
        return None

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("update target model!!!")
            self.target_model.load_state_dict(self.model.state_dict())

    # get the action index with the max action values
    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass
