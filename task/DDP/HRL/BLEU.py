from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

import pickle
file0 = '/remote-home/czhong/RL/data/dxy_dataset/dxy_dataset/100symptoms/HRL/'
goal_set = file0 + '/goal_set.p' 
pred_path = '/remote-home/czhong/RL/log/MeicalChatbot-HRL-master/MeicalChatbot-HRL-master/HRL/dialogue_system/records/1113093224_record.p'

goal_set = pickle.load(open(goal_set,'rb'))['test']

goal_action_dict = dict()
pred_action_dict = pickle.load(open(pred_path,'rb'))
for record in goal_set:
    goal_action_dict[record['consult_id']] = list(record['goal']['implicit_inform_slots'])
    
 

gold_actions = [['a', 'b'], ['a', 'b', 'c']]
pred_actions = [['a', 'c'], ['a', 'd', 'c']]

pred = list()
goal = list()
for key, values in pred_action_dict.items():
    pred_record = list()
    for value in values:
        if value in goal_action_dict[key]:
            pred_record.append(value)
    goal.append(goal_action_dict[key]) 
    pred.append(pred_record)

pred_actions = pred
gold_actions = goal

pred_actions2 = [[act for act in pred_act if act in gold_act] for gold_act, pred_act in zip(gold_actions, pred_actions)]


bleu1 = corpus_bleu([[gold_action] for gold_action in gold_actions], pred_actions2, smoothing_function=SmoothingFunction().method1, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu([[gold_action] for gold_action in gold_actions], pred_actions2, smoothing_function=SmoothingFunction().method1, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu([[gold_action] for gold_action in gold_actions], pred_actions2, smoothing_function=SmoothingFunction().method1, weights=(1/3, 1/3, 1/3, 0))
bleu4 = corpus_bleu([[gold_action] for gold_action in gold_actions], pred_actions2, smoothing_function=SmoothingFunction().method1, weights=(0.25, 0.25, 0.25, 0.25))
print(bleu1, bleu2, bleu3, bleu4)

eps=1e-3
jd = [ (len(set(gold_act).intersection(set(pred_act))) + eps) / (len(set(gold_act).union(set(pred_act))) + eps) for gold_act, pred_act in zip(gold_actions, pred_actions)]
print(sum(jd)/len(jd))