import pickle
import copy
import numpy as np
import torch

sys_inform_slots = ['disease']

start_dia_acts = {
    'request': ['disease']
}
#path = '/remote-home/czhong/RL/data/data/dataset/label/allsymptoms/dataset_dxy/'
#path = '/remote-home/czhong/RL/data/new_data/mz10/allsymptoms/dataset_dxy/'
#path = '/remote-home/czhong/RL/Dialogue-System-for-Automatic-Diagnosis-master/dataset_dxy/'

# muzhi

# sys_inform_slots_values = ['上呼吸道感染', '小儿支气管炎', '小儿腹泻', '小儿消化不良']
# sys_request_slots_highfreq = ['发热', '咳嗽', '鼻流涕', '普通感冒', '中等度热', '有痰', '鼻塞', '低热', '喷嚏', '呕吐', '支气管炎', '痰鸣音', '咳痰', '急性气管支气管炎', '腹泻', '稀便', '水样便', '消化不良', '绿便', '血便', '大便粘液', '屁', '哭闹', '厌食']
# sys_request_slots = ['普通感冒', '干咳', '咳嗽', '厌食', '发热', '上呼吸道感染', '中等度热', '出汗', '高热', '头痛', '咽喉不适', '低热', '呕吐', '精神软', '鼻流涕', '喷嚏', '鼻塞', '四肢厥冷', '急性气管支气管炎', '咳痰', '稀便', '食欲不佳', '腹痛', '恶心', '干呕', '肠炎', '过敏', '有痰', '痰鸣音', '扁桃体炎', '退热', '支气管炎', '大便酸臭', '消化不良', '腹泻', '气管炎', '肺炎', '血便', '皮疹', '咽喉炎', '喘息', '水样便', '食欲不振', '绿便', '肛门红肿', '支气管肺炎', '口臭', '哭闹', '湿疹', '鼻炎', '病毒感染', '睡眠障碍', '反复发热', '嗜睡', '便秘', '贫血', '大便粘液', '粗糙呼吸音', '腹胀', '屁', '沙哑', '细菌感染', '尿量减少', '腹部不适', '肠鸣音', '支原体感染']

#print(len(sys_request_slots))
################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0
WRONG_DISEASE = 2
# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0


################################################################################
#  Diagnosis
################################################################################
NO_DECIDE = 0
NO_MATCH = "no match"
NO_MATCH_BY_RATE = "no match by rate"

################################################################################
#  Special Slot Values
################################################################################
I_AM_NOT_SURE = -1
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"

################################################################################
#  Slot Values
################################################################################
TRUE = 1
FALSE = -1
NOT_SURE = -2
NOT_MENTION = 0

################################################################################
#  Constraint Check
################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  NLG Beam Search
################################################################################
nlg_beam_size = 10


