# -*- coding: utf8 -*-

import time
import os

def verify_params(params):
    dqn_type = params.get("dqn_type")
    if dqn_type not in ['DQN', 'DoubleDQN', 'DuelingDQN']:
        raise ValueError("dqn_type should be one of ['DQN', 'DoubleDQN','DuelingDQN']")

    return construct_info(params)

def construct_info(params):
    """
    Constructing a string which contains the primary super-parameters.
    Args:
        params: the super-parameter

    Returns:
        A dict, the updated parameter.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu"]
    gpu_str = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_str.replace(' ', '')
    if len(gpu_str.split(',')) > 1:
        params.setdefault("multi_GPUs",True)
    else:
        params.setdefault("multi_GPUs", False)

    agent_id = params.get("agent_id")
    lr = params.get("dqn_learning_rate")
    reward_for_success = params.get("reward_for_success")
    reward_for_fail = params.get("reward_for_fail")
    reward_for_not_come_yet = params.get("reward_for_not_come_yet")
    reward_for_inform_right_symptom = params.get("reward_for_inform_right_symptom")
    reward_for_repeated_action = params.get("reward_for_repeated_action")
    reward_for_reach_max_turn = params.get("reward_for_reach_max_turn")
    check_related_symptoms = params.get("check_related_symptoms")

    max_turn = params.get("max_turn")
    gamma = params.get("gamma")
    gamma_worker = params.get('gamma_worker')
    epsilon = params.get("epsilon")
    data_set_name = params.get("data_files")
    run_id = params.get('run_id')
    dqn_type = params["dqn_type"]
    weight_for_reward_shaping = params["weight_for_reward_shaping"]
    simulation_size = params["simulation_size"]
    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    info = run_time + \
           "_" + str(data_set_name) + \
           "_" + agent_id + \
           "_T" + str(max_turn) + \
           "_ss" + str(simulation_size) + \
           "_lr" + str(lr) + \
           "_RFS" + str(reward_for_success) + \
           "_RFF" + str(reward_for_fail) + \
           "_RFNCY" + str(reward_for_not_come_yet) + \
           "_RFIRS" + str(reward_for_inform_right_symptom) +\
           "_RFRA" + str(reward_for_repeated_action) +\
           "_RFRMT" + str(reward_for_reach_max_turn) +\
           "_gamma" + str(gamma) + \
           "_gammaW" + str(gamma_worker) + \
           "_epsilon" + str(epsilon) + \
           "_crs" + str(int(check_related_symptoms)) + \
           "_wfrs" + str(weight_for_reward_shaping) + \
           "_RID" + str(run_id)
    params['run_info'] = info
    checkpoint_path = params['save_model_path'] + dqn_type + "/checkpoint/" + info
    params["checkpoint_path"] = checkpoint_path

    performance_save_path = "./HRL/model/" + dqn_type + "/performance_new/"
    params["performance_save_path"] = performance_save_path

    visit_save_path = "./HRL/model/" + dqn_type + "/visit/"
    params["visit_save_path"] = visit_save_path

    return params