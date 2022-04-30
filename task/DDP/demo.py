import OpenMedicalChatBox as OMCB
from warnings import simplefilter
import os
import sys
import argparse


simplefilter(action='ignore', category=FutureWarning)
os.chdir(os.path.dirname(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest="dataset", type=str, default=None, help="The dataset:(mz4, mz10, dxy, simul)")
parser.add_argument("--method", dest="method", type=str, default=None,
                    help="The method:(HRL, FlatDQN, REFUEL, KRDS, GAMP) ")
args = parser.parse_args()
parameters = vars(args)

print(parameters.get("dataset"))
print(parameters.get("method"))
if parameters.get("dataset") == 'mz4':
    dataset_path = './/OpenMedicalChatBox/Data/mz4/HRL//'
    groups = 2
elif parameters.get("dataset") == 'mz10':
    dataset_path = './/OpenMedicalChatBox/Data/mz10/HRL//'
    groups = 4
elif parameters.get("dataset") == 'dxy':
    dataset_path = './/OpenMedicalChatBox/Data/dxy_dataset/HRL//'
    groups = 3
elif parameters.get("dataset") == 'simul':
    dataset_path = './/OpenMedicalChatBox/Data/Fudan-Medical-Dialogue2.0/synthetic_dataset/HRL//'
else:
    print(parameters.get("dataset"))
    raise ValueError('Dataset should be one of [mz4, mz10, dxy, simul].')

if parameters.get("method") == 'HRL':
    HRL_test = OMCB.HRL(dataset_path=dataset_path, model_save_path='./simulate//', groups=groups,
                        model_load_path='./simulate/DQN/checkpoint/0411092858_MZ-10_agenthrljoint2_T20_ss100_lr0.0005_RFS20_RFF0_RFNCY0_RFIRS30_RFRA-4_RFRMT-100_gamma1_gammaW0.9_epsilon0.1_crs0_wfrs1_RID0/model_d10agenthrljoint2_s0.299_r-20.951_t9.5_mr0.007_mr2-0.004_e-0.pkl',
                        cuda_idx=0, train_mode=False)
    HRL_test.run()
elif parameters.get("method") == 'KRDS':
    dataset_path.replace('HRL', 'dataset_dxy')
    KRDS_test = OMCB.KRDS(dataset_path=dataset_path, model_save_path='./simulate//', model_load_path=None, cuda_idx=0,
                          warm_start=1, train_mode=True)
    KRDS_test.run()
elif parameters.get("method") == 'FlatDQN':
    Flat_DQN_test = OMCB.Flat_DQN(dataset_path=dataset_path, model_save_path='./simulate//',
                                  model_load_path='/remote-home/czhong/RL/DISCOpen-MedBox/simulate/DQN/checkpoint/0411114102_MZ-10_agentdqn_T20_ss100_lr0.0005_RFS20_RFF0_RFNCY0_RFIRS6_RFRA-4_RFRMT-100_gamma1_gammaW0.9_epsilon0.1_crs0_wfrs1_RID0/model_d10agentdqn_s0.299_r6.417_t2.5_mr0.024_mr2-0.014_e-2.pkl',
                                  cuda_idx=0, warm_start=True, train_mode=True)
    Flat_DQN_test.run()
elif parameters.get("method") == 'GAMP':
    GAMP_test = OMCB.GAMP(dataset_path=dataset_path, model_save_path='./simulate//',
                          model_load_path='./simulate/0411125423/s0.612_obj2.652_t2.954_mr0.107_outs0.183_e-0',
                          cuda_idx=0, train_mode=True)
    GAMP_test.run()
elif parameters.get("method") == 'REFUEL':
    REFUEL_test = OMCB.REFUEL(dataset_path=dataset_path, model_save_path='./simulate//',
                              model_load_path='./simulate/0411132328/s9.043_obj-16.433_t1.0_mr0.0_outs0.0_e-1.pkl',
                              cuda_idx=0, train_mode=True)
    REFUEL_test.run()
else:
    print(parameters.get("method"))
    raise ValueError('Method should be one of [HRL, FlatDQN, REFUEL, KRDS, GAMP].')
