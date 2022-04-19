'''
数据处理模块
'''

import pickle
import json
import copy
import numpy as np

f_dxy = open('dxy_dataset/dxy_dataset/dxy_dialog_data_dialog_v2.pickle', 'rb')
f1 = open('dataset/dise_sym_num_dict.p', 'rb')
f2 = open('dataset/goal_dict_original.p', 'rb')
f3 = open('dataset/req_dise_sym_dict.p', 'rb')
f4 = open('dataset/test_goal_dict.p', 'rb')
f5 = open('dataset/train_goal_dict.p', 'rb')

f1_dxy = open('dataset_dxy/dise_sym_num_dict_dxy.p', 'rb')
f2_dxy = open('dataset_dxy/goal_dict_original_dxy.p', 'rb')
f3_dxy = open('dataset_dxy/req_dise_sym_dict_dxy.p', 'rb')
f4_dxy = open('dataset_dxy/test_goal_dict_dxy.p', 'rb')
f5_dxy = open('dataset_dxy/train_goal_dict_dxy.p', 'rb')

info = pickle.load(f2_dxy)
print(info)


'''
将丁香园数据转化为txt文件
'''
# file = open('dataset_dxy/dxy_dialog_data_dialog_v2.txt', 'w')
#
# for k,v in info.items():
#     file.write(str(k) + '@' + str(v) + '\n')
#
# file.close()


'''
统计f2的信息
'''

# for k in info.keys():
#     if k == 'test':
#         print(len(info[k]))
#
# count = 0
# for v in info.values():
#     count += len(v)
#
# print(count)


# count = 0
# for v in info.values():
#     for i in v:
#         for v1 in i.values():
#             if v1 == {'disease' : 'UNK'}:
#                 count += 1
# print(count)
'''
统计f4和f5的信息
'''
# print(len(info))
# count = 0
# for i in info:
#     for v in i.values():
#         if v == {'disease' : 'UNK'}:\
#             count += 1
# print(count)




'''
生成goal_dict_original_dxy.txt & test_goal_dict_dxy.txt & train_goal_dict_dxy.txt
'''
result_all = {}
train_list = []
test_list = []
all_list = []
temp = {}
count = 0
for i in info['train']:
    temp['request_slots'] = {'disease':'UNK'}
    temp['implicit_inform_slots'] = i['implicit_inform_slots']
    temp['explicit_inform_slots'] = i['explicit_inform_slots']
    temp['disease_tag'] = i['disease_tag']
    temp['consult_id'] = 'train-' + str(count)
    count += 1
    train_list.append(copy.deepcopy(temp))   # 注意：这里要深拷贝，不然最后结果全都一样

count = 0
for j in info['test']:
    temp['request_slots'] = {'disease':'UNK'}
    temp['implicit_inform_slots'] = j['implicit_inform_slots']
    temp['explicit_inform_slots'] = j['explicit_inform_slots']
    temp['disease_tag'] = j['disease_tag']
    temp['consult_id'] = 'test-' + str(count)
    test_list.append(copy.deepcopy(temp))
    count += 1

all_list = train_list + test_list


result_all['all'] = all_list
result_all['train'] = train_list
result_all['test'] = test_list
# 保存结果部分

file = open('dataset_dxy/goal_dict_original_dxy.txt', 'w')
js = json.dumps(result_all, ensure_ascii=False)  #防止中文乱码
file.write(js)
file.close()

file = open('dataset_dxy/test_goal_dict_dxy.txt', 'w')
js = json.dumps(test_list, ensure_ascii=False)
file.write(js)
file.close()

file = open('dataset_dxy/train_goal_dict_dxy.txt', 'w')
js = json.dumps(train_list, ensure_ascii=False)
file.write(js)
file.close()

file = open('dataset_dxy/goal_dict_original_dxy.p', 'wb')
pkl = pickle.dumps(result_all)
file.write(pkl)
file.close()

file = open('dataset_dxy/test_goal_dict_dxy.p', 'wb')
pkl = pickle.dumps(test_list)
file.write(pkl)
file.close()

file = open('dataset_dxy/train_goal_dict_dxy.p', 'wb')
pkl = pickle.dumps(train_list)
file.write(pkl)
file.close()






'''
file = open('dataset_dxy/daoru.txt', 'w')
temp_test = {'harden':13, 'curry':30}
js = json.dumps(temp_test)
file.write(js)
file.close()
'''

temp = {}
x = 0
l = []
for i in range(10):
    temp['a'] = x
    temp['b'] = x + 1
    temp['c'] = x + 2
    temp['d'] = x + 3
    l.append(copy.deepcopy(temp))
    x += 4




'''
对dise_sym_num_dict.p & req_dise_sym_dict.p进行调研   以及生成dise_sym_num_dict_dxy.p & req_dise_sym_dict.p文件
'''

num = 0 # 统计下各个symptom在各个disease中的数量
for i in info['all']:
    if i['disease_tag'] == '小儿消化不良':
        for key, value in i['explicit_inform_slots'].items():
            if key == '稀便' and value == True:
                num += 1
        for key, value in i['implicit_inform_slots'].items():
            if key == '稀便' and value == True:
                num += 1
print(num)




result = {'过敏性鼻炎':{}, '上呼吸道感染':{}, '肺炎':{}, '小儿手足口病':{}, '小儿腹泻':{}}
result1 = {'过敏性鼻炎':[], '上呼吸道感染':[], '肺炎':[], '小儿手足口病':[], '小儿腹泻':[]}
for i in info['train']:
    # 过敏性鼻炎
    if i['disease_tag'] == '过敏性鼻炎':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['过敏性鼻炎']: # 该symptom已经在result中
                result['过敏性鼻炎'][key] += 1
            elif value == True and bool(1-(key in result['过敏性鼻炎'])):
                result['过敏性鼻炎'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['过敏性鼻炎']: # 该symptom已经在result中
                result['过敏性鼻炎'][key] += 1
            elif value == True and bool(1-(key in result['过敏性鼻炎'])):
                result['过敏性鼻炎'][key] = 1
    # 上呼吸道感染
    elif i['disease_tag'] == '上呼吸道感染':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['上呼吸道感染']: # 该symptom已经在result中
                result['上呼吸道感染'][key] += 1
            elif value == True and bool(1-(key in result['上呼吸道感染'])):
                result['上呼吸道感染'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['上呼吸道感染']: # 该symptom已经在result中
                result['上呼吸道感染'][key] += 1
            elif value == True and bool(1-(key in result['上呼吸道感染'])):
                result['上呼吸道感染'][key] = 1
    # 肺炎
    elif i['disease_tag'] == '肺炎':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['肺炎']: # 该symptom已经在result中
                result['肺炎'][key] += 1
            elif value == True and bool(1-(key in result['肺炎'])):
                result['肺炎'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['肺炎']: # 该symptom已经在result中
                result['肺炎'][key] += 1
            elif value == True and bool(1-(key in result['肺炎'])):
                result['肺炎'][key] = 1
    # 小儿手足口病
    elif i['disease_tag'] == '小儿手足口病':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['小儿手足口病']: # 该symptom已经在result中
                result['小儿手足口病'][key] += 1
            elif value == True and bool(1-(key in result['小儿手足口病'])):
                result['小儿手足口病'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['小儿手足口病']: # 该symptom已经在result中
                result['小儿手足口病'][key] += 1
            elif value == True and bool(1-(key in result['小儿手足口病'])):
                result['小儿手足口病'][key] = 1
    # 小儿腹泻
    elif i['disease_tag'] == '小儿腹泻':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['小儿腹泻']: # 该symptom已经在result中
                result['小儿腹泻'][key] += 1
            elif value == True and bool(1-(key in result['小儿腹泻'])):
                result['小儿腹泻'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['小儿腹泻']: # 该symptom已经在result中
                result['小儿腹泻'][key] += 1
            elif value == True and bool(1-(key in result['小儿腹泻'])):
                result['小儿腹泻'][key] = 1



for i in info['test']:
    # 过敏性鼻炎
    if i['disease_tag'] == '过敏性鼻炎':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['过敏性鼻炎']: # 该symptom已经在result中
                result['过敏性鼻炎'][key] += 1
            elif value == True and bool(1-(key in result['过敏性鼻炎'])):
                result['过敏性鼻炎'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['过敏性鼻炎']: # 该symptom已经在result中
                result['过敏性鼻炎'][key] += 1
            elif value == True and bool(1-(key in result['过敏性鼻炎'])):
                result['过敏性鼻炎'][key] = 1
    # 上呼吸道感染
    elif i['disease_tag'] == '上呼吸道感染':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['上呼吸道感染']: # 该symptom已经在result中
                result['上呼吸道感染'][key] += 1
            elif value == True and bool(1-(key in result['上呼吸道感染'])):
                result['上呼吸道感染'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['上呼吸道感染']: # 该symptom已经在result中
                result['上呼吸道感染'][key] += 1
            elif value == True and bool(1-(key in result['上呼吸道感染'])):
                result['上呼吸道感染'][key] = 1
    # 肺炎
    elif i['disease_tag'] == '肺炎':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['肺炎']: # 该symptom已经在result中
                result['肺炎'][key] += 1
            elif value == True and bool(1-(key in result['肺炎'])):
                result['肺炎'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['肺炎']: # 该symptom已经在result中
                result['肺炎'][key] += 1
            elif value == True and bool(1-(key in result['肺炎'])):
                result['肺炎'][key] = 1
    # 小儿手足口病
    elif i['disease_tag'] == '小儿手足口病':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['小儿手足口病']: # 该symptom已经在result中
                result['小儿手足口病'][key] += 1
            elif value == True and bool(1-(key in result['小儿手足口病'])):
                result['小儿手足口病'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['小儿手足口病']: # 该symptom已经在result中
                result['小儿手足口病'][key] += 1
            elif value == True and bool(1-(key in result['小儿手足口病'])):
                result['小儿手足口病'][key] = 1
    # 小儿腹泻
    elif i['disease_tag'] == '小儿腹泻':
        for key, value in i['explicit_inform_slots'].items():
            if value == True and key in result['小儿腹泻']: # 该symptom已经在result中
                result['小儿腹泻'][key] += 1
            elif value == True and bool(1-(key in result['小儿腹泻'])):
                result['小儿腹泻'][key] = 1
        for key, value in i['implicit_inform_slots'].items():
            if value == True and key in result['小儿腹泻']: # 该symptom已经在result中
                result['小儿腹泻'][key] += 1
            elif value == True and bool(1-(key in result['小儿腹泻'])):
                result['小儿腹泻'][key] = 1

print(result)
#
temp_alergic_rhinitis = sorted(result['过敏性鼻炎'].items(), key = lambda item:item[1], reverse = True)
alergic_rhinitis = []
for i in temp_alergic_rhinitis:
    alergic_rhinitis.append(i[0])
alergic_rhinitis = alergic_rhinitis[0:10]
result1['过敏性鼻炎'] = alergic_rhinitis

temp_upper_res_inf = sorted(result['上呼吸道感染'].items(), key = lambda item:item[1], reverse = True)
upper_res_inf = []
for i in temp_upper_res_inf:
    upper_res_inf.append(i[0])
upper_res_inf = upper_res_inf[0:10]
result1['上呼吸道感染'] = upper_res_inf

temp_pneu = sorted(result['肺炎'].items(), key = lambda item:item[1], reverse = True)
pneu = []
for i in temp_pneu:
    pneu.append(i[0])
pneu = pneu[0:10]
result1['肺炎'] = pneu

temp_hand_foot_mouth = sorted(result['小儿手足口病'].items(), key = lambda item:item[1], reverse = True)
hand_foot_mouth = []
for i in temp_hand_foot_mouth:
    hand_foot_mouth.append(i[0])
hand_foot_mouth = hand_foot_mouth[0:10]
result1['小儿手足口病'] = hand_foot_mouth

temp_ped_dia = sorted(result['小儿腹泻'].items(), key = lambda item:item[1], reverse = True)
ped_dia = []
for i in temp_ped_dia:
    ped_dia.append(i[0])
ped_dia = ped_dia[0:10]
result1['小儿腹泻'] = ped_dia

print(result1)
#
# file = open('dataset_dxy/dise_sym_num_dict_dxy.txt', 'w')
# js = json.dumps(result, ensure_ascii=False)
# file.write(js)
# file.close()
#
# file = open('dataset_dxy/dise_sym_num_dict_dxy.p', 'wb')
# pkl = pickle.dumps(result)
# file.write(pkl)
# file.close()
#
#
# file = open('dataset_dxy/req_dise_sym_dict_dxy.txt', 'w')
# js = json.dumps(result1, ensure_ascii=False)
# file.write(js)
# file.close()
#
# file = open('dataset_dxy/req_dise_sym_dict_dxy.p', 'wb')
# pkl = pickle.dumps(result1)
# file.write(pkl)
# file.close()






'''
对dise_sym_pro.txt & sym_dise_pro.txt & sym_prio.txt调研     以及生成dise_sym_pro_dxy.txt & sym_dise_pro_dxy.txt & sym_prio_dxy.txt文件
'''
d1 = np.loadtxt('dataset/sym_prio.txt')
print(type(d1))
print(d1)

d2 = np.loadtxt('dataset/dise_sym_pro.txt')
d3 = np.loadtxt('dataset/sym_dise_pro.txt')
print(d2.shape, '\n', d3.shape)

d4 = open('dataset_dxy/symptoms_dxy.txt', 'r', encoding = 'utf-8')
sym = d4.readlines()
print(sym)


# # --------------------------------------------------------统计MZ病例中一共出现的症状数--------------------------------------------------
temp = {}
for i in info['all']:
    for key, value in i['explicit_inform_slots'].items():
        if value == True and key in temp:
            temp[key] += 1
        elif value == True and (1 - (key in temp)):
            temp[key] = 1

    for key, value in i['implicit_inform_slots'].items():
        if value == True and key in temp:
            temp[key] += 1
        elif value == True and (1 - (key in temp)):
            temp[key] = 1

print(temp)
temp1 = sorted(temp.items(), key = lambda item:item[1], reverse = True)
print(temp1)
temp2 = []
for i in temp1:
    temp2.append(i[0])
print(temp2)  # 得出结论：symptoms.txt文件中的症状并不是按出现次数的多少进行排序的
print(len(temp2))
x = 0
for i in temp2:
    temp2[x] = i + '\n'
    x += 1
print(temp2)

# file = open('dataset_dxy/symptoms_dxy.txt', 'w')
# file.writelines(temp2)
# file.close()


# # ----------------------------------------------查看sym_prio.txt以及生成sym_prio_dxy.txt------------------------------------------------
temp = {}
for i in info['all']:
    for key, value in i['explicit_inform_slots'].items():
        if value == True and key in temp:
            temp[key] += 1
        elif value == True and (1 - (key in temp)):
            temp[key] = 1

    for key, value in i['implicit_inform_slots'].items():
        if value == True and key in temp:
            temp[key] += 1
        elif value == True and (1 - (key in temp)):
            temp[key] = 1

print(temp)
print(len(info['all']))
sym_p = []
for i in sym:
    sym_p.append(temp[i[:-1]]/527)
sym_p = np.array(sym_p)
sym_p = sym_p.reshape((1, 41))
print(sym_p)
#
# np.savetxt('dataset_dxy/sym_prio_dxy.txt', sym_p, fmt = "%f", delimiter = ' ')




# -------------------------------对dise_sym_pro.txt和sym_dise_pro.txt进行调研 并且生成dise_sym_pro_dxy.txt和sym_dise_pro_dxy.txt---------------------

file = open('dataset_dxy/diseases_dxy.txt', 'r', encoding = 'utf-8')
dis = file.readlines()
file.close()
file = open('dataset_dxy/symptoms_dxy.txt', 'r', encoding = 'utf-8')
sym = file.readlines()
file.close()

p_dis_sym_pro = np.zeros((len(dis), len(sym)))
p_sym_dis_pro = np.zeros((len(dis), len(sym)))
# print(p_dis_sym_pro)
print(dis)
print(sym)
# print(dis.index('小儿消化不良\n'))

for i in info['all']:
    for key, value in i['explicit_inform_slots'].items():
        if value == True:
            p_dis_sym_pro[dis.index(i['disease_tag'] + '\n')][sym.index(key + '\n')] += 1
    for key, value in i['implicit_inform_slots'].items():
        if value == True:
            p_dis_sym_pro[dis.index(i['disease_tag'] + '\n')][sym.index(key + '\n')] += 1


p_sym_dis_pro = np.copy(p_dis_sym_pro)
#print(p_sym_dis_pro)

temp = {}
for i in info['all']:
    for key, value in i['explicit_inform_slots'].items():
        if value == True and key in temp:
            temp[key] += 1
        elif value == True and (1 - (key in temp)):
            temp[key] = 1

    for key, value in i['implicit_inform_slots'].items():
        if value == True and key in temp:
            temp[key] += 1
        elif value == True and (1 - (key in temp)):
            temp[key] = 1

print(temp)

for i in range(len(p_dis_sym_pro[0])):
    p_dis_sym_pro[:, i] = p_dis_sym_pro[:, i] / temp[sym[i][:-1]]
p_dis_sym_pro = np.transpose(p_dis_sym_pro)
print(p_dis_sym_pro)
# np.savetxt('dataset_dxy/sym_dise_pro_dxy.txt', p_dis_sym_pro, fmt = "%f", delimiter = ' ')
#
temp1 = {}
for i in info['all']:
    if i['disease_tag'] in temp1:
        temp1[i['disease_tag']] += 1
    elif 1-(i['disease_tag'] in temp1):
        temp1[i['disease_tag']] = 1


print(temp1)

for i in range(len(p_sym_dis_pro)):
    p_sym_dis_pro[i, :] = p_sym_dis_pro[i, :] / temp1[dis[i][:-1]]

print(p_sym_dis_pro)
# np.savetxt('dataset_dxy/dise_sym_pro_dxy.txt', p_sym_dis_pro, fmt = "%f", delimiter = ' ')



'''
调研action_mat.txt以及生成action_mat_dxy.txt文件
'''
mat = np.loadtxt('dataset/action_mat.txt')
print(mat.shape)
file = open('dataset/slot_set.txt', 'r', encoding = 'utf-8')
temp = file.readlines()
file.close()
print(len(temp))

# 思路是生成四个矩阵，最后拼接成一个大矩阵
file = open('dataset_dxy/diseases_dxy.txt', 'r', encoding = 'utf-8')
dis = file.readlines()
file.close()
file = open('dataset_dxy/symptoms_dxy.txt', 'r', encoding = 'utf-8')
sym = file.readlines()
file.close()
print(dis)
print(sym)

dis_dis = np.zeros((len(dis), len(dis)))
#print(dis_dis)
sym_dis = np.loadtxt('dataset_dxy/dise_sym_pro_dxy.txt')
#print(sym_dis)
dis_sym = np.loadtxt('dataset_dxy/sym_dise_pro_dxy.txt')
#print(dis_sym)

sym_sym = np.zeros((len(sym), len(sym)))
print(sym_sym)
sym_app = []
for i in info['all']:
    for k_e, v_e in i['explicit_inform_slots'].items():
        if v_e == True :#and (1 - (k_e in sym_app)):
            sym_app.append(k_e)
    for k_i, v_i in i['implicit_inform_slots'].items():
        if v_i == True :#and (1 - (k_i in sym_app)):
            sym_app.append(k_i)
    for i in sym_app:
        for j in sym_app:
            if i != j:
                sym_sym[sym.index(i + '\n')][sym.index(j + '\n')] += 1
    sym_app.clear()

print(sym_sym)

temp = {}
for i in info['all']:
    for key, value in i['explicit_inform_slots'].items():
        if value == True and key in temp:
            temp[key] += 1
        elif value == True and (1 - (key in temp)):
            temp[key] = 1

    for key, value in i['implicit_inform_slots'].items():
        if value == True and key in temp:
            temp[key] += 1
        elif value == True and (1 - (key in temp)):
            temp[key] = 1

print(temp)

for r in range(len(sym_sym)):
    sym_sym[r, :] = sym_sym[r, :] / temp[sym[r][:-1]]

print(sym_sym)

# 拼接矩阵
t_upper_sim = np.concatenate((dis_dis, sym_dis), axis = 1)
t_under_sim = np.concatenate((dis_sym, sym_sym), axis = 1)
action_mat_sim = np.concatenate((t_upper_sim, t_under_sim), axis = 0)
print(action_mat_sim)

# 二次拼接
t1 = np.eye(2, dtype = float)
t2 = np.zeros((2, len(dis) + len(sym)))
t3 = np.zeros((len(dis) + len(sym), 2))
t_upper = np.concatenate((t1, t2), axis = 1)
t_under = np.concatenate((t3, action_mat_sim), axis = 1)
action_mat = np.concatenate((t_upper, t_under), axis = 0)
print(action_mat)
# np.savetxt('dataset_dxy/action_mat_dxy.txt', action_mat, fmt = "%f", delimiter = ' ')



'''
调研slot_set和symtoms的区别
'''

# file = open('dataset_dxy/symptoms_dxy.txt', 'r')
# sym = file.readlines()
# file.close()



# file = open('dataset_dxy/diseases_dxy.txt', 'r')
# dis = file.readlines()
# file.close()
#
# temp = ['UNK\n']
#
# slot = dis + temp + sym
#
# file = open('dataset_dxy/slot_set_dxy.txt', 'w')
# file.writelines(slot)
# file.close()






