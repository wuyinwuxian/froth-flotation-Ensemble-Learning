import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from confusion import ConfusionMatrix
import os
import json

x_hue_range  = np.arange(0.06, 0.15, 0.001, np.float32)
x_blue_range = np.arange(80, 160, 0.01, np.float32)
x_red_range  = np.arange(0.95, 1.5, 0.01, np.float32)
x_coarseness_range = np.arange(3, 12, 0.01, np.float32)
x_energy_range = np.arange(7000000, 14000000, 1, np.float32)
y_class_range  = np.arange(0, 10, 1, np.float32)

# 创建模糊控制变量
x_hue  = ctrl.Antecedent(x_hue_range, 'hue')
x_blue = ctrl.Antecedent(x_blue_range, 'blue')
x_red  = ctrl.Antecedent(x_red_range, 'red')
x_coarseness = ctrl.Antecedent(x_coarseness_range, 'coarseness')
x_energy = ctrl.Antecedent(x_energy_range, 'energy')
y_class  = ctrl.Consequent(y_class_range, 'class')

# 定义模糊集和其隶属度函数
x_hue['L']  = fuzz.trimf(x_hue_range,[0.06,0.065,0.065])
x_hue['N']  = fuzz.trimf(x_hue_range,[0.065,0.075,0.08])
x_hue['H']  = fuzz.trimf(x_hue_range,[0.08,0.085,0.095])
x_hue['VH'] = fuzz.trimf(x_hue_range,[0.09,0.095,0.15])

x_blue['L']   = fuzz.trimf(x_blue_range,[80,85,92])
x_blue['N']   = fuzz.trimf(x_blue_range,[90,110,120])
x_blue['H']   = fuzz.trimf(x_blue_range,[115,120,130])
x_blue['VH']  = fuzz.trimf(x_blue_range,[120,130,140])
x_blue['VVH'] = fuzz.trimf(x_blue_range,[140,140,160])

x_red['L']  = fuzz.trimf(x_red_range,[0.97,0.97,0.98])
x_red['N']  = fuzz.trimf(x_red_range,[0.98,0.98,0.99])
x_red['H']  = fuzz.trimf(x_red_range,[0.99,0.99,1])
x_red['VH'] = fuzz.trimf(x_red_range,[1,1,1.1])

x_coarseness['L']  = fuzz.trimf(x_coarseness_range,[3,3,4])
x_coarseness['N']  = fuzz.trimf(x_coarseness_range,[4,4,6])
x_coarseness['H']  = fuzz.trimf(x_coarseness_range,[6,6,8])
x_coarseness['VH'] = fuzz.trimf(x_coarseness_range,[8,9,11])

x_energy['L']  = fuzz.trimf(x_energy_range,[7000000,7000000,9000000])
x_energy['N']  = fuzz.trimf(x_energy_range,[8000000,9000000,10000000])
x_energy['H']  = fuzz.trimf(x_energy_range,[10000000,11000000,13000000])
x_energy['VH'] = fuzz.trimf(x_energy_range,[13000000,13000000,14000000])

y_class['1'] = fuzz.trimf(y_class_range,[0,1,2])
y_class['2'] = fuzz.trimf(y_class_range,[1,2,3])
y_class['3'] = fuzz.trimf(y_class_range,[2,3,4])
y_class['4'] = fuzz.trimf(y_class_range,[3,4,5])
y_class['5'] = fuzz.trimf(y_class_range,[4,5,6])
y_class['6'] = fuzz.trimf(y_class_range,[5,6,7])
y_class['7'] = fuzz.trimf(y_class_range,[6,7,8])
y_class['8'] = fuzz.trimf(y_class_range,[7,8,9])

# 设定输出powder的解模糊方法——质心解模糊方式

rule1 = ctrl.Rule(antecedent=((x_hue['N'] & x_blue['L']& x_energy['VH']) | x_hue['L'] | x_red['VH'] | x_energy['VH']), consequent=y_class['1'], label='rule 1')   # 输出为1的规则
rule2 = ctrl.Rule(antecedent=((x_hue['N'] & x_blue['N']) | (x_blue['H'] & x_hue['H']& x_energy['N'])),consequent=y_class['2'], label='rule 2')                    # 输出为2的规则
rule3 = ctrl.Rule(antecedent=((x_hue['H'] & x_blue['N']) | (x_hue['H'] & x_blue['H'])),consequent=y_class['3'], label='rule 3')                                   # 输出为3的规则
rule4 = ctrl.Rule(antecedent=((x_hue['H'] & x_blue['N'] & x_coarseness['N'] & x_energy['H']) ),consequent=y_class['4'], label='rule 4')                           # 输出为4的规则
rule5 = ctrl.Rule(antecedent=((x_hue['VH'] & x_blue['N']) | (x_coarseness['N'] & x_energy['H'])),consequent=y_class['5'], label='rule 5')                         # 输出为5的规则
rule6 = ctrl.Rule(antecedent=((x_hue['VH'] & x_blue['H'] & x_red['N']) | (x_coarseness['H'] & x_energy['N'])),consequent=y_class['6'], label='rule 6')            # 输出为6的规则
rule7 = ctrl.Rule(antecedent=((x_hue['VH'] & x_blue['VH']) | (x_coarseness['H'] & x_energy['L'])),consequent=y_class['7'], label='rule 7')                        # 输出为7的规则
rule8 = ctrl.Rule(antecedent=((x_hue['VH'] & x_blue['VVH']& x_energy['L']) | (x_coarseness['VH']& x_blue['VVH'] & x_energy['L'])| x_blue['VVH']),consequent=y_class['8'], label='rule 8')   # 输出为8的规则

# 系统和运行环境初始化
system = ctrl.ControlSystem(rules=[rule1, rule2, rule3,rule4,rule5,rule6,rule7,rule8])
sim    = ctrl.ControlSystemSimulation(system)

testx = np.loadtxt("../dataset/testx.txt")
testy = np.loadtxt("../dataset/testy.txt")
x_vals_test = np.array([[x[2],x[6],x[9], x[13],x[17]]for x in testx])
y_vals_test = np.array(testy).astype(int) -1
output_powder = []            # 输出

num_classes = 8               # 类别数
pro  = np.zeros((192,8))      # 概率
pro1 = np.zeros((192,8))      # 归一化概率

for i in range(len(x_vals_test)):      
        sim.input['hue'] = x_vals_test[i,0]
        sim.input['blue'] = x_vals_test[i,1]
        sim.input['red'] = x_vals_test[i,2]
        sim.input['coarseness'] = x_vals_test[i,3]
        sim.input['energy'] = x_vals_test[i,4]			
        sim.compute()
        # 运行系统
        output_powder.append(sim.output['class'])

# 打印输出结果
#print(output_powder)    

for i in range(len(x_vals_test)):
    for j in range(num_classes):
        pro[i,j]=1.0 / abs(j + 1 - output_powder[i])
    predict = pro.argmax(axis=1)
np.save('../probabilistic_output/expert_output.npy', pro)  # 注意带上后缀名

correct_sample_number = 0                                  # 分类正确的样本数
for i in range(len(x_vals_test)): 
    if predict[i] == y_vals_test[i]:
        correct_sample_number += 1
acc = correct_sample_number / len(x_vals_test)

def plot_matrix():
    # 读取标签
    name='expert'
    json_label_path = '../class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=8, labels=labels)
    confusion.update(predict, y_vals_test)
    confusion.plot(name)
    table = confusion.summary(name)
    np.save('../confusion_matrix/' + name + '_confusion_matrix.npy', table)

plot_matrix()