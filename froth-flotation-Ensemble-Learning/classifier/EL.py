import os
import json
import numpy as np
import matplotlib.pyplot as plt
from confusion import ConfusionMatrix

'''
函数描述：用于计算分类精度，即多少样本被正确识别
输入参数：
     pre   - 预测的标签
     lable - 真实的标签
返回值： 
     accuracy - 精度（准确率）
'''
def accuracy(pre, label):
    correct_sample_number = 0
    for i in range(len(pre)):
        if pre[i] == label[i]:
            correct_sample_number += 1
    accuracy = correct_sample_number / len(pre)
    return (accuracy)

"""导入真实标签和分类器权重"""
testy = np.loadtxt("../dataset/testy.txt")              # 导入测试数据
Y_test = np.array(testy).astype(int) - 1                # 类别从 0 开始，所以减1
weight = np.load('../classifier_weight/weight.npy')     # 权重
num_classes = 8                                         # 定义8分类

'''导入各个模型已有的测试结果（以概率矩阵形式保存）'''
adaboost = np.load('../probabilistic_output/adaboost_output.npy', allow_pickle=True)
expert = np.load('../probabilistic_output/expert_output.npy', allow_pickle=True)
forest = np.load('../probabilistic_output/forest_output.npy', allow_pickle=True)
knn = np.load('../probabilistic_output/knn_output.npy', allow_pickle=True)
logistic = np.load('../probabilistic_output/logistic_output.npy', allow_pickle=True)
svm = np.load('../probabilistic_output/svm_output.npy', allow_pickle=True)
tree = np.load('../probabilistic_output/tree_output.npy', allow_pickle=True)

''' 对结果进行加权集成'''
GDM1 = np.zeros((len(Y_test), num_classes))

for i in range(len(Y_test)):
    for j in range(8):             # 列就代表类别
        GDM1[i, j] = weight[j, 0] * adaboost[i][j] + weight[j, 1] * expert[j][j] + weight[j, 2] * forest[i][j] + weight[
            j, 3] * knn[i][j] + weight[j, 4] * logistic[i][j] + weight[j, 5] * svm[j][j] + weight[j, 6] * tree[i][j]

GDM = np.argmax(GDM1, axis=1)     # 每一行选择概率最大的那一列作为最后的类别 ，刚好类别从 0 开始嘛
acc_GDM = accuracy(GDM, Y_test)   # 计算分类精度

'''导入数字标签对应的评语等级'''
json_label_path = '../class_indices.json'
assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
json_file = open(json_label_path, 'r')
class_indict = json.load(json_file)


''' 画图 '''
def plot_matrix():
    name = 'expert'
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=8, labels=labels)
    confusion.update(GDM, Y_test)
    confusion.plot(name)
    # table = confusion.summary(name)
    # np.save('./' + name + '_matrix.npy', table)

plot_matrix()