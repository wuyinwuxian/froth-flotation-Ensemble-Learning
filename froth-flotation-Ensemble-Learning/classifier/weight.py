import pandas as pd
import numpy as np

'''
函数描述：用于求解每个分类器的权重
输入参数：
       P - 查准率
       R - 召回率，查全率
       A - 精度
返回值：
       w - 权重矩阵
'''
def weight(P,R,A):
    a = np.array(A)
    r = np.array(R)
    p = np.array(P)
    w = p + a + r
    return w


'''导入各个模型已有的测试结果（以概率矩阵形式保存）'''
knn  = np.load('../evaluation_index_mat/knn_index.npy', allow_pickle=True).item()
svm  = np.load('../evaluation_index_mat/svm_index.npy', allow_pickle=True).item()
tree = np.load('../evaluation_index_mat/tree_index.npy', allow_pickle=True).item()
expert = np.load('../evaluation_index_mat/expert_index.npy', allow_pickle=True).item()
forest = np.load('../evaluation_index_mat/forest_index.npy', allow_pickle=True).item()
logistic = np.load('../evaluation_index_mat/logistic_index.npy', allow_pickle=True).item()
adaboost = np.load('../evaluation_index_mat/adaboost_index.npy', allow_pickle=True).item()

data = pd.DataFrame(
    {'P': [adaboost['precision'],expert['precision'],forest['precision'],knn['precision'],logistic['precision'],svm['precision'],tree['precision']],
     'R': [adaboost['recall'],expert['recall'],forest['recall'],knn['recall'],knn['recall'],logistic['recall'],tree['recall']],
     'A': [adaboost['Acc'],expert['Acc'],forest['Acc'],knn['Acc'],logistic['Acc'],svm['Acc'],tree['Acc']]},
     index=['adaboost','expert','forest','knn','logistic','svm','tree'])


'''计算权重'''
knn_w  = weight(data['P']['knn'],data['R']['knn'],data['A']['knn'])
svm_w  = weight(data['P']['svm'],data['R']['svm'],data['A']['svm'])
tree_w = weight(data['P']['tree'],data['R']['tree'],data['A']['tree'])
expert_w = weight(data['P']['expert'],data['R']['expert'],data['A']['expert'])
forest_w = weight(data['P']['forest'],data['R']['forest'],data['A']['forest'])
adaboost_w = weight(data['P']['adaboost'],data['R']['adaboost'],data['A']['adaboost'])
logistic_w = weight(data['P']['logistic'],data['R']['logistic'],data['A']['logistic'])


'''构成权重矩阵，行代表类别，列代表分类器'''
s = np.zeros((8,7))
for i in range(8):
    s[i] = [adaboost_w[i],expert_w[i],forest_w[i],knn_w[i],logistic_w[i],svm_w[i],tree_w[i]]

'''权重归一化，对于一个类别的判断，不同分类器有不同的话语重要性'''
sum1 = np.sum(s, axis=1, keepdims=True)                        # 行求和
w = s / sum1                                                   # probaij为预测为i实际为j的概率
np.save('../classifier_weight/weight.npy', w)                  # 保存 注意带上后缀名

