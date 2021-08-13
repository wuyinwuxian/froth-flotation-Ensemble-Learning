import numpy as np
import joblib    # 版本0.22之后的scikit_learn中就除掉了joblib这个函数或包。 需要直接下载joblib这个包
from sklearn.preprocessing import normalize


testx = np.loadtxt("../dataset/testx.txt")                                     # 导入测试数据
testy = np.loadtxt("../dataset/testy.txt")
x_test = np.array([[x[2],x[6],x[9], x[13],x[17]]for x in testx])    # 特征选择以后的特征,这些特征是我们提前用特征选择算法优化而来
y_vals_test = np.array(testy).astype(int) -1
x_vals_test = normalize(x_test,axis=0,norm='max')                   # 特征归一化

'''导入各个模型'''
ADA = joblib.load('../model/adaboost.model')
SVM = joblib.load('../model/svm.model')
LR  = joblib.load('../model/logistic.model')
KNN = joblib.load('../model/knn.model')
TREE   = joblib.load('../model/tree.model')
FOREST = joblib.load('../model/forest.model')

'''计算各个模型测试结果（以概率矩阵形式保存）可以与 predict 函数做对比'''
adaboost = ADA.predict_proba(x_vals_test)
svm = SVM.predict_proba(x_vals_test)
lr  = LR.predict_proba(x_vals_test)
knn = KNN.predict_proba(x_vals_test)
tree   = TREE.predict_proba(x_vals_test)
forest = FOREST.predict_proba(x_vals_test)