from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.preprocessing import normalize
from confusion import ConfusionMatrix
from sklearn.metrics import precision_score, recall_score
import joblib    # 版本0.22之后的scikit_learn中就除掉了joblib这个函数或包。 需要直接下载joblib这个包
import os
import json

''''训练'''
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=200, learning_rate=0.8)   #定义机器学习算法

num_classes = 8                                  # 定义8分类
name   = 'adaboost'                              # 采用的方法名
trainx = np.loadtxt("../dataset/trainx.txt")     # 导入训练数据
trainy = np.loadtxt("../dataset/trainy.txt")
x_train = np.array([[x[2],x[6],x[9], x[13],x[17]] for x in trainx])    # 特征选择以后的特征,这些特征是我们提前用特征选择算法优化而来
x_vals_train = normalize(x_train, axis=0, norm='max')                  # 特征归一化
y_vals_train = np.array(trainy).astype(int) -1                         # 类别从 0 开始，所以减1

classifier.fit(x_vals_train, y_vals_train)                    # 对分类器进行训练
joblib.dump(classifier, '../model/' + name + '.model')        # 保存模型
predict_train = classifier.predict(x_vals_train)              # 训练集预测结果
predict_pro_train = classifier.predict_proba(x_vals_train)    # 训练集预测结果的概率
acc_train = classifier.score(x_vals_train, y_vals_train)      # 训练集准确率

Acc = [acc_train,acc_train,acc_train,acc_train,acc_train,acc_train,acc_train,acc_train]          # 训练集评价指标 精度
precision = list(precision_score(classifier.predict(x_vals_train), y_vals_train, average=None))  # 训练集评价指标 查准率
recall = list(recall_score(classifier.predict(x_vals_train), y_vals_train, average=None))        # 训练集评价指标 召回率 查全率
index = {"precision" : precision, "recall" : recall, "Acc":Acc}                                  # 将列表a，b转换成字典
np.save('../evaluation_index_mat/' + name+'_index.npy', index)                                   # 保存注意带上后缀名


'''测试'''
testx = np.loadtxt("../dataset/testx.txt")                               # 导入测试数据
testy = np.loadtxt("../dataset/testy.txt")
x_test = np.array([[x[2],x[6],x[9], x[13],x[17]]for x in testx])         # 特征选择以后的特征,这些特征是我们提前用特征选择算法优化而来
x_vals_test = normalize(x_test,axis=0,norm='max')                        # 特征归一化
y_vals_test = np.array(testy).astype(int) -1                             # 类别从 0 开始，所以减1

predict = classifier.predict(x_vals_test)                                # 测试集预测结果
predict_pro = classifier.predict_proba(x_vals_test)                      # 测试集预测结果的概率
np.save('../probabilistic_output/' + name + '_output.npy', predict_pro)  # 保存注意带上后缀名
acc = classifier.score(x_vals_test,y_vals_test)                          # 准确率，精度


'''画图'''
def plot_matrix():
    json_label_path = '../class_indices.json'      # 读取标签
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


