from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self,name):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, accuracy
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Accuracy"]
        p=[]
        r=[]
        a=[]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Accuracy=acc
            table.add_row([self.labels[i], Precision, Recall,Accuracy])
            p.append(Precision)
            r.append(Recall)
            a.append(Accuracy)
        index={"precision" : p, "recall" : r,  "Acc":a}
        np.save('../evaluation_index_mat/' + name+'_index.npy', index)
        print(table)
        return self.matrix

    def plot(self,name):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)    # 设置x轴坐标label
        plt.yticks(range(self.num_classes), self.labels)                  # 设置y轴坐标label
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        '''以下两句话是解决TrueType font 问题'''
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'    # 也可以设置新罗马字体 plt.rcParams['font.family'] = 'Times New Roman'

        plt.savefig('../result_picture_pdf/' + name + '_confusion.pdf', bbox_inches='tight')
        plt.show()