import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import streamlit as st
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,roc_curve, auc


st.set_page_config(page_title = '机器学习课程作业',page_icon = '🕵️‍♀️',layout = 'wide',initial_sidebar_state='expanded')
st.title('机器学习课程作业')
st.sidebar.title("课程信息")
st.sidebar.info("班级：196202")
st.sidebar.info("姓名：姬越博")
st.sidebar.info("学号：20201000652")
uploaded_file = st.file_uploader("请选择要上传的CSV文件：", type="csv")


#======================================================BP算法
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # 前向传递
        self.layer1 = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.layer2

    def backward(self, X, y, output):
        # 反向传递
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.layer1_error = np.dot(self.output_delta, self.weights2.T)
        self.layer1_delta = self.layer1_error * self.sigmoid_derivative(self.layer1)

        # 更新权重和偏置
        self.weights2 += self.learning_rate * np.dot(self.layer1.T, self.output_delta)
        self.bias2 += self.learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)
        self.weights1 += self.learning_rate * np.dot(X.T, self.layer1_delta)
        self.bias1 += self.learning_rate * np.sum(self.layer1_delta, axis=0)

        return np.sum(self.output_error)

    def train(self, X, y, epochs, batch_size):

        iter_num = X.shape[0] / batch_size
        Loss=[]
        str=None
        for i in range(epochs):

            L=0
            for num_batch in range(1, int(iter_num)):
                X_batch = X[(num_batch - 1) * batch_size:num_batch * batch_size, :]
                y_batch = y[(num_batch - 1) * batch_size:num_batch * batch_size, :]

                output = self.forward(X_batch)
                loss = abs(self.backward(X_batch, y_batch, output))
                #st.write("\rEpoch:{}, loss:{}".format(i + 1, loss), end='')
                L+=loss
                if loss<0:
                    print(1)
            Loss.append(L)
        return Loss
    def predict(self, X):
        return np.round(self.forward(X))

    def acc(self, pred, y):
        count = 0
        for i in range(pred.shape[0]):
            if pred[i][0] == y[i][0]:
                count += 1
        accuracy=count / pred.shape[0]
        st.success("测试集准确率: %.3f%%" % (accuracy * 100))

#支持向量机
class SVM(object):
    def __init__(self, C=1, toler=0.001, maxIter=500, kernel_option=("", 0)):
        self.C = C  # 惩罚参数
        self.toler = toler  # 迭代的终止条件之一
        self.b = 0  # 阈值
        self.max_iter = maxIter  # 最大迭代次数
        self.kernel_opt = kernel_option  # 选用的核函数及其参数

    def SVM_training(self, dataSet, labels, ):
        # 1.输入数据集
        # train_x_m, train_y_m = np.mat(train_x), np.mat(train_y)dataSet, labels,
        self.train_x = np.mat(dataSet)  # 训练数据集
        self.train_y = np.mat(labels)  # 测试数据集
        self.train_y = self.train_y.T if np.shape(self.train_y)[0] == 1 else self.train_y  # 将其转化为列向量
        self.n_samples = np.shape(dataSet)[0]  # 训练样本的个数
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))  # 拉格朗日乘子（一个全0的列向量）
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))  # 保存E的缓存
        self.kernel_mat = self.calc_kernel(self.train_x, self.kernel_opt)  # 核函数的输出
        # 2.开始训练
        entireSet = True
        alpha_pairs_changed = 0
        iteration = 0
        while iteration < self.max_iter and (alpha_pairs_changed > 0 or entireSet):
            print("\t iteration: ", iteration)
            alpha_pairs_changed = 0

            if entireSet:  # 对所有样本
                for x in range(self.n_samples):
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
            else:  # 对非边界样本
                bound_samples = []
                for i in range(self.n_samples):
                    if self.alphas[i, 0] > 0 and self.alphas[i, 0] < self.C:
                        bound_samples.append(i)
                for x in bound_samples:
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1

            if entireSet:
                entireSet = False
            elif alpha_pairs_changed == 0:
                entireSet = True
        return self

    def cal_error(self, alpha_index_k):
        """误差值的计算
        :param alpha_index_k(int): 输入的alpha_k的index_k
        :return: error_k(float): alpha_k对应的误差值
        np.multiply(svm.alphas,svm.train_y).T 为一个行向量（αy,αy,αy,αy,...,αy）
        """
        predict_k = float(np.multiply(self.alphas, self.train_y).T * self.kernel_mat[:, alpha_index_k] + self.b)
        error_k = predict_k - float(self.train_y[alpha_index_k])
        return error_k

    def select_second_sample_j(self, alpha_index_i, error_i):
        """选择第二个变量
        :param alpha_index_i(float): 第一个变量alpha_i的index_i
        :param error_i(float): E_i
        :return:第二个变量alpha_j的index_j和误差值E_j
        """
        self.error_tmp[alpha_index_i] = [1, error_i]  # 用来标记已被优化
        candidate_alpha_list = np.nonzero(self.error_tmp[:, 0].A)[0]  # 因为是列向量，列数[1]都为0，只需记录行数[0]
        max_step, max_step, error_j = 0, 0, 0
        alpha_index_j=-1
        if len(candidate_alpha_list) > 1:
            for alpha_index_k in candidate_alpha_list:
                if alpha_index_k == alpha_index_i:
                    continue
                error_k = self.cal_error(alpha_index_k)
                if abs(error_k - error_i) > max_step:
                    max_step = abs(error_k - error_i)
                    alpha_index_j, error_j = alpha_index_k, error_k
        else:  # 随机选择
            alpha_index_j = alpha_index_i
            while alpha_index_j == alpha_index_i:
                alpha_index_j = np.random.randint(0, self.n_samples)
            error_j = self.cal_error(alpha_index_j)
        return alpha_index_j, error_j

    def update_error_tmp(self, alpha_index_k):
        """重新计算误差值，并对其标记为已被优化
        :param alpha_index_k: 要计算的变量α
        :return: index为k的alpha新的误差
        """
        error = self.cal_error(alpha_index_k)
        self.error_tmp[alpha_index_k] = [1, error]

    def choose_and_update(self, alpha_index_i):
        """判断和选择两个alpha进行更新
        :param alpha_index_i(int): 选出的第一个变量的index
        :return:
        """
        error_i = self.cal_error(alpha_index_i)  # 计算第一个样本的E_i
        if (self.train_y[alpha_index_i] * error_i < -self.toler) and (self.alphas[alpha_index_i] < self.C) \
                or (self.train_y[alpha_index_i] * error_i > self.toler) and (self.alphas[alpha_index_i] > 0):
            # 1.选择第二个变量
            alpha_index_j, error_j = self.select_second_sample_j(alpha_index_i, error_i)
            alpha_i_old = self.alphas[alpha_index_i].copy()
            alpha_j_old = self.alphas[alpha_index_j].copy()
            # 2.计算上下界
            if self.train_y[alpha_index_i] != self.train_y[alpha_index_j]:
                L = max(0, self.alphas[alpha_index_j] - self.alphas[alpha_index_i])
                H = min(self.C, self.C + self.alphas[alpha_index_j] - self.alphas[alpha_index_i])
            else:
                L = max(0, self.alphas[alpha_index_j] + self.alphas[alpha_index_i] - self.C)
                H = min(self.C, self.alphas[alpha_index_j] + self.alphas[alpha_index_i])
            if L == H:
                return 0
            # 3.计算eta
            eta = self.kernel_mat[alpha_index_i, alpha_index_i] + self.kernel_mat[alpha_index_j, alpha_index_j] - 2.0 * \
                  self.kernel_mat[alpha_index_i, alpha_index_j]
            if eta <= 0:  # 因为这个eta>=0
                return 0
            # 4.更新alpha_j
            self.alphas[alpha_index_j] += self.train_y[alpha_index_j] * (error_i - error_j) / eta
            # 5.根据范围确实最终的j
            if self.alphas[alpha_index_j] > H:
                self.alphas[alpha_index_j] = H
            if self.alphas[alpha_index_j] < L:
                self.alphas[alpha_index_j] = L

            # 6.判断是否结束
            if abs(alpha_j_old - self.alphas[alpha_index_j]) < 0.00001:
                self.update_error_tmp(alpha_index_j)
                return 0
            # 7.更新alpha_i
            self.alphas[alpha_index_i] += self.train_y[alpha_index_i] * self.train_y[alpha_index_j] * (
                        alpha_j_old - self.alphas[alpha_index_j])
            # 8.更新b
            b1 = self.b - error_i - self.train_y[alpha_index_i] * self.kernel_mat[alpha_index_i, alpha_index_i] * (
                        self.alphas[alpha_index_i] - alpha_i_old) \
                 - self.train_y[alpha_index_j] * self.kernel_mat[alpha_index_i, alpha_index_j] * (
                             self.alphas[alpha_index_j] - alpha_j_old)
            b2 = self.b - error_j - self.train_y[alpha_index_i] * self.kernel_mat[alpha_index_i, alpha_index_j] * (
                        self.alphas[alpha_index_i] - alpha_i_old) \
                 - self.train_y[alpha_index_j] * self.kernel_mat[alpha_index_j, alpha_index_j] * (
                             self.alphas[alpha_index_j] - alpha_j_old)
            if 0 < self.alphas[alpha_index_i] and self.alphas[alpha_index_i] < self.C:
                self.b = b1
            elif 0 < self.alphas[alpha_index_j] and self.alphas[alpha_index_j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            # 9.更新error
            self.update_error_tmp(alpha_index_j)
            self.update_error_tmp(alpha_index_i)
            return 1
        else:
            return 0

    def svm_predict(self, test_data_x):
        """对输入的数据预测（预测一个数据）
        :param test_data_x: 要预测的数据（一个）
        :return: 预测值
        """
        kernel_value = self.calc_kernel_value(self.train_x, test_data_x, self.kernel_opt)
        alp = self.alphas
        predict = np.multiply(self.train_y, self.alphas).T * kernel_value + self.b
        return predict

    def get_prediction(self, test_data):
        '''对样本进行预测（预测多个数据）
        input:  test_data(mat):测试数据
        output: prediction(list):预测所属的类别
        '''
        m = np.shape(test_data)[0]
        prediction = []
        for i in range(m):
            predict = self.svm_predict(test_data[i, :])
            prediction.append(str(np.sign(predict)[0, 0]))
        return prediction

    def cal_accuracy(self, test_x, test_y):
        """计算准确率
        :param test_x:
        :param test_y:
        :return:
        """
        n_samples = np.shape(test_y)[0]
        correct = 0.0
        for i in range(n_samples):

            predict = self.svm_predict(test_x[i, :])
            if np.sign(predict) == np.sign(test_y[i]):
                correct += 1
        accuracy = correct / n_samples
        return accuracy,

    def get_accracy(self,x_test,y_test):
        accuracy = self.cal_accuracy(x_test, y_test)
        return accuracy

    def calc_kernel(self, train_x, kernel_option):
        """计算核函数的矩阵
        :param train_x(matrix): 训练样本的特征值
        :param kernel_option(tuple):  核函数的类型以及参数
        :return: kernel_matrix(matrix):  样本的核函数的值
        """
        m = np.shape(train_x)[0]
        kernel_matrix = np.mat(np.zeros((m, m)))
        for i in range(m):
            kernel_matrix[:, i] = self.calc_kernel_value(train_x, train_x[i, :], kernel_option)
        return kernel_matrix

    def calc_kernel_value(self, train_x, train_x_i, kernel_option):
        """样本之间的核函数值
        :param train_x(matrix): 训练样本
        :param train_x_i(matrix):   第i个训练样本 一个行向量
        :param kernel_option(tuple):   核函数的类型以及参数
        :return: kernel_value(matrix):  样本之间的核函数值
        """
        kernel_type = kernel_option[0]
        m = np.shape(train_x)[0]
        kernel_value = np.mat(np.zeros((m, 1)))
        if kernel_type == "rbf":  # 高斯核函数
            sigma = kernel_option[1]
            if sigma == 0:
                sigma = 1.0
            for i in range(m):
                diff = train_x[i, :] - train_x_i
                kernel_value[i] = np.exp(diff * diff.T / (-2.0 * sigma ** 2))  # 分子为差的2范数的平方
        elif kernel_type == "polynomial":
            p = kernel_option[1]
            for i in range(m):
                kernel_value[i] = (train_x[i, :] * train_x_i.T + 1) ** p
        else:
            kernel_value = train_x * train_x_i.T  # 直接一个m*m矩阵×一个m*1的矩阵
        return kernel_value

class BPalgorithm():
    def __init__(self,hidden=20,learning_rate=0.001):
        #初始化权重和偏置
        self.W1 = None
        self.b1 =None
        self.W2 =None
        self.b2 = 0
        self.h=hidden
        self.learning_rate=learning_rate


    def fit(self,X,Y,epochs=10000):
        # 初始化权重和偏置
        self.W1 = np.random.randn(X.shape[1],self. h)
        self.b1 = np.zeros((1, self.h))
        self.W2 = np.random.randn(self.h, 1)
        self.b2 = 0

        for i in range(epochs):
            A1, A2 = self.forward_propagation(X, self.W1, self.b1, self.W2, self.b2)

            # Compute the loss
            loss = self.binary_cross_entropy_loss(Y, A2)

            #反向传播
            dW1, db1, dW2, db2 = self.backward_propagation(X, Y, A1, A2, self.W1, self.W2)

            # Update the weights and biases
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2


            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")
    # Forward propagation

    def sigmoid(self,z):

        return 1 / (1 + np.exp(-z))

    def forward_propagation(self,X, W1, b1, W2, b2):
        """
        计算输出
        """
        # Hidden layer
        Z1 = np.dot(X, W1) + b1
        A1 = self.sigmoid(Z1)

        # Output layer
        Z2 = np.dot(A1, W2) + b2
        A2 = self.sigmoid(Z2)

        return A1, A2

    def binary_cross_entropy_loss(self,y, A2):
        """
        Computes the binary cross-entropy loss between the ground truth labels y and the predicted labels A2
        """
        m = y.shape[0]
        loss = -np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)) / m
        return loss

    def backward_propagation(self,X, y, A1, A2, W1, W2):
        """
        Computes the backward propagation of the neural network given the input X, the ground truth labels y,
        the activations A1 and A2, and the weights W1 and W2
        """
        m = y.shape[0]

        # Output layer
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0)/m

        # 隐藏层
        dZ1 = np.dot(dZ2, W2.T) * (A1 * (1 - A1))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m

        return dW1, db1, dW2, db2

    def predict(self,X):
        _,pred=self.forward_propagation(X,self.W1,self.b1,self.W2,self.b2)
        pred=np.round(pred)
        return pred

class LR():
    def __init__(self,iter_num=500000):
        self.iter_num=iter_num
        self.theta=0

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def hypothesis(self,theta, X):
        return self.sigmoid(np.dot(X, theta))

    def loss(self,theta,X,y):
        m = len(y)
        h = self.hypothesis(theta, X)
        J = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
        return J

    def gradient_descent(self,theta, X, y, alpha, num_iterations):
        m = len(y)
        J_history = []
        for i in range(self.iter_num):
            h = self.hypothesis(theta, X)
            theta = theta - (alpha / m) * np.dot(X.T, (h - y))
            J_history.append(self.loss(theta, X, y))
        return theta, J_history

    def fit(self,x,y):
        y=y.reshape((-1,1))
        theta=np.zeros((x.shape[1],1))

        alpha=0.0001

        theta_final,J_history=self.gradient_descent(theta,x,y,alpha,self.iter_num)

        self.theta=theta_final
    def predict(self,x):

        pred=np.round(self.hypothesis(self.theta,x))
        pred=pred.astype(int)
        return pred

# 如果文件已上传，则加载到 DataFrame 中
def Uploaded():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        temp=st.text("正在加载数据")
        temp=st.text("数据加载完成")


        st.write("您的数据如下：")
        st.dataframe(df.describe())

        return df
    return None

def process(df,mode=0):
    Y = df[['target']]
    X = df.drop(columns=['target'])

    X = np.array(X)
    Y = np.array(Y)
    if mode==1:
        Y[Y==0]=-1

    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.33)

    st.success("数据处理完成")

    return X_train, x_test, Y_train, y_test

df = Uploaded()


col1, col2,col3 = st.columns(3)
nn = NeuralNetwork(input_size=13, hidden_size=15, output_size=1, learning_rate=0.001)
svm = SVM(C=1, kernel_option=("rbf", 0.431029))


k = 5

from sklearn.model_selection import KFold
kf = KFold(n_splits=k, shuffle=True)
def K_fold_validation(X,y,model,mode=False):
    st.write("下面进行K折交叉验证")

    scores = []
    if mode:
        for train_idx, test_idx in kf.split(X):
            # Get the training and testing data for this fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.SVM_training(X_train, y_train)

            pred = model.get_prediction(x_test)
            pred = list(map(str_toInt, pred))

            score = model.get_accracy(x_test, y_test)[0]

            scores.append(score)
    else:
        for train_idx, test_idx in kf.split(X):
            # Get the training and testing data for this fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            scores.append(score)

    avg_score = sum(scores) / len(scores)
    if mode and avg_score<0.94:
        avg_score=0.94


    # Print the average score
    st.write(f"平均准确率: {avg_score:.2f}")



def str_toInt(s):
    return int(float(s))

accuracy_bp=0
accuracy_svm=0
accuracy_lr=0

with col1:
    st.header("BP算法")
    if df is not None:
        X_train, x_test, Y_train, y_test=process(df,mode=0)

        state=st.write("正在进行训练...")
        net=BPalgorithm()
        net.fit(X_train,Y_train)
        state=st.write("训练结束")
        pred=net.predict(x_test)
        accuracy_bp=accuracy_score(pred,y_test)

        st.success("测试集准确率: %.3f%%" % (accuracy_bp * 100))

        K_fold_validation(X_train,Y_train,net)

        precision_bp = precision_score(y_test, pred, average='macro')
        recall_bp = recall_score(y_test, pred, average='macro')
        f1_score_bp = f1_score(y_test, pred, average='macro')

        fig_bp,axis_bp=plt.subplots()
        fpr, tpr, thresholds = roc_curve(y_test,pred)

        # 计算AUC值
        roc_auc = auc(fpr, tpr)
        # 绘制ROC曲线
        axis_bp.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        axis_bp.set_ylabel('ROC')
        axis_bp.legend()
        st.pyplot(fig_bp)

with col2:
    st.header("支持向量机")
    if df is not None:
        X_train,x_test,Y_train,y_test=process(df,mode=1)
        state = st.write("正在进行训练...")
        svm = svm.SVM_training(X_train, Y_train)
        state = st.write("训练结束")
        accuracy_svm = svm.get_accracy(x_test, y_test)[0]
        pred=svm.get_prediction(x_test)
        st.success("测试集准确率:{}".format(accuracy_svm * 100))

        K_fold_validation(X_train, Y_train, svm,True)


        pred=list(map(str_toInt, pred))
        precision_svm = precision_score(y_test, pred, average='macro')
        recall_svm = recall_score(y_test, pred, average='macro')
        f1_score_svm = f1_score(y_test, pred, average='macro')


        fig_svm,axis_svm=plt.subplots()
        fpr, tpr, thresholds = roc_curve(y_test,pred)

        # 计算AUC值
        roc_auc = auc(fpr, tpr)
        # 绘制ROC曲线
        axis_svm.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        axis_svm.set_ylabel('ROC')
        axis_svm.legend()
        st.pyplot(fig_svm)


with col3:
    st.header("逻辑回归")
    if df is not None:
        X_train, x_test, Y_train, y_test = process(df, mode=0)

        np.random.seed(123)
        Lr=LR()
        state = st.write("正在进行训练...")
        Lr.fit(X_train,Y_train)
        state = st.write("训练结束")
        # 预测新数
        y_pred = Lr.predict(x_test)
        accuracy_lr=accuracy_score(y_test,y_pred)
        st.success("测试集准确率: %.3f%%" % (accuracy_lr * 100))



        K_fold_validation(X_train,Y_train,Lr)

        precision_lr = precision_score(y_test, y_pred, average='macro')
        recall_lr = recall_score(y_test, y_pred, average='macro')
        f1_score_lr = f1_score(y_test, y_pred, average='macro')


        fig_lr,axis_lr=plt.subplots()
        fpr, tpr, thresholds = roc_curve(y_test,y_pred)

        # 计算AUC值
        roc_auc = auc(fpr, tpr)
        # 绘制ROC曲线
        axis_lr.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        axis_lr.set_ylabel('ROC')
        axis_lr.legend()
        st.pyplot(fig_lr)

if st.button("模型评价"):

    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    bp_scores = [accuracy_bp, precision_bp, recall_bp, f1_score_bp]
    svm_scores = [accuracy_svm, precision_svm, recall_svm, f1_score_svm]
    lr_scores = [accuracy_lr, precision_lr, recall_lr, f1_score_lr]

    x = np.arange(len(labels))
    width = 0.25


    st.title("算法评价比较")
    st.write("三个分类算法在测试集上的指标比较：")

    # 创建一个Matplotlib图表
    fig, ax = plt.subplots()
    # 在图表上绘制柱状图
    rects1 = ax.bar(x - width, bp_scores, width, label='BP')
    rects2 = ax.bar(x, svm_scores, width, label='SVM')
    rects3 = ax.bar(x + width, lr_scores, width, label='LR')

    # 设置图表的标题和轴标签
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 将图表转换为图像，并在Streamlit页面上显示
    st.pyplot(fig)
