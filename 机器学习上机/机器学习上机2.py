import numpy as np
import pandas as pd
import streamlit as st
from math import log
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,auc,roc_curve
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class Bayes_classfier():
    def __init__(self,lamda=1):
        self.X_train=None
        self.y_train=None
        self.N=0
        self.lamda=lamda


        self.pos_priors=None
        self.nege_priors=None
        self.pos_condition_likehood=None
        self.nege_condition_likehood=None
        self.one_minus_pos_condition_likehood=None
        self.one_minus_nege_condtion_likehood=None

    def Check_Format(self):

        x_type_str = type(self.X_train)
        y_type_str=type(self.y_train)
        # 检查输入的训练数据格式
        if not x_type_str is np.ndarray :
            print('输入的训练数据要求numpy数组，您输入的是 :{}'.format(type(self.X_train)))
            return False

        #检查输入的标签格式
        is_list=y_type_str is list
        is_np=y_type_str is np.ndarray
        if is_list==False and is_np==False:
            print("输入的标签要求numpy数组或者列表")
            return False

        if y_type_str is list:
            self.y_train=np.array(self.y_train)

        self.y_train=self.y_train.reshape(-1,1)

        #记录样本数
        sample_nums=self.y_train.shape[0]

        if self.X_train.shape[0]!=sample_nums and self.X_train.shape[0]!=sample_nums:
            print("输入的数据量与标签量不匹配")
            return  False
        return True

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train

        if self.Check_Format()==False:
            print("训练失败,检查输入格式")
            return

        #统计样本数
        self.N=self.y_train.shape[0]

        #伯努利模型
        self.X_train[self.X_train!=0]=1

        #要计算每个类别中，每个词出现的总数
        #首先把正负类分别取出来
        postive_class_idx=np.where(self.y_train==1)[0]
        negetive_class_idx=np.where(self.y_train==0)[0]

        postive_class=self.X_train[postive_class_idx]
        negetive_class=self.X_train[negetive_class_idx]
        pos_nums=len(postive_class)
        nege_nums=len(negetive_class)

        #计算每个词值为1在每个类别下出现的次数
        pword_static=np.sum(postive_class,axis=0)
        nword_static=np.sum(negetive_class,axis=0)

        #计算每一个类别的先验概率
        self.pos_priors=pos_nums/self.N
        self.nege_priors=nege_nums/self.N

        #计算条件概率p(x=xj|y=ck),采用拉普拉斯平滑
        #p(x=1|y=ck),也就是某个特征取1时候的条件概率
        self.pos_condition_likehood=(pword_static+self.lamda)/(pos_nums+self.lamda*2)
        self.nege_condition_likehood=(nword_static+self.lamda)/(nege_nums+self.lamda *2)

        #p(x=0|y=ck)，也就是某个特征取0时候的条件概率
        self.one_minus_pos_condition_likehood=(self.N-pword_static)+self.lamda/(pos_nums+self.lamda*2)
        self.one_minus_nege_condtion_likehood=(self.N-nword_static)+self.lamda/(nege_nums+self.lamda *2)


    def predict(self,x_test:np.ndarray):
        #特征二值化
        x_test[x_test>0]=1
        y_pred = np.zeros(x_test.shape[0], dtype=int)

        # 计算属于某个类得概率，用对数计算把乘法变成加法。对于每个特征p(xj=aj|y)=(1-x)*p(x=1)+x*p(x=0)实现特征的不同取值连乘
        pos_likelihoods = np.sum(np.log(self.pos_condition_likehood) * x_test, axis=1) + np.sum(np.log(self.one_minus_pos_condition_likehood) * (1 - x_test), axis=1) + np.log(self.pos_priors)
        neg_likelihoods = np.sum(np.log(self.nege_condition_likehood) * x_test, axis=1) + np.sum(np.log(self.one_minus_nege_condtion_likehood) * (1 - x_test), axis=1) +np.log(self.nege_priors)

        # 确定类别
        y_pred[pos_likelihoods > neg_likelihoods] = 1
        return y_pred

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx # index of feature used for splitting
        self.threshold = threshold # threshold used for splitting
        self.left = left # left subtree
        self.right = right # right subtree
        self.value = value #

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth # 最大深度
        self.min_samples_split = min_samples_split # 最小切分
        self.tree = None # 树根

    def fit(self, X, y):
        #y变成一维数组
        y = y.ravel()
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 停止条件
        if (self.max_depth is not None and depth == self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (n_classes == 1):
            value = np.argmax(np.bincount(y))
            return Node(value=value)

        # 选择最优切分节点
        best_feature_idx, best_threshold = self._find_best_split(X, y, n_samples, n_features)

        # 创建节点
        left_idxs = np.where(X[:, best_feature_idx] <= best_threshold)[0]
        right_idxs = np.where(X[:, best_feature_idx] > best_threshold)[0]
        left = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
        return Node(best_feature_idx, best_threshold, left, right)

    def _find_best_split(self, X, y, n_samples, n_features):
        best_info_gain = -1
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                info_gain = self._info_gain(y, feature_values, threshold, n_samples)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _info_gain(self, y, feature_values, threshold, n_samples):
        parent_entropy = self._entropy(y, n_samples)
        left_idxs = np.where(feature_values <= threshold)[0]
        right_idxs = np.where(feature_values > threshold)[0]


        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # 计算子节点的信息熵
        n_left = len(left_idxs)
        n_right = len(right_idxs)
        entropy_left = self._entropy(y[left_idxs], n_left)
        entropy_right = self._entropy(y[right_idxs], n_right)

        # 计算G（D|a)
        child_entropy = (n_left / n_samples) * entropy_left + (n_right / n_samples) * entropy_right

        # 计算基尼指数
        info_gain = parent_entropy - child_entropy
        return info_gain

    def _entropy(self, y, n_samples):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n_samples
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        y=y.ravel()
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])

        # loop over all test examples
        for i, x_test in enumerate(X_test):
            # 计算与训练集的距离
            distances = np.sqrt(np.sum((self.X_train - x_test) ** 2, axis=1))

            #获取最邻近
            knn_indices = np.argsort(distances)[:self.k]

            #获取最近邻类别
            knn_labels = self.y_train[knn_indices]

            # 分类
            y_pred[i] = np.bincount(knn_labels).argmax()

        return y_pred


st.set_page_config(page_title = '机器学习课程作业',page_icon = '🕵️‍♀️',layout = 'wide',initial_sidebar_state='expanded')
st.title('机器学习课程作业2')
st.sidebar.title("课程信息")
st.sidebar.info("班级：196202")
st.sidebar.info("姓名：姬越博")
st.sidebar.info("学号：20201000652")
uploaded_file = st.file_uploader("请选择要上传的CSV文件：", type="csv")


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
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.33)

    st.success("数据处理完成")

    return X_train, x_test, Y_train, y_test

def str_toInt(s):
    return int(float(s))

k = 5

# 初始化K折交叉验证
kf = KFold(n_splits=k, shuffle=True)


def K_fold_validation(X,y,model):
    st.write("下面进行K折交叉验证")

    scores = []

    for train_idx, test_idx in kf.split(X):
        # Get the training and testing data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score = accuracy_score(y_test, y_pred)

        scores.append(score)

    avg_score = sum(scores) / len(scores)

    # Print the average score
    st.write(f"平均准确率: {avg_score:.2f}")


if __name__ == '__main__':

    df = Uploaded()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("朴素贝叶斯算法")
        if df is not None:
            X_train, x_test, Y_train, y_test = process(df, mode=0)

            state = st.write("正在进行训练...")
            bys=Bayes_classfier()
            bys.fit(X_train, Y_train)
            state = st.write("训练结束")
            pred = bys.predict(x_test)
            accuracy_bys = accuracy_score(pred, y_test)

            st.success("测试集准确率: %.3f%%" % (accuracy_bys * 100))

            K_fold_validation(X_train,Y_train,bys)
#=============================================绘图
            precision_bp = precision_score(y_test, pred, average='macro')
            recall_bp = recall_score(y_test, pred, average='macro')
            f1_score_bp = f1_score(y_test, pred, average='macro')

            fig_bp, axis_bp = plt.subplots()
            fpr, tpr, thresholds = roc_curve(y_test, pred)

            # 计算AUC值
            roc_auc = auc(fpr, tpr)
            # 绘制ROC曲线
            axis_bp.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            axis_bp.set_ylabel('ROC')
            axis_bp.legend()
            st.pyplot(fig_bp)

    with col2:
        st.header("ID3决策树")
        if df is not None:
            X_train, x_test, Y_train, y_test = process(df, mode=0)
            state = st.write("正在进行训练...")

            dt = DecisionTree(max_depth=10)
            dt.fit(X_train, Y_train)
            pred = dt.predict(x_test)  # Output: [0 1 1 0]
            accuracy_id3= accuracy_score(y_test, pred)
            st.write("训练结束")
            st.success("测试集准确率:{}".format(accuracy_id3 * 100))

            K_fold_validation(X_train,Y_train,dt)

            precision_svm = precision_score(y_test, pred, average='macro')
            recall_svm = recall_score(y_test, pred, average='macro')
            f1_score_svm = f1_score(y_test, pred, average='macro')

            fig_svm, axis_svm = plt.subplots()
            fpr, tpr, thresholds = roc_curve(y_test, pred)

            # 计算AUC值
            roc_auc = auc(fpr, tpr)
            # 绘制ROC曲线
            axis_svm.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            axis_svm.set_ylabel('ROC')
            axis_svm.legend()
            st.pyplot(fig_svm)
    with col3:
        st.header("knn算法")
        if df is not None:
            X_train, x_test, Y_train, y_test = process(df, mode=0)

            np.random.seed(123)
            knn=KNN(k=3)
            state = st.write("正在进行训练...")
            knn.fit(X_train, Y_train)
            state = st.write("训练结束")
            # 预测新数
            y_pred = knn.predict(x_test)
            accuracy_knn = accuracy_score(y_test, y_pred)
            st.success("测试集准确率: %.3f%%" % (accuracy_knn * 100))

            K_fold_validation(X_train, Y_train, knn)

            precision_lr = precision_score(y_test, y_pred, average='macro')
            recall_lr = recall_score(y_test, y_pred, average='macro')
            f1_score_lr = f1_score(y_test, y_pred, average='macro')

            fig_lr, axis_lr = plt.subplots()
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)

            # 计算AUC值
            roc_auc = auc(fpr, tpr)
            # 绘制ROC曲线
            axis_lr.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            axis_lr.set_ylabel('ROC')
            axis_lr.legend()
            st.pyplot(fig_lr)


if st.button("模型评价"):

    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    bp_scores = [accuracy_bys, precision_bp, recall_bp, f1_score_bp]
    svm_scores = [accuracy_id3, precision_svm, recall_svm, f1_score_svm]
    lr_scores = [accuracy_knn, precision_lr, recall_lr, f1_score_lr]

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
