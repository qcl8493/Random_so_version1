from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import numpy as np
from scipy.stats import multivariate_normal, stats
from stats import mean
from sklearn.covariance import MinCovDet

classes = ["combinate_modulation","cross_modulation","echo","frequency_diversity"
    ,"harmonic_interference","inter_modulation","pulse_compression",
    "repetitive_frequency_grouping","repetitive_frequency_jitter","spread_spectrum"]


class OpenSetClassifier:
    def __init__(self, base_clf = RandomForestClassifier(n_estimators=200, random_state=42),threshold=0.85, entropy_threshold=0.5,
                 novelty_threshold= 10, update_interval=10,):
        self.clf = base_clf
        self.threshold = threshold
        self.entropy_threshold = entropy_threshold
        self.novelty_threshold = novelty_threshold
        self.update_interval = update_interval
        self.known_classes = None
        self.stat_buffer = []  # 存储已知类别的统计信息
        self.novel_samples = []  # 存储潜在的新类样本

    def fit(self, X, y):
        self.known_classes = np.unique(y)
        self.clf.fit(X, y)
        # 初始化统计缓冲区
        self.stat_buffer = self._compute_class_statistics(X, y)

    def _compute_class_statistics(self, X, y):
        """计算已知类别的均值和协方差矩阵"""
        stats = {}
        classes = np.unique(y)
        for cls in classes:  # 遍历每个类别
            X_cls = X[y == cls]  # 筛选出当前类别的所有样本

            # 计算当前类别的均值向量（形状：[特征维度]）
            mean = np.mean(X_cls, axis=0)

            # 使用最小协方差行列式估计（鲁棒方法）计算协方差矩阵
            cov = MinCovDet().fit(X_cls).covariance_  # 形状：[特征维度, 特征维度]

            # 将均值和协方差存入字典
            stats[cls] = {'mean': mean, 'cov': cov}

        return stats

    def _mahalanobis_distance(self, x_new, X_mean, cov_inv):
        """计算马氏距离"""
        #X_mean = np.full_like(x_new.values, X_mean)  # 创建与x相同形状的数组，填充mean值
        diff = x_new - X_mean
        return np.sqrt(diff @ cov_inv @ diff.T)    #diff @ cov_inv: 对差向量 diff 与
        # 协方差逆矩阵 cov_inv 进行点积运算，得到一个标量（形状：[1]）。这一步等价于 (x - μ)^T Σ^{-1} (x - μ)。

    def predict_single_sample(self, stats, x_new, i):
        """
        改进的OpenSet预测流程：
        1. 动态计算置信度阈值
        2. 结合多种不确定性指标
        3. 自动管理未知样本
        """
        y_pred_proba = self.clf.predict_proba(x_new)[0]  # 获取样本属于各类别的概率
        max_prob = np.max(y_pred_proba)  # 最高概率值（置信度）
        entropy = -np.sum(y_pred_proba * np.log2(y_pred_proba + 1e-10))  # 计算预测分布的熵

        # 初始化最小马氏距离为一个很大的值
        min_mahalanobis_dist = float('inf')
        closest_class = None

        # 遍历所有已知类别，计算每个类别的马氏距离
        for cls in self.known_classes:
            # 提取当前类别的均值和协方差矩阵
            X_mean = stats[cls]['mean']
            cov = stats[cls]['cov']

            # 确保协方差矩阵可逆
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                print(f"类别 {cls} 的协方差矩阵不可逆，尝试使用伪逆...")
                cov_inv = np.linalg.pinv(cov)

            # 计算当前类别的马氏距离
            mahalanobis_dist = self._mahalanobis_distance(x_new.to_numpy().flatten(), X_mean, cov_inv)

            # 更新最小马氏距离及其对应的类别
            if mahalanobis_dist < min_mahalanobis_dist:
                min_mahalanobis_dist = mahalanobis_dist
                closest_class = cls

        # 综合决策机制
        is_novel = (
                (max_prob < self.threshold and entropy > self.entropy_threshold) or
                (max_prob < self.threshold and min_mahalanobis_dist > self.novelty_threshold) or
                (entropy > self.entropy_threshold and min_mahalanobis_dist > self.novelty_threshold)
        )

        if is_novel:
            print(f"信号{i}检测为新类别（概率：{max_prob:.4f}, 熵：{entropy:.2f}, "
                  f"最小马氏距离：{min_mahalanobis_dist:.2f}）")
            new_label = max(self.known_classes) + 1 if self.known_classes.size > 0 else 0
            self.novel_samples.append((x_new, new_label))

            # 定期更新模型
            # if len(self.novel_samples) >= self.update_interval:
            #     self._update_model()
            return new_label, True
        else:
            predicted_class = self.known_classes[np.argmax(y_pred_proba)]
            print(f"信号{i}分类为：{predicted_class}类（概率：{max_prob:.4f}, 熵：{entropy:.2f}, "
                  f"最小马氏距离：{min_mahalanobis_dist:.2f})")
            return predicted_class, False

    # def _update_model(self):
    #     """增量学习机制"""
    #     if not self.novel_samples:
    #         return
    #
    #     # 合并新样本到训练集
    #     X_novel, y_novel = zip(*self.novel_samples)
    #     X_novel = np.array(X_novel)
    #     y_novel = np.array(y_novel)
    #
    #     # 扩展已知类别
    #     self.known_classes = np.unique(np.concatenate([self.known_classes, y_novel]))
    #
    #     # 重新训练模型
    #     self.clf.fit(np.vstack([self.clf.support_vectors_, X_novel]),  # 假设使用SVM
    #                  np.concatenate([self.clf.classes_, y_novel]))
    #
    #     # 清空缓冲区并重新计算统计量
    #     self.stat_buffer = self._compute_class_statistics(
    #         np.vstack([self.clf.support_vectors_, X_novel]),
    #         self.clf.classes_
    #     )
    #     self.novel_samples = []


def train_model(clf, data, train_size, n_features_to_select=25):
    """
    训练模型的函数。

    参数:
    - data: 包含特征和标签的 Pandas DataFrame。
    - train_size: 初始训练集的大小，默认为 3995。
    - n_features_to_select: 使用 RFE 选择的特征数量，默认为 48。

    返回:
    - clf: 训练好的模型。
    - X_increasing: 当前训练集的特征。
    - Y_increasing_encoded: 当前训练集的编码标签。
    - label_encoder: 标签编码器。
    """
    # # 分离特征和标签
    # X = data.iloc[:, 1:]
    # Y = data.iloc[:, 0]
    #
    # # 对标签进行编码
    # label_encoder = LabelEncoder()
    # Y_encoded = label_encoder.fit_transform(Y)
    #
    # # 初始训练数据
    # X_train = X.iloc[:train_size, :]
    # Y_train = Y_encoded[:train_size]
    #
    # # 初始化基础分类器和 RFE
    # rfe_clf = RFE(estimator=clf, n_features_to_select = n_features_to_select)
    #
    # # 训练模型
    # rfe_clf.fit(X_train, Y_train)
    #
    # # 初始化变量
    # X_increasing = X_train
    # Y_increasing_encoded = Y_train
    #
    # return rfe_clf, X_increasing, Y_increasing_encoded, label_encoder



def process_multiple_samples(clf, X_increasing, Y_increasing_encoded, label_encoder, X, Y_encoded):
    """
    遍历多条数据进行预测，并统计识别准确率。

    参数:
    - clf: 已经训练好的模型。
    - X_increasing: 当前训练集的特征。
    - Y_increasing_encoded: 当前训练集的编码标签。
    - label_encoder: 标签编码器。
    - X: 包含所有特征的 Pandas DataFrame。
    - Y_encoded: 编码后的标签数组。
    - threshold: 不确定性阈值，用于判断是否为新样本，默认为 0.8。
    - some_threshold: 控制是否更新训练集的阈值，默认为 (0.9, 0.75)。
    - max_iterations: 最大迭代次数（处理的数据量），默认为 None（处理所有数据）。

    返回:
    - predictions: 预测的标签列表。
    - true_labels: 真实的标签列表。
    - accuracy: 模型在预测信号上的正确识别率。
    """


    return predictions, true_labels, accuracy


if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv('E:\\1.0研究项目\\1.0Self-created file\simple_version\dataset\\features.csv')
    train_size = 2985

    classifier = OpenSetClassifier(
    base_clf=RandomForestClassifier(n_estimators=200, random_state=42),
    threshold=0.85,
    entropy_threshold=0.45,
    novelty_threshold=10.0,
    update_interval=10
)



    # 分离特征和标签
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]

    # 对标签进行编码
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    # 初始训练数据
    X_train = X.iloc[:train_size, :]
    Y_train = Y_encoded[:train_size]

    # 初始化模型
    # clf = RFE(classifier, n_features_to_select= 25)  # 特征选择

    # 初始训练
    stats = classifier._compute_class_statistics(X_train, Y_train)
    classifier.fit(X_train, Y_train)

    # 初始化变量
    X_increasing = X_train
    Y_increasing_encoded = Y_train

    # 训练模型
    # rfe_clf, X_increasing, Y_increasing_encoded, label_encoder = train_model(open_set_clf,data, train_size, n_features_to_select=20)

    # 预测新数据并更新模型
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    Y_encoded = label_encoder.transform(Y)

    predictions = []
    true_labels = []

    train_size = len(X_increasing)
    # new_label = max(Y_increasing_encoded) + 1
    continue_training = True


    for i in range(train_size, len(X)):
        if not continue_training:
            break

        x_new = X.iloc[i:i + 1, :]  # 取出新的一条数据
        true_label = Y_encoded[i]

        # 调用单条数据识别函数

        y_new_encoded = classifier.predict_single_sample(stats,x_new, i)

        # 更新真实标签和预测标签
        predictions.append(y_new_encoded)
        true_labels.append(true_label)

        # 每处理 10 条数据询问是否继续
        if (i - train_size + 1) % 10 == 0:
            print("---------------------------")
            print(f'当前信号类别：{np.unique(Y_increasing_encoded)}')
            # print(f"当前信号数量：{Y_increasing_encoded.size}")
            user_input = input("是否继续预测和训练？输入 1 继续，输入 0 停止：")
            print("---------------------------")
            if user_input == '0':
                continue_training = False
            elif user_input == '1':
                continue_training = True
            else:
                print("无效输入，默认继续训练")

    # 计算正确识别率
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    accuracy = np.mean(predictions == true_labels)


    # 输出结果
    print(f"模型在抽取出来的预测信号上的正确识别率: {accuracy:.4f}")