import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# 忽略不必要的警告，保持输出整洁
warnings.filterwarnings("ignore")

# ==============================================
# 实验内容1：利用 pandas 库从本地读取 iris 数据集
# ==============================================
# 第一步：先生成本地iris数据文件（如果已有老师给的iris.data可跳过此步）
iris_sk = load_iris()
df_save = pd.DataFrame(iris_sk.data, columns=iris_sk.feature_names)
df_save['species'] = iris_sk.target
df_save.to_csv('iris.data', index=False, header=False, encoding='utf-8')

# 第二步：pandas 读取本地 iris.data 文件
# 原始数据无表头，手动指定列名
col_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类别']
iris_local = pd.read_csv('iris.data', names=col_names)
print("="*60)
print("【实验1】pandas本地读取iris数据集成功")
print(f"数据集形状：{iris_local.shape}")
print("前5行数据：")
print(iris_local.head())
print("="*60)

# ==============================================
# 实验内容2：从 scikit-learn 库中直接加载 iris 数据集
# ==============================================
iris = load_iris()
X = iris.data    # 特征数据（4个属性）
y = iris.target  # 标签数据（3类鸢尾花）
print("\n【实验2】sklearn加载iris数据集成功")
print(f"特征集形状：{X.shape}")
print(f"标签类别：{list(iris.target_names)}")
print("="*60)

# ==============================================
# 实验内容3：实现五折交叉验证进行模型训练
# ==============================================
# 1. 定义五折交叉验证迭代器：打乱数据、固定随机种子保证结果可复现
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. 定义分类模型（逻辑回归，适配多分类任务）
model = LogisticRegression(solver='liblinear', random_state=42)

# 3. 五折交叉验证预测：自动完成5次训练+预测，返回所有样本的预测结果
y_pred = cross_val_predict(model, X, y, cv=kf)
print("\n【实验3】五折交叉验证完成")
print(f"折数：{kf.get_n_splits(X)} 折")
print("="*60)

# ==============================================
# 实验内容4：计算并输出模型评估指标
# ==============================================
accuracy = accuracy_score(y, y_pred)
# 多分类任务使用 macro 宏平均，对每个类别的指标取平均值
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')
f1 = f1_score(y, y_pred, average='macro')

print("\n【实验4】模型评估结果")
print(f"准确度（Accuracy）：{accuracy:.4f}")
print(f"精  度（Precision）：{precision:.4f}")
print(f"召回率（Recall）：  {recall:.4f}")
print(f"F1 值 （F1-score）： {f1:.4f}")
print("="*60)
