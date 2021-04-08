import torch
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
import pandas as pd

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

feature_cols = ['query', 'reply']
label_cols = ['label']
train_x, train_y = train_data[feature_cols].loc[:], train_data[label_cols].loc[:]

path1 = './model/btbase-classifier.ckpt'
path2 = './model/chinese-bert-wwm-classifier.ckpt'
path3 = './model/chinese-bert-wwm-ext-classifier.ckpt'
path4 = './model/chinese-robert-wwm-ext-classifier.ckpt'
path5 = ''

# file1 = pd.read_csv(r'./submits/submit_cv_btbase.tsv', sep='\t', header=None)
# file2 = pd.read_csv(r'./submits/submit_cv_bt_wwm.tsv', sep='\t', header=None)
# file3 = pd.read_csv(r'./submits/submit_cv_bt_wwm_ext.tsv', sep='\t', header=None)
# file4 = pd.read_csv(r'./submits/submit_cv_rbt_wwm_ext.tsv', sep='\t', header=None)
# file5 = pd.read_csv(r'./submits/submit_cv_bert_wwm_ext_translate.tsv', sep='\t', header=None)
# file6 = pd.read_csv(r'./submits/submit_cv_bt_wwm_ext.tsv', sep='\t', header=None)

file1 = pd.read_csv(r'./submits/submit_cv_bert_wwm_ext_translate.tsv', sep='\t', header=None)
file2 = pd.read_csv(r'./submits/submit_cv_rbt_large-trans.tsv', sep='\t', header=None)
file3 = pd.read_csv(r'./submits/submit_cv_bt_wwm_ext.tsv', sep='\t', header=None)
file4 = pd.read_csv(r'./submits/submit_cv_btbase.tsv', sep='\t', header=None)

'''Voting'''
res = []
for i in range(file1.shape[0]):
    ans = 0
    ans += file1.iloc[i][2]
    ans += file2.iloc[i][2]
    ans += file3.iloc[i][2]
    ans += file4.iloc[i][2]
    if ans >= 2:
        res.append(1)
    else:
        res.append(0)

submit = pd.DataFrame()
submit['id'] = test_data['id']
submit['reply_sort'] = test_data['reply_sort']
submit['label'] = res
submit.to_csv('./data/submit202.tsv', sep='\t', header=False, index=False)

# 101 不加权
# 102 复用Robert
# 103 复用bert
# 104 如果有人支持最好的说是1就是1
# 105 只用四个原始的，如果>=2个就判1
# 106 处理过前15000的
# 107 处理过前45000的
# 110 7个模型>3 复用最好的和最新的
# 111 7个模型>=3 复用最好的和最新的
# 112 6个模型>3 复用最好的

'''Stacking'''
model1 = torch.load(path1)
model1._estimator_type = "classifier"
model2 = torch.load(path2)
model2._estimator_type = "classifier"
model3 = torch.load(path3)
model3._estimator_type = "classifier"
model4 = torch.load(path4)
model4._estimator_type = "classifier"
eclf = VotingClassifier(estimators=[model1, model2, model3, model4], voting='hard')
eclf.fit(train_x,train_y)
eclf.predict(test_data)

# 模型融合中使用到的各个单模型
clfs = [model1, model2 , model3, model4]

# 切分一部分数据作为测试集
X, X_predict, y, y_predict = train_test_split(train_x, train_y, test_size=0.3, random_state=2020)

dataset_stack_train = np.zeros((X.shape[0], len(clfs)))
dataset_stack_test = np.zeros((X_predict.shape[0], len(clfs)))

# 5折stacking
n_splits = 5
skf = StratifiedKFold(n_splits)
skf = skf.split(X, y)

for j, clf in enumerate(clfs):
    # 依次训练各个单模型
    dataset_stack_test_j = np.zeros((X_predict.shape[0], 5))
    for i, (train, test) in enumerate(skf):
        # 5-Fold交叉训练，使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_stack_train[test, j] = y_submission
        dataset_stack_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    # 对于测试集，直接用这k个模型的预测值均值作为新的特征。
    dataset_stack_test[:, j] = dataset_stack_test_j.mean(1)
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_stack_test[:, j]))

clf = LogisticRegression(solver='lbfgs')
clf.fit(dataset_stack_train, y)
y_submission = clf.predict(dataset_stack_test)[:, 1]

'''blending'''
model1 = torch.load(path1)
model1._estimator_type = "classifier"
model2 = torch.load(path2)
model2._estimator_type = "classifier"
model3 = torch.load(path3)
model3._estimator_type = "classifier"
model4 = torch.load(path4)
model4._estimator_type = "classifier"
eclf = VotingClassifier(estimators=[model1, model2, model3, model4], voting='hard')
eclf.fit(train_x,train_y)
eclf.predict(test_data)

# 模型融合中使用到的各个单模型
clfs = [model1, model2 , model3, model4]

# 切分一部分数据作为测试集
X, X_predict, y, y_predict = train_test_split(train_x, train_y, test_size=0.3, random_state=2020)

# 切分训练数据集为d1,d2两部分
X_d1, X_d2, y_d1, y_d2 = train_test_split(X, y, test_size=0.5, random_state=2020)
dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
dataset_d2 = np.zeros((X_predict.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    # 依次训练各个单模型
    clf.fit(X_d1, y_d1)
    y_submission = clf.predict_proba(X_d2)[:, 1]
    dataset_d1[:, j] = y_submission
    # 对于测试集，直接用这k个模型的预测值作为新的特征。
    dataset_d2[:, j] = clf.predict_proba(X_predict)[:, 1]
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_d2[:, j]))

# 融合使用的模型
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_d1, y_d2)
y_submission = clf.predict(dataset_d2)[:, 1]


# clfs=4
#
# submit_btbase=pd.read_csv('./results/submit_cv_btbase.tsv', sep='\t')
#
# dataset_blend_train = np.zeros((submit_btbase.shape[0], len(clfs)))
# dataset_blend_test = np.zeros((1, len(clfs)))
#
# dataset_stacking_train[:,0]=
#
#
# clf = LogisticRegression(solver='lbfgs')
#
#
#
# clf.fit(dataset_stacking_train, y)
# y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
#
# print("Val auc Score of Stacking: %f" % (roc_auc_score(y_predict, y_submission)))

