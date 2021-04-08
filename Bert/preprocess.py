import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
import seaborn as sns


# 1.训练数据聚合

def data_merge(query_path, reply_path, train:bool):
    query = pd.read_csv(query_path, sep='\t', header=None, encoding='utf-8')
    query.columns = ['id', 'query']
    reply = pd.read_csv(reply_path, sep='\t', header=None, encoding='utf-8')
    if train:
        reply.columns = ['id','reply_sort','reply','label']
    else:
        reply.columns = ['id','reply_sort','reply']

    new_data = pd.DataFrame()
    new_data['id'] = reply['id']
    new_data['query'] = [query.iloc[i]['query'] for i in reply['id']]
    new_data['reply'] = reply['reply']
    new_data['reply_sort'] = reply['reply_sort']
    if train:
        new_data['label'] = reply['label']

    new_data['query_cut'] = new_data['query'].apply(lambda x: ' '.join(list(jieba.cut(str(x)))))
    new_data['reply_cut'] = new_data['reply'].apply(lambda x: ' '.join(list(jieba.cut(str(x)))))

    new_data = new_data.dropna()
    return new_data

train_data = data_merge('./data/train.query.tsv','./data/train.reply.tsv', train=True)
# train_data = data_merge('./data/train_query_merged_clean_5x.tsv','./data/train_reply_merged_clean.tsv', train=True)
test_data = data_merge('./data/test.query.tsv','./data/test.reply.tsv',train=False)

train_data.head()
test_data.head()

train_data.to_csv('./data/train_argument.tsv', index=False)
test_data.to_csv('./data/test.csv', index=False)

# 2.数据分析

ds = train_data['label'].value_counts()
ds.plot.bar()
plt.title('label distribution')
plt.show()

print('问题数：{}, 答案数：{}'.format(len(train_data['query'].unique()), len(train_data['reply'].unique())))

train_data['query_cut'] = train_data['query'].apply(lambda x: ' '.join(list(jieba.cut(str(x)))))
train_data['reply_cut'] = train_data['reply'].apply(lambda x: ' '.join(list(jieba.cut(str(x)))))

train_data.head()

train_data['query_len'] = train_data['query'].apply(lambda x:len(x))
train_data['reply_len'] = train_data['reply'].apply(lambda x:len(x))

print(train_data['query_len'].describe(), train_data['reply_len'].describe())

ds = train_data['query_len']
g = sns.distplot(ds)
plt.title('length of reply distribution')
plt.show()

train_data['reply_sort'].value_counts()

train_data.isnull().sum()

# 3.测试集分析

test_data['query_len'] = test_data['query_cut'].apply(lambda x:len(x))
test_data['reply_len'] = test_data['reply_cut'].apply(lambda x:len(x))

ds = test_data['query_len']
g = sns.distplot(ds)
plt.title('length of query distribution')
plt.show()

ds = test_data['reply_len']
g = sns.distplot(ds)
plt.title('length of query distribution')
plt.show()

test_data['query_len'].describe()
test_data['reply_len'].describe()















