import pandas as pd
import numpy as np
import jieba

train_left = pd.read_csv('./data/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','query']
train_right = pd.read_csv('./data/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sort','reply','label']
df_train = train_left.merge(train_right, how='left')
df_train['reply'] = df_train['reply'].fillna('好的')
test_left = pd.read_csv('./data/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','query']
test_right =  pd.read_csv('./data/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sort','reply']
df_test = test_left.merge(test_right, how='left')

print(df_train.head())
#df_train.to_csv('./data/train.csv',index=False,encoding='utf-8_sig')
#df_test.to_csv('./data/test.csv', index=False)

