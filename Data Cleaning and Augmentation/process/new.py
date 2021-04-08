import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba

# 1.训练数据聚合

def data_merge(query_path, reply_path, train:bool):
    query = pd.read_csv(query_path, sep='\t', header=None)
    query.columns = ['id', 'query']
    reply = pd.read_csv(reply_path, sep='\t', header=None)
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

    #new_data['query_cut'] = new_data['query'].apply(lambda x: ' '.join(list(jieba.cut(str(x)))))
    #new_data['reply_cut'] = new_data['reply'].apply(lambda x: ' '.join(list(jieba.cut(str(x)))))
    new_data = new_data.dropna()
    return new_data

train_data = data_merge('data/train_query_augmented_3x.tsv','data/train_reply_augmented_3x.tsv', train=True)

#test_data = data_merge('./data/test.query.tsv','./data/test.reply.tsv',train=False)

train_data.head()

#test_data.head()

train_data.to_csv('./data/train_augmented.csv', index=False)
#test_data.to_csv('./data/test_new.csv', index=False)