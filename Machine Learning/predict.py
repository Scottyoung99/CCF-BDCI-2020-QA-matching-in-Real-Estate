import tensorflow as tf
import joblib
import jieba
from config.lr_config import LrConfig
from lr_model import LrModel
import pandas as pd
import numpy as np

def pre_data(data, config):
    """分词去停用词"""
    stopwords = list()
    text_list = list()
    with open(config.stopwords_path, 'r', encoding='utf-8') as f:
         stopwords=f.readlines()
    seg_text = jieba.cut(data)
    text = [word for word in seg_text if word not in stopwords]
    text_list.append(' '.join(text))
    return text_list


def read_categories():
    """读取类别"""
    with open(config.categories_save_path, 'r', encoding='utf-8') as f:
        categories = f.readlines()
    return categories[0].split('|')


def predict_line(data, categories):
    """预测结果"""
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess=session, save_path=config.lr_save_path)
    y_pred_cls = session.run(model.y_pred_cls, feed_dict={model.x: data})
    return categories[y_pred_cls[0]]


if __name__ == "__main__":
    with open('./data/test.txt', encoding='utf-8') as f1:
        test_data = f1.readlines()
    id_col = []
    replysort_id_col = []
    res = []
    itr = 0
    config = LrConfig()
    tfidf_model = joblib.load(config.tfidf_model_save_path)
    categories = read_categories()

    for line in test_data:
        id, query, reply, reply_sort, query_cut, reply_cut = line.split('\t', 5)

        # text = query + reply
        # itr += 1
        # print("itr=" + str(itr), end='  ')
        # print(text, end='  ')
        # line = pre_data(text, config)
        # X_test = tfidf_model.transform(line).toarray()
        # model = LrModel(config, len(X_test[0]))

        # print(predict_line(X_test, categories))
        id_col.append(id)
        replysort_id_col.append(reply_sort)

        # ans = predict_line(X_test, categories)

        ans = 0
        print(ans)
        res.append(ans)
        # if ans == "零":
        #     res.append(0)
        # else:
        #     res.append(1)



    np.array(id_col)
    np.array(replysort_id_col)
    submit = pd.DataFrame()
    np.array(res)
    submit['id'] = id_col
    submit['reply_sort'] = replysort_id_col
    submit['label'] = res

    submit.to_csv('./data/submit1.tsv', sep='\t', header=False, index=False)


