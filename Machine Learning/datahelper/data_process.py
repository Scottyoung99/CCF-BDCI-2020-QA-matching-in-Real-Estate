from config.lr_config import LrConfig
from sklearn import model_selection    # 用于数据集划分
from sklearn.feature_extraction.text import TfidfVectorizer # tf-idf算法
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # 类别编码
import joblib  # 保存模型
import jieba  # 中文分词库
import numpy as np

config = LrConfig()

# 数据处理
class DataProcess(object):
    def __init__(self, dataset_path=None, stopwords_path=None, model_save_path=None):
        self.dataset_path = dataset_path  # 训练数据集
        self.stopwords_path = stopwords_path # 停用词数据
        self.model_save_path = model_save_path

    def read_data(self):
        """读取数据"""
        with open(self.dataset_path, encoding='utf-8') as f1:
            data = f1.readlines()  # 返回列表
        with open(self.stopwords_path, encoding='utf-8') as f2:
            stopwords = f2.readlines()
        return data, stopwords

    def save_categories(self, data, save_path):
        """将文本的类别写到本地"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('|'.join(data)) #将序列中的元素以指定的字符连接生成一个新的字符串

    def pre_data(self, data, stopwords, test_size=0.2):
        """数据预处理"""
        label_list = list()
        text_list = list()
        i = 0
        for line in data:
            query, reply, label = line.split('\t', 2)
            # 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
            # print(label)
            text = query + reply
            label = label[0]
            # print(label)
            # print(text)
            # seg_query = [word for word in jieba.cut(query) if word not in stopwords]
            # seg_reply = [word for word in jieba.cut(reply) if word not in stopwords]
            seg_text = [word for word in jieba.cut(text) if word not in stopwords]
            text_list.append(' '.join(seg_text))
            # row = [' '.join(seg_query), ' '.join(seg_reply)]
            # text_list.append(row)
            label_list.append(label)  # 获得标签列表

        # 标签转化为one-hot格式（先文本转数字，后数字转one-hot)
        encoder_nums = LabelEncoder()   #获取一个LabelEncoder
        label_nums = encoder_nums.fit_transform(label_list) #训练LabelEncoder并将标签值标准化
        print(label_nums)
        categories = list(encoder_nums.classes_) # 获取标签值（文本类别）
        print(categories)
        self.save_categories(categories, config.categories_save_path)
        label_nums = np.array([label_nums]).T # 标准化后的标签向量
        encoder_one_hot = OneHotEncoder()
        label_one_hot = encoder_one_hot.fit_transform(label_nums)
        label_one_hot = label_one_hot.toarray()
        print(label_one_hot)
        # 流出法
        print(text_list)
        return model_selection.train_test_split(text_list, label_one_hot, test_size=test_size, random_state=1024)
        # random_state就是为了保证程序每次运行都分割到一样的训练集和测试集(随机种子)
        # 利用train_test_split进行训练集和测试机随机分开
    # TODO:后续做
    def get_bow(self):
        """提取词袋模型特征"""
        pass

    # TODO:这里可能出现维度过大，内存不足的问题，目前是去除低频词解决，可以做lda或者pca降维（后续做）
    #TFIDF算法：根据字词的在文本中出现的次数和在整个语料中出现的文档频率来计算一个字词在整个语料中的重要程度
    #优点是能过滤掉一些常见的却无关紧要本的词语，同时保留影响整个文本的重要字词。
    def get_tfidf(self, X_train, X_test):
        """提取tfidf特征"""
        vectorizer = TfidfVectorizer(min_df=10)  # min_df用于删除不经常出现的术语
        vectorizer.fit_transform(X_train)
        # print("*************")
        # print(vectorizer.get_feature_names()) # 统计关键词
        X_train_vec = vectorizer.transform(X_train)  # 前提是fit()或fit_transform()学到过
        # print(X_train_vec)
        X_test_vec = vectorizer.transform(X_test)
        # print(X_test_vec)
        return X_train_vec, X_test_vec, vectorizer

    # TODO:后续做
    def get_word2vec(self):
        """提取word2vec特征"""
        pass

    def provide_data(self):
        """提供数据"""
        data, stopwords = self.read_data()
        #  1、提取bag of word参数
        #  2、提取tf-idf特征参数
        X_train, X_test, y_train, y_test = self.pre_data(data, stopwords, test_size=0.2)
        X_train_vec, X_test_vec, vectorizer = self.get_tfidf(X_train, X_test)
        joblib.dump(vectorizer, self.model_save_path)# 模型保存
        #  3、提取word2vec特征参数
        return X_train_vec, X_test_vec, y_train, y_test

    def batch_iter(self, x, y, batch_size=64):
        """迭代器，将数据分批传给模型"""
        data_len = len(x)
        num_batch = int((data_len-1)/batch_size)+1
        # 打乱
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start_id = i*batch_size
            end_id = min((i+1)*batch_size, data_len)
            yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]



