import os

pwd_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# os.path.dirname(path)--去掉文件名，返回目录
# _file_--表示当前文件的完整路径
# os.path.abspath(__file__)--返回当前完整路径
# print(pwd_path)  E:\NLP\nlp-beginner-finish-master\task1
class LrConfig(object):
    #  训练模型用到的路径
    dataset_path = os.path.join(pwd_path + '/data' + "/train.txt")
    stopwords_path = os.path.join(pwd_path + '/data' + "/stopwords.txt")
    tfidf_model_save_path = os.path.join(pwd_path + '/model' + "/tfidf_model.m")
    categories_save_path = os.path.join(pwd_path + '/data' + '/label.txt') # 分类类别数据
    lr_save_dir = os.path.join(pwd_path + '/model' + "/checkpoints")
    lr_save_path = os.path.join(lr_save_dir, 'best_validation')
    #  变量
    num_epochs = 100  # 总迭代轮次
    num_classes = 2  # 类别数
    print_per_batch = 10  # 每多少轮输出一次结果
