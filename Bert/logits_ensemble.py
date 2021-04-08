import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jieba
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoModel, BertForNextSentencePrediction, BertTokenizer, BertForQuestionAnswering
import random
from sklearn.model_selection import KFold
import os
from main_cv import zyDataset, Bert_Fc


seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'




def test(model, test_iter):
    res = []
    # model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        for idx, te_batch in enumerate(test_iter):
            input_ids = te_batch['input_ids'].to(device)
            attention_mask = te_batch['attention_mask'].to(device)
            token_type_ids = te_batch['token_type_ids'].to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            # output = torch.argmax(output, axis=1).tolist()
            output = output.tolist()
            res += output
    return np.array(res)



if __name__ == '__main__':
    max_seq_len = 100
    hidden_size = 768
    n_class = 2
    batch_size = 16
    test_data = pd.read_csv('./data/test.csv')
    res_proba = np.zeros((len(test_data), 2))
    model_name = ["bert-base-chinese", "hfl/chinese-roberta-wwm-ext", "hfl/chinese-bert-wwm-ext", "hfl/chinese-bert-wwm"]
    weights = [0.2, 0.25, 0.35, 0.2]
    num_model = len(model_name)

    for i in range(num_model):
        #model_name = 'hfl/chinese-bert-wwm-ext'

        tokenizer = BertTokenizer.from_pretrained(model_name[i])

        test_encodings = tokenizer(test_data['query'].tolist(), test_data['reply'].tolist(), truncation=True,
                                    padding=True, max_length=max_seq_len)
        test_dataset = zyDataset(test_encodings, None, test=True)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = Bert_Fc(model_name[i], max_seq_len, hidden_size, n_class)
        model.to(device)
        path = './ensemble/model-' + str(i) + '.ckpt'
        #checkpoint = torch.load(path)
        #model.load_state_dict(checkpoint['model'])
        model.load_state_dict(torch.load(path))
        res = test(model, test_iter)
        res_proba += weights[i]*res
        torch.cuda.empty_cache()

    # res_proba = res_proba / num_model
    res = np.argmax(res_proba, axis=1)
    submit = pd.DataFrame()
    submit['id'] = test_data['id']
    submit['reply_sort'] = test_data['reply_sort']
    submit['label'] = res
    submit.to_csv('./results/submit_4model.tsv', sep='\t', header=False, index=False)