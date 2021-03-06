import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import jieba
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import BertModel, AutoModel, BertForNextSentencePrediction, BertTokenizerFast, BertForQuestionAnswering
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelWithLMHead
import torch.nn.functional as F

seed = 3
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Bert_Fc(nn.Module):
    def __init__(self, model_name, max_seq_len, hidden_size, n_class):
        super(Bert_Fc,self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        self.maxpool = nn.MaxPool1d(max_seq_len)
        self.avgpool = nn.AvgPool1d(max_seq_len)
        self.linear1 = nn.Linear(3*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_class)
        self.tanh = nn.Tanh()

    def forward(self, x, mask, token_type_ids):
        bert_out = self.bert(x, attention_mask=mask, token_type_ids=token_type_ids)
        sentence_emb = bert_out.last_hidden_state[:, 0, :]
        bert_out = bert_out.last_hidden_state.permute(0,2,1).contiguous()
        maxpool_out = self.maxpool(bert_out).squeeze(2)
        avgpool_out = self.avgpool(bert_out).squeeze(2)
        outputs = torch.cat((maxpool_out, avgpool_out, sentence_emb), 1)
        outputs = self.tanh(self.linear1(outputs))
        outputs = self.linear2(outputs)
        return outputs


class zyDataset(Dataset):
    def __init__(self, encodings, labels, test=False):
        self.encodings = encodings
        self.len = len(self.encodings['input_ids'])
        self.labels = labels
        self.test = test

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if not self.test:
            item['labels'] = torch.tensor(self.labels['label'].tolist()[idx])
        return item

    def __len__(self):
        return self.len


def train(model, optimizer, loss_func, train_iter, epoch):
    model.train()
    total_loss = 0
    for idx, tr_batch in enumerate(train_iter):
        input_ids = tr_batch['input_ids'].to(device)
        token_type_ids = tr_batch['token_type_ids'].to(device)
        attention_mask = tr_batch['attention_mask'].to(device)
        label = tr_batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask, token_type_ids)
        loss = loss_func(output, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        log_inte = 30
        if idx%log_inte == 0 and idx != 0:
            print('train-loss', total_loss/log_inte)
            total_loss = 0
        return model

def evaluate(model, val_iter):
    model.eval()
    res = []
    labels = []
    total_loss = 0
    with torch.no_grad():
        for idx, val_batch in enumerate(val_iter):
            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            token_type_ids = val_batch['token_type_ids'].to(device)
            label = val_batch['labels'].to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(output,label)
            total_loss += loss.item()
            output = torch.argmax(output, axis=1).tolist()
            label = label.tolist()
            labels += label
            res += output

        res = np.array(res)
        labels = np.array(labels)
        accuracy = metrics.accuracy_score(res, labels)
        precision = metrics.precision_score(res, labels)
        recall = metrics.recall_score(res, labels)
        f1 = metrics.f1_score(res, labels)
        print(accuracy, precision, recall, f1)
        return total_loss/len(val_iter), f1



def test(model, test_iter):
    res = []
    model.eval()
    with torch.no_grad():
        for idx, test_batch in enumerate(test_iter):
            input_ids = test_batch['input_ids'].to(device)
            attention_mask = test_batch['attention_mask'].to(device)
            token_type_ids = test_batch['token_type_ids'].to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            output = torch.argmax(output, axis=1).tolist()
            res += output

    return np.array(res)


if __name__ == '__main__':
    max_seq_len = 100
    hidden_size = 768
    n_class = 2
    batch_size = 16
    lstm_hidden = 256
    num_layers = 2
    model_name = 'hfl/chinese-bert-wwm'

    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

    feature_cols = ['query', 'reply']
    label_cols = ['label']

    train_x, val_x, train_y, val_y = train_test_split(train_data[feature_cols], train_data[label_cols], test_size=0.2, random_state=seed)

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_x['query'].tolist(), train_x['reply'].tolist(), truncation=True, padding=True, max_length=max_seq_len)
    val_encodings = tokenizer(val_x['query'].tolist(), val_x['reply'].tolist(), truncation=True, padding=True, max_length=max_seq_len)
    test_encodings = tokenizer(test_data['query'].tolist(), test_data['reply'].tolist(), truncation=True, padding=True, max_length=max_seq_len)

    train_dataset = zyDataset(train_encodings, train_y[label_cols].reset_index(drop=True))
    val_dataset = zyDataset(val_encodings, val_y[label_cols].reset_index(drop=True))
    test_dataset = zyDataset(test_encodings, None, test=True)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Bert_Fc(model_name, max_seq_len=max_seq_len, hidden_size=hidden_size, n_class=n_class)
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)
    save_path = './model/bert_classifier.ckpt'
    epochs = 20
    min_loss = float('inf')
    f1_init = 0
    last_improve = 0
    total_batch = 0
    early_stop = 5

    for epoch in range(1, epochs):
        print('[{}/{}]'.format(epoch, epochs))
        model = train(model, optimizer, loss_func, train_iter, epoch)
        val_loss, f1 = evaluate(model, val_iter)
        print('test-loss',val_loss)

        total_batch = epoch
        if f1 > f1_init:
            f1_init = f1
            print('saving')
            torch.save(model.state_dict(), save_path)
            last_improve = total_batch

        if (total_batch - last_improve) >= early_stop:
            break

    res = test(model, test_iter)
    submit = pd.DataFrame()
    submit['id'] = test_data['id']
    submit['reply_sort'] = test_data['reply_sort']
    submit['label'] = res

    submit.to_csv('./data/submit1.tsv', sep='\t', header=False, index=False)





















