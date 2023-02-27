import torch
import torch.nn as nn
from transformers import BertModel
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

class BertRNN(nn.Module):
    def __init__(self, num_classes):
        super(BertRNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # for param in self.bert.parameters():
        #     param.requires_grad = True
         # 定义RNN层
        self.gru  = nn.GRU(input_size=768, hidden_size=768, num_layers=2, 
                          bidirectional=True, batch_first=True)
        
        # 线性层用于分类
        self.fc = nn.Linear(768 * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        # 将 BERT 输出输入到 RNN 中
        rnn_output, _ = self.gru(bert_output)
        
        # 取 RNN 输出的最后一层，拼接正向和反向的隐藏状态
        last_rnn_output = torch.cat((rnn_output[:, -1, :self.gru.hidden_size], 
                                     rnn_output[:, 0, self.gru.hidden_size:]), dim=-1)
        
        # 输入分类层
        logits = self.fc(last_rnn_output)
        return logits

class BertCNN(nn.Module):
    def __init__(self, num_classes):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self,  input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # shape of outputs: (batch_size, sequence_length, hidden_size)
        outputs = outputs.last_hidden_state.transpose(1, 2)
        # shape of outputs: (batch_size, hidden_size, sequence_length)
        outputs = F.relu(self.conv1(outputs))
        outputs = F.relu(self.conv2(outputs))
        outputs = F.max_pool1d(outputs, kernel_size=outputs.size(-1)).squeeze(-1)
        outputs = self.fc(outputs)
        return outputs

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        label=[int(x) for x in label]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )
        label = torch.tensor(label)
        return inputs['input_ids'].squeeze(0),inputs['attention_mask'].squeeze(0),label

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for i, (input_ids,attention_mask, labels) in tqdm(enumerate(train_loader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        outputs2 = torch.squeeze(outputs)  # 去掉不必要的维度
        preds = torch.argmax(outputs2, dim=1)  # 找到最大预测值对应的类别
        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
        total_acc += preds.eq(labels.view_as(preds)).sum().item()
        # a=1
    return total_loss / len(train_loader.dataset), total_acc / len(train_loader.dataset)

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for i, (input_ids,attention_mask, labels) in tqdm(enumerate(val_loader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            outputs2 = torch.squeeze(outputs)  # 去掉不必要的维度
            preds = torch.argmax(outputs2, dim=1)  # 找到最大预测值对应的类别
            loss = criterion(outputs, labels.squeeze(1))
            # loss.backward()
            # optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            total_acc += preds.eq(labels.view_as(preds)).sum().item()
        return total_loss / len(val_loader.dataset), total_acc / len(val_loader.dataset)

train_texts=[]
train_labels=[]
val_texts=[]
val_labels=[]
with open('train.txt', 'r', encoding='UTF-8') as f:
    lines=f.readlines()
    for line in lines:
        lin = line.strip()
        if not lin:
            continue
        content, label = lin.split('\t')
        train_texts.append(content)
        train_labels.append(label)
        # print(label)

with open('test.txt', 'r', encoding='UTF-8') as f:
    lines=f.readlines()
    for line in lines:
        lin = line.strip()
        if not lin:
            continue
        content, label = lin.split('\t')
        val_texts.append(content)
        val_labels.append(label)

# 定义训练和测试参数
train_dataset = CustomDataset(train_texts, train_labels)
# train_dataset = CustomDataset(val_texts, val_labels)
val_dataset = CustomDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# model = BertRNN(num_classes=10)
# model = BertClassifier(num_classes=10)
model = BertCNN(num_classes = 10)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()

num_epochs = 3

# 训练和测试循环
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    print(f'Epoch {epoch+1} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')