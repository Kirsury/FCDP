import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# 模型定义
import torch
import torch.nn as nn
from transformers import RobertaModel


class DefectPredictionModel(nn.Module):
    def __init__(self, pretrained_model_name='microsoft/codebert-base'):
        super(DefectPredictionModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name)

        # 正确冻结底部6层
        for name, param in self.encoder.named_parameters():
            if name.startswith("encoder.encoder.layer.") and int(name.split(".")[3]) < 6:
                param.requires_grad = False

        # 打印被冻结的参数名
        for name, param in model.encoder.named_parameters():
            if not param.requires_grad:
                print("Frozen:", name)

        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 2)  # 二分类：缺陷 / 非缺陷
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]  # [CLS] token 表征
        logits = self.classifier(cls_rep)
        return logits


# 损失函数
def compute_loss(logits, targets):
    return F.cross_entropy(logits, targets)

# 评估指标
def compute_metrics(logits, targets):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
    try:
        prob = F.softmax(torch.tensor(logits), dim=1)[:, 1].detach().cpu().numpy()
        auc = roc_auc_score(targets, prob)
    except:
        auc = 0.0  # AUC 无法计算（如全1或全0）

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from collections import Counter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例数据
source_train_data = [
    {"code": "def foo(x): return x + 1", "label": 0},
    {"code": "def bar(x): if x: return 1 else: return 0", "label": 1},
    # ...
]
source_test_data = [
    {"code": "def baz(x): return x * 2", "label": 0},
    # ...
]

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# 自定义 Dataset
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data[idx]['code']
        label = self.data[idx]['label']
        encoding = self.tokenizer(code, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = CodeDataset(source_train_data, tokenizer)
test_dataset = CodeDataset(source_test_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 🟡 类不平衡处理：加权 CrossEntropyLoss
label_counts = Counter([item['label'] for item in source_train_data])
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in range(len(label_counts))]  # inverse freq
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# 模型 & 优化器
model = DefectPredictionModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 🔁 训练
for epoch in range(5):
    model.train()
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: train loss = {loss.item():.4f}")

# 🧪 在 test set 上评估平均指标
model.eval()
all_logits = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        all_logits.append(logits)
        all_labels.append(labels)

# 聚合并评估
all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.cat(all_labels, dim=0)
test_metrics = compute_metrics(all_logits, all_labels)

print("Test Metrics:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")
