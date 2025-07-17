import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from collections import Counter
from datetime import datetime


"""
缺陷预测模型设计、实现和测试评估
"""

# 1. 配置设备
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except:
    npu_available = False

device = torch.device("npu:6" if npu_available else "cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备{device}")

# 2. 日志和模型保存路径设置
"""
记录训练、测试评估过程中的关键信息，以便于回顾和分析。
每条日志信息应包含时间信息（年月日时分秒）。
"""
def current_time():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

import logging

log_path = f"log_{datetime.now().isoformat()}.txt"
logging.basicConfig(filename=log_path, level=logging.INFO, format='[%(asctime)s] %(message)s')

def log(msg):
    print(msg)
    logging.info(msg)




# 3. 超参数设置
num_epoch = 5
batch_size = 16
max_len = 512
learning_rate = 2e-5
freeze_layers = 6
model_save_dir = "./saved_models"
os.makedirs(model_save_dir, exist_ok=True)


# 4. 训练集和测试集路径
train_path = "./Dataset/cleaned/249.json"
test_path = "./Dataset/cleaned/250.json"

# 5. 加载数据
def load_jsonlines(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [{"code": item["code"], "label": item["label"]} for item in data]

print(f"✅ 加载训练数据集...")
train_data = load_jsonlines(train_path)
print(f"✅ 加载测试数据集...")
test_data = load_jsonlines(test_path)

# 6. 加载tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")



# 7. 自定义 Dataset
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

print(f"✅ 创建训练集和测试集...")
train_dataset = CodeDataset(train_data, tokenizer)
test_dataset = CodeDataset(test_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 8. 解决类不平衡
label_counts = Counter([item['label'] for item in train_data])
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in range(len(label_counts))]  # inverse freq
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
print(f"✅ 类权重: {class_weights}")

# 9. 定义模型
class DefectPredictionModel(nn.Module):
    def __init__(self, pretrained_model_name='microsoft/codebert-base', freeze_layers=6):
        super(DefectPredictionModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name)

        # 正确冻结底部6层
        for name, param in self.encoder.named_parameters():
            if name.startswith("encoder.layer."):
                layer_num = int(name.split(".")[2])
                if layer_num < freeze_layers:
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

print(f"✅ 创建模型...")
model = DefectPredictionModel(freeze_layers=freeze_layers).to(device)
optimizer = torch.optim.AdamW(lambda p: p.requires_grad, filter(model.parameters()), lr=2e-5)


# 10. 评估指标
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

# 11. 日志init

print(f"✅ 初始化日志...")



# 12. 🔁 训练主循环
print(f"✅ 开始训练...")
for epoch in range(5):
    """
    每轮的model都要保存下来，模型名要包含当前的年月日时分秒信息和epoch id信息以使名称唯一化和便于管理化
    """
    model.train()
    total_loss, num_steps = 0, 0
    for batch in tqdm.tqdm(train_dataloader, "Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        num_steps += 1
    avg_loss = total_loss / num_steps

    print(f"Epoch {epoch + 1}/{num_epoch} | train loss: {avg_loss.item():.4f}")
    model_path = os.path.join(model_save_dir, f"model_{current_time()}_epoch{epoch + 1}.pt")
    torch.save(model.state_dict(), model_path)
    log(f"✅ 模型已保存到 {model_path}")

    # write train log

# 13. 🧪 在 test set 上评估平均指标
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

"""
保存logits和labels，以及对应的code，文件名应包含相应的模型名和epoch num信息
以便于后续分析具体的point
"""

print("Test Metrics:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

output_path = f"outputs/predict_{current_time()}_epoch{epoch+1}.jsonl"
os.makedirs("outputs", exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    for logit, label, item in zip(all_logits, all_labels, test_data):
        prob = F.softmax(logit, dim=0).tolist()
        entry = {
            "code": item["code"],
            "label": int(label),
            "prob_0": prob[0],
            "prob_1": prob[1]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

log(f"✅ 保存测试输出至 {output_path}")
