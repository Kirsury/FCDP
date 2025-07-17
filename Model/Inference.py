import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel

# === 1. 定义模型结构 ===
# 必须和训练时一致
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

# === 2. 构造用于推理的数据集 ===
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

# === 3. 加载模型 ===
def load_model(path='model.pt', map_location='cpu'):
    model = DefectPredictionModel()
    state_dict = torch.load(path, map_location=map_location)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])  # 如果你保存了字典
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model

# === 4. 执行批量推理 ===
def run_inference(model, dataloader, device='cpu'):
    model.to(device)
    results = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            results.extend(preds.cpu().numpy())

    return results

# === 5. 示例执行 ===
if __name__ == '__main__':
    # 模拟一些测试数据
    input_data = [[0.1] * 10, [0.2] * 10, [0.3] * 10]  # 3条数据，每条10维
    dataset = MyDataset(input_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # 加载模型并推理
    model = load_model('model.pt', map_location='cpu')
    predictions = run_inference(model, dataloader)

    print("Predictions:", predictions)
