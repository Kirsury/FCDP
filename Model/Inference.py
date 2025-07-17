import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

# === 模型结构定义 ===
class DefectPredictionModel(nn.Module):
    def __init__(self, pretrained_model_name='microsoft/codebert-base', freeze_layers=6):
        super(DefectPredictionModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name)

        for name, param in self.encoder.named_parameters():
            if name.startswith("encoder.layer."):
                layer_num = int(name.split(".")[2])
                if layer_num < freeze_layers:
                    param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_rep)
        return logits

# === 加载模型 ===
def load_model(path='model.pt', map_location='cpu'):
    model = DefectPredictionModel()
    state_dict = torch.load(path, map_location=map_location)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model

# === 推理函数 ===
def run_inference(model, tokenizer, code_str, device='cpu'):
    model.to(device)
    inputs = tokenizer(code_str, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted = torch.argmax(logits, dim=1).item()
    return predicted

# === 主函数 ===
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法：python predict.py \"<你的代码字符串>\"")
        sys.exit(1)

    code_str = sys.argv[1]
    model_path = 'model.pt'

    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = load_model(model_path, map_location='cpu')

    prediction = run_inference(model, tokenizer, code_str)
    print(prediction)  # 输出 0 或 1
