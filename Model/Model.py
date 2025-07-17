import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer

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