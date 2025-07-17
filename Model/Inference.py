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
from timedate import timedate

# 1. 配置设备
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except:
    npu_available = False

device = torch.device("npu:6" if npu_available else "cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备{device}")