import os
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from utils import *
from models import StockPriceLSTM
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_folder = r'D:\yuanman\PycharmProject\stockprediction\data\test'
test_cache_path = r'D:\yuanman\PycharmProject\stockprediction\data\test_cached_data.pt'

seq_length = 30
batch_size = 100
input_size = 50  # 输入特征维度
hidden_size = 200
output_size = 1
num_layers = 5
dropout = 0.3

# 加载测试数据集
if not os.path.exists(test_cache_path):
    print("Now creating test dataset...")
    all_test_data, all_test_labels, test_file_paths = load_all_data(test_folder)
    if all_test_data is not None:
        print("Test data loaded successfully.")
        save_preprocessed_data(all_test_data, all_test_labels, test_cache_path)
    else:
        print("No valid test data found.")
else:
    print("Now loading test dataset...")
    all_test_data, all_test_labels = load_preprocessed_data(test_cache_path)
    print("Test dataset loading complete.")
    print(all_test_data.shape)
    print(all_test_labels.shape)

# 创建测试序列
test_data, test_labels = create_sequences(all_test_data, all_test_labels, seq_length)

# 构建测试数据集和数据加载器
test_dataset = TensorDataset(test_data, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = StockPriceLSTM(input_size, hidden_size, output_size, num_layers, dropout=dropout)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
# 加载最佳模型
model.load_state_dict(torch.load('checkpoint/test3/best_model_epoch_1.pth'))
model.to(device)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in tqdm(test_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.cpu())
        all_labels.append(targets.cpu())

all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# 计算相关系数
correlation = corr_fn(all_preds, all_labels)
print(f'Test Correlation: {correlation:.4f}')