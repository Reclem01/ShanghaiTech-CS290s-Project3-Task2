import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from models import StockPriceCNNLSTM
from utils import *
from tqdm import tqdm
import os
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 准备数据
train_folder = r'D:\yuanman\PycharmProject\stockprediction\data\train'
cache_path = r'D:\yuanman\PycharmProject\stockprediction\data\cached_data.pt'

# 超参数设置
input_size = 50  # 输入特征维度
hidden_size = 200
output_size = 1
num_layers = 5
num_epochs = 10
learning_rate = 1e-3
seq_length = 30
batch_size = 100
dropout = 0.3

# 初始化模型、损失函数和优化器
model = StockPriceCNNLSTM(input_size, hidden_size, output_size, num_layers, dropout, seq_length)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scaler = GradScaler()

# 加载数据集
if not os.path.exists(cache_path):
    print("Now creating dataset...")
    all_data, all_labels, file_paths = load_all_data(train_folder)
    save_preprocessed_data(all_data, all_labels, cache_path)
else:
    print("Now loading dataset...")
    all_data, all_labels = load_preprocessed_data(cache_path)
    print("Dataset loading complete.")
    # print(all_data.shape)
    # print(all_labels.shape)

# 创建序列
data, labels = create_sequences(all_data, all_labels, seq_length)
data = data.permute(0, 2, 1)
# print(data.shape)   # [54600 * seq_length, 50, seq_length]
# print(labels.shape) # [54600 * seq_length, 1]
# exit()

# 构建数据集和数据加载器
dataset = TensorDataset(data, labels)

# 按8:2划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
print("Now model is training...")
best_val_loss = float('inf')

for epoch in range(num_epochs):
    epoch_loss = 0  # 用于计算每个epoch的总损失
    model.train()
    for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
    scheduler.step()

    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

    # 保存模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'checkpoint/test3/best_model_epoch_{epoch+1}.pth')
        print(f'Model saved at epoch {epoch+1}')

# 清理 CUDA 资源
torch.cuda.empty_cache()

# 确保所有进程正确退出
sys.exit(0)
