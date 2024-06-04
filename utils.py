import pandas as pd
import torch
import os


# def load_data(fname: str):
#     # load data from one csv file
#     df = pd.read_csv(fname, index_col=0)    # 读取CSV文件，第一列作为索引
#     label = torch.tensor(df.label, dtype=torch.float32).unsqueeze(dim=0)    # 提取标签并转换为PyTorch张量，增加一个维度
#     del df["label"] # 删除标签列，剩下的数据是特征
#     data = torch.tensor(df.values, dtype=torch.float32).unsqueeze(dim=0)    # 将特征转换为PyTorch张量，增加一个维度
#     # print(data.shape)   # [1, 241, 50]
#     # print("Label:", label.shape)    # [1, 241]
#     return data, label
#
#
# def load_all_data(base_path: str):
#     all_data = []
#     all_labels = []
#     file_paths = []
#
#     for subdir, _, files in os.walk(base_path):
#         for file in files:
#             if file.endswith('.csv'):
#                 file_path = os.path.join(subdir, file)
#                 data, label = load_data(file_path)
#                 all_data.append(data)
#                 all_labels.append(label)
#                 file_paths.append(file_path)
#             # print(all_data)
#     #     print("Now the subdir is:", subdir)
#     # exit()
#     # 拼接所有数据和标签
#     all_data = torch.cat(all_data, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
#     # print(all_data.shape)     # [54600, 241, 50]
#
#     return all_data, all_labels, file_paths

def load_data(fname: str):
    try:
        df = pd.read_csv(fname, index_col=0)  # Read CSV file with the first column as index
        if 'label' not in df.columns:
            raise ValueError(f"File {fname} does not contain 'label' column")
        label = torch.tensor(df.label, dtype=torch.float32).unsqueeze(dim=0)  # Extract label and convert to PyTorch tensor
        df.drop(columns=['label'], inplace=True)  # Drop the label column, remaining data is features
        data = torch.tensor(df.values, dtype=torch.float32).unsqueeze(dim=0)  # Convert features to PyTorch tensor
        return data, label
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None, None


def load_all_data(base_path: str):
    all_data = []
    all_labels = []
    file_paths = []

    for subdir, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                data, label = load_data(file_path)
                if data is not None and label is not None:  # Only append if data loading was successful
                    all_data.append(data)
                    all_labels.append(label)
                    file_paths.append(file_path)

    if all_data:
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return all_data, all_labels, file_paths
    else:
        return None, None, []


def save_preprocessed_data(all_data, all_labels, save_path):
    torch.save((all_data, all_labels), save_path)


def load_preprocessed_data(save_path):
    return torch.load(save_path)


def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    num_samples, num_time_steps, _ = data.size()

    for i in range(num_time_steps - seq_length):
        x = data[:, i:i + seq_length, :]  # 输入序列
        y = labels[:, i + seq_length]  # 目标值
        xs.append(x)
        ys.append(y)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0).unsqueeze(1)


def corr_fn(pred: torch.Tensor, label: torch.Tensor):
    pred = (pred - pred.mean(dim=1, keepdim=True)) / pred.std(
        dim=1, keepdim=True, unbiased=False
    )
    label = label / label.std(dim=1, keepdim=True, unbiased=False)
    corr = (pred * label).sum(axis=1) / pred.shape[1]
    return corr.mean()
