# import torch
# import torch.nn as nn
# from s5 import S5Block

# class S5ForecastModel(nn.Module):
#     def __init__(self, input_dim, d_model, n_layers, output_dim):
#         super().__init__()
#         self.input_proj = nn.Linear(input_dim, d_model)
#         self.s5_layers = nn.Sequential(
#             *[S5Block(d_model, d_model, prenorm=False) for _ in range(n_layers)]
#         )
#         self.output_proj = nn.Linear(d_model, output_dim)

#     def forward(self, x):  # x: [B, T, input_dim]
#         x = self.input_proj(x)           # [B, T, d_model]
#         x = self.s5_layers(x)            # [B, T, d_model]
#         out = self.output_proj(x)        # [B, T, output_dim]
#         return out


# from torch.utils.data import Dataset, DataLoader

# class ToySequenceDataset(Dataset):
#     def __init__(self, length=1000, seq_len=64):
#         x = torch.linspace(0, 100, steps=length)
#         self.data = torch.sin(x).unsqueeze(-1)  # [1000, 1]
#         self.seq_len = seq_len

#     def __len__(self):
#         return len(self.data) - self.seq_len

#     def __getitem__(self, idx):
#         x = self.data[idx : idx + self.seq_len]        # input
#         y = self.data[idx + 1 : idx + self.seq_len + 1]  # next-step target
#         return x, y

# train_ds = ToySequenceDataset()
# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


# model = S5ForecastModel(input_dim=1, d_model=64, n_layers=4, output_dim=1).cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()

# for epoch in range(20):
#     model.train()
#     total_loss = 0
#     for x, y in train_loader:
#         x, y = x.cuda(), y.cuda()
#         pred = model(x)
#         loss = loss_fn(pred, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
    
#     print(f"[Epoch {epoch}] Loss: {total_loss / len(train_loader):.6f}")


# import matplotlib.pyplot as plt

# model.eval()
# with torch.no_grad():
#     x, y = train_ds[100]
#     x = x.unsqueeze(0).cuda()  # [1, T, 1]
#     pred = model(x).squeeze().cpu()
#     gt = y.squeeze()

#     plt.plot(gt.numpy(), label="GT")
#     plt.plot(pred.numpy(), label="Pred")
#     plt.legend()
#     plt.title("S5 Forecast")
#     plt.show()

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
from s5 import S5Block
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seq_len", type=int, default=96, help="Length of input sequence")
parser.add_argument("--pred_len", type=int, default=24, help="Prediction length")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
args = parser.parse_args()

class S5ForecastModel(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.s5_layers = nn.Sequential(
            *[S5Block(d_model, d_model, bidir=False) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):  # x: [B, T, input_dim]
        x = self.input_proj(x)           # [B, T, d_model]
        x = self.s5_layers(x)            # [B, T, d_model]
        out = self.output_proj(x)        # [B, T, output_dim]
        return out


class ETTh1Dataset(Dataset):
    def __init__(self, path, seq_len=96, pred_len=24, feature='OT', train=True):
        df = pd.read_csv(path)
        data = df[feature].values.astype(np.float32)
        # 정규화
        mean = data.mean()
        std = data.std()
        data = (data - mean) / std
        self.data = torch.tensor(data).unsqueeze(-1)  # [T, 1]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train = train

        # 나누기
        n = len(self.data)
        if train:
            self.data = self.data[:int(n*0.7)]  # 70% 학습
        else:
            self.data = self.data[int(n*0.7):]  # 나머지 검증/테스트

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y

train_ds = ETTh1Dataset("ETDataset/ETT-small/ETTh1.csv", seq_len=96, pred_len=24, train=True)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

model = S5ForecastModel(input_dim=1, d_model=128, n_layers=6, output_dim=1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # 학습률이 너무 크면 overshooting처럼 예측이 튐 → 줄임.
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        pred = pred[:, -y.shape[1]:, :] # 뒤에서부터 pred_len만 추출
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch}] Loss: {total_loss / len(train_loader):.6f}")

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    x, y = train_ds[100]
    x = x.unsqueeze(0).cuda()  # [1, T, 1]
    pred = model(x).squeeze().cpu()[-y.shape[0]:]
    gt = y.squeeze()

    plt.plot(gt.numpy(), label="Ground Truth")
    plt.plot(pred.numpy(), label="Prediction")
    plt.title("S5 Forecast (ETTh1)")
    plt.legend()
    plt.show()
