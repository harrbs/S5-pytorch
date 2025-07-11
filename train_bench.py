import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
from s5 import S5Block
import argparse
import matplotlib.pyplot as plt

# 명령줄 인자
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--csv_path', type=str, default="ETDataset/ETT-small/ETTh1.csv")
parser.add_argument('--save_plot', type=str, default=None, help="시각화 결과를 저장할 파일 경로 (ex: './forecast.png')")
args = parser.parse_args()

# 모델 정의
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

# 데이터셋 정의
# class ETTh1Dataset(Dataset):
#     def __init__(self, path, seq_len=96, pred_len=24, feature='OT', train=True):
#         df = pd.read_csv(path)
#         data = df[feature].values.astype(np.float32)
#         # 정규화
#         mean = data.mean()
#         std = data.std()
#         data = (data - mean) / std
#         self.data = torch.tensor(data).unsqueeze(-1)  # [T, 1]
#         self.seq_len = seq_len
#         self.pred_len = pred_len

#         n = len(self.data)
#         if train:
#             self.data = self.data[:int(n * 0.7)]
#         else:
#             self.data = self.data[int(n * 0.7):]

#     def __len__(self):
#         return len(self.data) - self.seq_len - self.pred_len

#     def __getitem__(self, idx):
#         x = self.data[idx : idx + self.seq_len]
#         y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
#         return x, y
class ETTh1Dataset(Dataset):
    def __init__(self, path, seq_len=96, pred_len=24, feature='OT', train=True, mean=None, std=None):
        df = pd.read_csv(path)
        data = df[feature].values.astype(np.float32)

        n = len(data)
        if train:
            data = data[:int(n * 0.7)]
        else:
            data = data[int(n * 0.7):]

        # 정규화
        if mean is None or std is None:
            self.mean = data.mean()
            self.std = data.std()
        else:
            self.mean = mean
            self.std = std

        data = (data - self.mean) / self.std
        self.data = torch.tensor(data).unsqueeze(-1)

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y



# # 학습 / 테스트 데이터 구성
# train_ds = ETTh1Dataset(args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len, train=True)
# test_ds = ETTh1Dataset(args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len, train=False)

# 1. train에서 mean, std 구함
train_ds = ETTh1Dataset(args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len, train=True)
mean, std = train_ds.mean, train_ds.std

# 2. test에 train 기준 mean, std 넘겨줌
test_ds = ETTh1Dataset(args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len, train=False, mean=mean, std=std)


train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# 모델 초기화
model = S5ForecastModel(input_dim=1, d_model=args.d_model, n_layers=args.n_layers, output_dim=1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.MSELoss()
mae_fn = torch.nn.L1Loss()

# 학습 루프
for epoch in range(args.epochs):
    model.train()
    total_mse = 0
    total_mae = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        pred = pred[:, -y.shape[1]:, :]
        mse = loss_fn(pred, y)
        mae = mae_fn(pred, y)

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        total_mse += mse.item()
        total_mae += mae.item()

    print(f"[Epoch {epoch}] Train MSE: {total_mse / len(train_loader):.6f} | MAE: {total_mae / len(train_loader):.6f}")

    if (epoch + 1) % 100 == 0: # 저장 체크포인트 (예: 100 epoch마다 저장)
        torch.save(model.state_dict(), f"./checkpoints/s5_epoch{epoch+1}.pt")
        print(f"[Checkpoint] Saved model at epoch {epoch+1}")

# 테스트 루프
model.eval()
test_mse, test_mae = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        pred = pred[:, -y.shape[1]:, :]
        test_mse += loss_fn(pred, y).item()
        test_mae += mae_fn(pred, y).item()

print(f"[Test] MSE: {test_mse / len(test_loader):.6f} | MAE: {test_mae / len(test_loader):.6f}")


# 시각화
with torch.no_grad():
    x, y = test_ds[100]
    x = x.unsqueeze(0).cuda()
    pred = model(x).squeeze().cpu()[-y.shape[0]:]
    gt = y.squeeze()

    plt.figure(figsize=(10, 4))
    plt.plot(gt.numpy(), label="Ground Truth")
    plt.plot(pred.numpy(), label="Prediction")
    plt.title("S5 Forecast (ETTh1 - Test Sample)")
    plt.legend()

    if args.save_plot:
        plt.savefig(args.save_plot, bbox_inches='tight')
        print(f"[Saved] Plot saved to: {args.save_plot}")
    else:
        plt.show()