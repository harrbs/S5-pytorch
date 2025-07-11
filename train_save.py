import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
from s5 import S5Block
import argparse
import matplotlib.pyplot as plt

import random
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 명령줄 인자
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--csv_path', type=str, default="ETDataset/ETT-small/ETTh1.csv")
parser.add_argument('--save_plot', type=str, default=None, help="시각화 결과를 저장할 파일 경로 (ex: './forecast.png')")
parser.add_argument('--resume_from', type=str, default=None, help="재시작할 체크포인트 경로 (ex: './checkpoints/s5_epoch200.pt')") # 체크포인트에서 이어서 학습 가능하게
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



# 학습 / 테스트 데이터 구성

# 1. train에서 mean, std 구함
train_ds = ETTh1Dataset(args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len, train=True)
mean, std = train_ds.mean, train_ds.std

# 2. test에 train 기준 mean, std 넘겨줌
test_ds = ETTh1Dataset(args.csv_path, seq_len=args.seq_len, pred_len=args.pred_len, train=False, mean=mean, std=std)


train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

# 모델 초기화
model = S5ForecastModel(input_dim=1, d_model=args.d_model, n_layers=args.n_layers, output_dim=1).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

total_epochs = args.epochs
warmup_epochs = int(total_epochs * 0.2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

def adjust_learning_rate(epoch):
    if epoch < warmup_epochs:
        lr = args.lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()

loss_fn = torch.nn.MSELoss()
mae_fn = torch.nn.L1Loss()

# 학습 루프
start_epoch = 0
if args.resume_from:
    checkpoint = torch.load(args.resume_from)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch']
    print(f"[Resume] 모델을 {start_epoch} epoch부터 재시작합니다.")

for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(epoch)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_mse += mse.item()
        total_mae += mae.item()

    print(f"[Epoch {epoch}] Train MSE: {total_mse / len(train_loader):.6f} | MAE: {total_mae / len(train_loader):.6f}")

    if (epoch + 1) % 100 == 0: # 저장 체크포인트 (예: 100 epoch마다 저장)
        # 체크포인트 저장 시 optimizer와 epoch도 함께 저장
        checkpoint = {
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }

        torch.save(checkpoint, f"./checkpoints/s5_8_epoch{epoch+1}.pt")
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