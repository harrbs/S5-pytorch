# from s5 import S5, S5Block
# import torch

# # Raw S5 operator
# x = torch.rand([2, 256, 32])
# model = S5(32, 32)
# model(x) # [2, 256, 32]

# # S5-former block (S5+FFN-GLU w/ layernorm, dropout & residual)
# model = S5Block(32, 32, False)
# model(x) # [2, 256, 32]

import torch
from s5 import S5, S5Block

x = torch.rand([2, 256, 32])

# Raw S5 operator
model = S5(32, 32)
y = model(x)
print("S5 output shape:", y.shape)

# S5Block (S5 + residual + FFN)
model = S5Block(32, 32, False)
y = model(x)
print("S5Block output shape:", y.shape)
