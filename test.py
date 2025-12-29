import torch
import torch.nn.functional as F
x = torch.randn(110)
y = torch.randn(100)
loss = F.smooth_l1_loss(x, y)
