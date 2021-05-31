import torch
import numpy as np


x=torch.Tensor([[1,2,3],[1,2,3]])
print(x)
# x = x.unsqueeze(1)
print(torch.cat([x ** i for i in range(1, 3)],-1))
