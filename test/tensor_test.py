import torch
import torch.nn.functional as F
from torch.distributions import Categorical

k = 2

padding_mask = torch.randint(0, 100, (64, 50, 6))

padding_mask[:,0] = True
print(padding_mask[:,0])

print(padding_mask[:,0:5].shape)   # (64, 5, 6)
# print(torch.where(~padding_mask[:,0])[0].shape)   # (3456)

# print(torch.zeros_like(padding_mask[:, 0:1+2*k]).shape)  # torch.Size([64, 5, 6])


