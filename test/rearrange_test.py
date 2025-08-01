import torch
from einops import rearrange

x = torch.arange(6).reshape(2,3)

print(x)
# tensor([[0, 1, 2],
#         [3, 4, 5]])

y = rearrange(x, 'b c -> c b')

print(y)
# tensor([[0, 3],
#         [1, 4],
#         [2, 5]])

z = x.view(3,2)

print(z)
# tensor([[0, 1],
#         [2, 3],
#         [4, 5]])


d = rearrange(x, 'b c -> (b c) 1')

print(d)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])

e = rearrange(x, 'b c -> (c b) 1')

print(e)
# tensor([[0],
#         [3],
#         [1],
#         [4],
#         [2],
#         [5]])