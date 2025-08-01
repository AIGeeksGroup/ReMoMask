import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x)
print(x.stride())  # 输出: (3, 1)

print("*"*10)

y = x.transpose(0, 1)
print(y)
print(y.stride())  # 输出: (1, 3)

'''
内存布局： [1,2,3,4,5,6]

tensor([[1, 2, 3],
        [4, 5, 6]])
(3, 1)
**********
tensor([[1, 4],
        [2, 5],
        [3, 6]])
(1, 3)
'''


z = y.contiguous()
print(z)
print(z.stride())  # 输出: (1, 3)
'''
内存布局: [1, 4, 2, 5, 3, 6]
tensor([[1, 4],
        [2, 5],
        [3, 6]])
(2, 1)
'''



