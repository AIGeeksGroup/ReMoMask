'''
python helper/read_npy.py
'''

import numpy as np

# 假设你有一个名为 data.npy 的文件
data = np.load('/data/AI4E/lzd/AAAI/ReMoMaskV2/ReMoMask/database_small/encoded_motions.npy')


# print(data)
print(type(data))   # 查看数据类型  
print(data.shape)   # 查看数据维度  # (32, 1, 512)


