import numpy as np

# 创建一个NumPy数组
arr = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

# 将数组保存为npy文件
np.save('./pointscloud/tulun/tulun/gt1.npy', arr)
