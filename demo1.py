# import argparse
# from pypcd import pypcd
# import torch
# import numpy as np
# data_file = "./pointscloud/tulun/tulun/cloud_model.pcd"
# cloud = pypcd.PointCloud.from_path("./pointscloud/tulun/tulun/cloud_model.pcd")
# #points = np.load(data_file)
import open3d as o3d
import numpy as np
import math
import torch


def rotation_matrix_from_euler_angles(angles):
    """生成从欧拉角 (x, y, z) 的旋转矩阵"""
    rx, ry, rz = angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    return R
angles = (0.1, 0.2, 0.3)

# 创建旋转矩阵
rotation_matrix =rotation_matrix_from_euler_angles(angles)
print(rotation_matrix)
print(rotation_matrix.T)

pcd_model = o3d.io.read_point_cloud('./pointscloud/tulun/tulun/cloud_model.pcd')
pcd = o3d.io.read_point_cloud('./pointscloud/tulun/tulun/cloudnorm_61.pcd')
points = np.asarray(pcd.points)
points_model = np.asarray(pcd_model.points)
#points = np.dot(points, rotation_matrix.T)
points = points*50
points_model = points_model*50
np.save('./pointscloud/tulun/tulun/cloudnorm_61.npy', points)
np.save('./pointscloud/tulun/tulun/cloud_model.npy', points_model)
"""
import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()

if cuda_available:
    # 获取GPU设备数量
    num_gpu = torch.cuda.device_count()

    # 获取当前使用的GPU索引
    current_gpu_index = torch.cuda.current_device()

    # 获取当前GPU的名称
    current_gpu_name = torch.cuda.get_device_name(current_gpu_index)

    # 获取GPU显存的总量和已使用量
    total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
    used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
    free_memory = total_memory - used_memory  # 剩余显存(GB)

    print(f"CUDA可用，共有 {num_gpu} 个GPU设备可用。")
    print(f"当前使用的GPU设备索引：{current_gpu_index}")
    print(f"当前使用的GPU设备名称：{current_gpu_name}")
    print(f"GPU显存总量：{total_memory:.2f} GB")
    print(f"已使用的GPU显存：{used_memory:.2f} GB")
    print(f"剩余GPU显存：{free_memory:.2f} GB")
else:
    print("CUDA不可用。")

# 检查PyTorch版本
print(f"PyTorch版本：{torch.__version__}")

import torch
print(f"CUDA版本：{torch.version.cuda}")
temp = torch.tensor(2., dtype=torch.float16, device='cuda')"""


