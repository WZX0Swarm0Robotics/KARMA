import json
import numpy as np
# import matplotlib.pyplot as plt
from ai2thor.controller import Controller

def first_map(initial_event):
    # 遍历场景中的每个物体，获取它们的位置、类型和objectId
    objects_locations = []
    for obj in initial_event.metadata['objects']:
        obj_info = {
            'objectType': obj['objectType'],
            'position': obj['position'],
            'objectId': obj['objectId']  # 确保这里保存objectId
        }
        objects_locations.append(obj_info)

    # 将物体位置信息保存到 JSON 文件中
    with open('/home/user/wzx/karma/memory/objects_locations1.json', 'w') as f:
        json.dump(objects_locations, f, indent=4)

    print("物体位置信息已保存到 'objects_locations1.json' 文件中")
import json

def first_map_for_next_time(initial_event):
    # 遍历场景中的每个物体，获取它们的位置、类型、objectId 和 axisAlignedBoundingBox
    objects_locations = []
    for obj in initial_event.metadata['objects']:
        obj_info = {
            'objectType': obj['objectType'],
            'position': obj['position'],
            'objectId': obj['objectId'],  # 确保这里保存objectId
            'axisAlignedBoundingBox': obj['axisAlignedBoundingBox']  # 保存物体的边界框信息
        }
        objects_locations.append(obj_info)

    # 将物体位置信息保存到 JSON 文件中
    with open('/home/user/wzx/karma/memory/objects_locations.json', 'w') as f:
        json.dump(objects_locations, f, indent=4)

    print("物体位置信息及其边界框已保存到 'objects_locations1json' 文件中")

# 这个函数需要在 AI2-THOR 环境初始化和事件处理之后调用。

def second_map(event):
    # 遍历场景中的每个物体，获取它们的位置、类型和objectId
    objects_locations = []
    for obj in event.metadata['objects']:
        obj_info = {
            'objectType': obj['objectType'],
            'position': obj['position'],
            'objectId': obj['objectId']  # 确保这里保存objectId
        }
        objects_locations.append(obj_info)

    # 将物体位置信息保存到 JSON 文件中
    with open('/home/user/wzx/karma/memory/objects_locations2.json', 'w') as f:
        json.dump(objects_locations, f, indent=4)

    print("物体位置信息已保存到 'objects_locations2.json' 文件中")
# def plot_grid_map(reachable_positions, path=None):
#     # 创建栅格地图
#     grid_size = 0.25  # 栅格大小
#     x_coords = [pos['x'] for pos in reachable_positions]
#     z_coords = [pos['z'] for pos in reachable_positions]
#     x_min, x_max = min(x_coords), max(x_coords)
#     z_min, z_max = min(z_coords), max(z_coords)

#     # 根据最小和最大坐标确定栅格大小
#     x_range = np.arange(x_min, x_max + grid_size, grid_size)
#     z_range = np.arange(z_min, z_max + grid_size, grid_size)
#     grid = np.zeros((len(z_range), len(x_range)))

#     # 标记可达位置
#     for pos in reachable_positions:
#         x_idx = np.searchsorted(x_range, pos['x'])
#         z_idx = np.searchsorted(z_range, pos['z'])
#         grid[z_idx, x_idx] = 1  # 可到达的点标为1

#     # 绘制路径（如果提供）
#     if path:
#         for step in path:
#             x_idx = np.searchsorted(x_range, step['x'])
#             z_idx = np.searchsorted(z_range, step['z'])
#             grid[z_idx, x_idx] = 0.5  # 路径点标为0.5

#     # 画图
#     plt.figure(figsize=(10, 10))
#     plt.imshow(grid, origin='lower', cmap='gray', extent=(x_min, x_max, z_min, z_max))
#     plt.colorbar(label='Reachability')
#     plt.title('Grid Map with Reachable Positions and Path')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Z Coordinate')
#     plt.grid(True)
#     plt.show()