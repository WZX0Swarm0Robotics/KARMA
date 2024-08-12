# # # 使用示例
# # compare_json_files('file1.json', 'file2.json', 'differences.json')
import json
import math
from collections import deque
def read_json_file(file_path):
    """读取 JSON 文件并返回数据."""
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_distance(position1, position2):
    """计算两个位置之间的欧氏距离."""
    return math.sqrt((position1['x'] - position2['x']) ** 2 + 
                     (position1['y'] - position2['y']) ** 2 +
                     (position1['z'] - position2['z']) ** 2)

def compare_objects_location(objects_locations1, objects_locations2, output_file, threshold=0.3, max_objects=100):
    """比较两个 JSON 文件中物体的位置是否有显著变化，并输出不一致的物体信息到新文件."""
    data1 = read_json_file(objects_locations1)
    data2 = read_json_file(objects_locations2)

    objects_data1 = {item['objectId']: item for item in data1}
    objects_data2 = {item['objectId']: item for item in data2}

    differences = []

    # 遍历所有物体，比较位置信息
    for object_id in objects_data1:
        if object_id in objects_data2:
            position1 = objects_data1[object_id]['position']
            position2 = objects_data2[object_id]['position']
            if calculate_distance(position1, position2) > threshold:
                differences.append(objects_data2[object_id])
    #检查是否需要维护队列长度
    try:
        with open(output_file, 'r') as file:
            output_data = json.load(file)
    except FileNotFoundError:
        output_data = []        
    merged_data = output_data + differences

    unique_objects = {}

    for obj in merged_data:
        obj_id = obj['objectId']
        unique_objects[obj_id] = obj 
    
    if len(unique_objects) > max_objects:
        sorted_objects = sorted(unique_objects.items(), key=lambda x: x[1]['timestamp'], reverse=True)
        unique_objects = dict(sorted_objects[:max_objects])
    
    with open(output_file, 'w') as file:
        json.dump(list(unique_objects.values()), file, indent=4)