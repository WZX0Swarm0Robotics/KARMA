import ai2thor.controller
import numpy as np
import json

def get_divided_positions(controller, grid_size=0.25, divisions=3):
    event = controller.step('GetReachablePositions')
    reachable_positions = event.metadata['actionReturn']

    # Find the bounds of the reachable positions
    min_x = min(pos['x'] for pos in reachable_positions)
    max_x = max(pos['x'] for pos in reachable_positions)
    min_z = min(pos['z'] for pos in reachable_positions)
    max_z = max(pos['z'] for pos in reachable_positions)

    x_interval = (max_x - min_x) / divisions
    z_interval = (max_z - min_z) / divisions

    centers = []
    for i in range(divisions):
        for j in range(divisions):
            center_x = min_x + (i + 0.5) * x_interval
            center_z = min_z + (j + 0.5) * z_interval
            centers.append((center_x, 0, center_z))

    return centers

def get_static_objects_in_regions(controller, centers, grid_size=0.25):
    regions = {center: [] for center in centers}

    for obj in controller.last_event.metadata['objects']:
        if not obj['pickupable']:
            obj_pos = obj['position']
            min_distance = float('inf')
            closest_center = None
            for center in centers:
                distance = np.linalg.norm([obj_pos['x'] - center[0], obj_pos['z'] - center[2]])
                if distance < min_distance:
                    min_distance = distance
                    closest_center = center
            regions[closest_center].append(obj)

    return regions

def extract_regions_from_json(filename='regions.json'):
    with open(filename, 'r') as f:
        data = json.load(f)

    sentences = []
    for center, objects in data.items():
        object_types = [obj['objectType'] for obj in objects]
        sentence = f'center {center} has {{{", ".join(object_types)}}}'
        sentences.append(sentence)
    
    return sentences

# # Get divided positions
# centers = get_divided_positions(controller)

# # Get static objects in regions
# regions = get_static_objects_in_regions(controller, centers)

# # Output the results
# for center, objects in regions.items():
#     print(f'Center: {center}')
#     for obj in objects:
#         print(f'  Object: {obj["objectType"]} at {obj["position"]}')

# Example usage:
# filename = 'regions.json'
# sentences = extract_regions_from_json(filename)
# for sentence in sentences:
#     print(sentence)


#######prompt#######
