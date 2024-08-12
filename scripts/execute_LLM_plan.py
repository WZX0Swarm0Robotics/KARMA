import math
import subprocess
import queue
# from queue import Queue
import re
import shutil
import time
import threading
import cv2
import numpy as np
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
from collections import deque
import random
import os
from glob import glob
from mapping import first_map
from mapping import first_map_for_next_time
from mapping import second_map
from memory_save import compare_objects_location
from memory_save import read_json_file
from longterm_save import get_divided_positions
from longterm_save import get_static_objects_in_regions
from longterm_save import extract_regions_from_json
import json
import importlib
import sys
import openai
import base64
import requests
# sys.path.append('/home/user/wzx/karma/logs')
task_queue = queue.Queue()

api_key = 'sk-proj-KkKPT0s3lzTeSoUx3NaIT3BlbkFJDf7klRn9f80jlHQexLvC'
directory_path = '/home/user/wzx/karma/memory/short_term'
task_description_file_path = '/home/user/wzx/karma/logs/task_description.json'
results_file_path = '/home/user/wzx/karma/memory/analysis_results.json'
def load_task_description():
    try:
        with open(task_description_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get("task_description", "")
    except FileNotFoundError:
        return ""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to analyze an image with a given task
def analyze_image(image_path, task):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "As an image analysis expert, your task is to infer the state of objects in the image through step-by-step reasoning."},
            {"role": "user", "content": f"1. Provide a detailed description of this image.\n2. From the given task [Task], extract the relevant content from the first step's image description that pertains to the mentioned objects.\n3. Based on the object descriptions extracted in the second step, match each object to one of the following states: heated, cooked, sliced, cleaned, dirty, filled, used up, off, on, opened, closed, none.\n4. Summarize the results from step three in the following format: object: state. Please only output the content of the last step summary."},
            {"role": "user", "content": f"[Task]: {task}"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 4096
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

# Function to analyze images in a directory with a given task

# def analyze_images_in_directory(directory, task):
#     # 尝试读取现有的JSON文件
#     results_file_path = os.path.join(directory, '/home/user/wzx/karma/memory/analysis_results.json')
#     if os.path.exists(results_file_path):
#         with open(results_file_path, 'r', encoding='utf-8') as file:
#             results = json.load(file)
#     else:
#         results = {}

#     # 对每个图像文件进行分析并追加新结果
#     for filename in os.listdir(directory):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
#             image_path = os.path.join(directory, filename)
#             result = analyze_image(image_path, task)
#             # 从返回结果中提取所需内容并添加到结果字典中
#             content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
#             # 将结果分开保存
#             objects_states = {}
#             for line in content.split('\n'):
#                 if ': ' in line:
#                     obj, state = line.split(': ')
#                     objects_states[obj] = state
#             results[filename] = objects_states

#     # 将合并后的结果保存到JSON文件
#     with open(results_file_path, 'w', encoding='utf-8') as file:
#         json.dump(results, file, ensure_ascii=False, indent=4)

#     return results

def analyze_specified_image(directory, filename, task):
    # 尝试读取现有的JSON文件
    if os.path.exists(results_file_path):
        with open(results_file_path, 'r', encoding='utf-8') as file:
            results = json.load(file)
    else:
        results = {}

    # 构建图像文件路径
    image_path = os.path.join(directory, filename)
    if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
        result = analyze_image(image_path, task)
        # 从返回结果中提取所需内容并添加到结果字典中
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        # 将结果分开保存
        objects_states = {}
        for line in content.split('\n'):
            if ': ' in line:
                obj, state = line.split(': ', 1)  # 修改为分割一次
                objects_states[obj] = state
        results[filename] = objects_states

    # 将合并后的结果保存到JSON文件
    with open(results_file_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    return results

def save_agent_view(image, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, filename), image)

def save_regions_to_json(regions, filename='/home/user/wzx/karma/memory/longterm_memory.json'):
    data = {}
    for center, objects in regions.items():
        center_key = f'({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})'
        data[center_key] = [{'objectType': obj['objectType'], 'position': obj['position']} for obj in objects]

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def generate_random_position_from_list(available_positions):
    rand_position = random.choice(available_positions)
    available_positions.remove(rand_position)
    return rand_position, available_positions

def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def generate_video(input_path, prefix, char_id=0, image_synthesis=['normal'], frame_rate=5, output_path=None):
    """ Generate a video of an episode """
    if output_path is None:
        output_path = input_path

    vid_folder = '{}/{}/{}/'.format(input_path, prefix, char_id)
    if not os.path.isdir(vid_folder):
        print("The input path: {} you specified does not exist.".format(input_path))
    else:
        for vid_mod in image_synthesis:
            command_set = ['ffmpeg', '-i',
                             '{}/Action_%04d_0_{}.png'.format(vid_folder, vid_mod), 
                             '-framerate', str(frame_rate),
                             '-pix_fmt', 'yuv420p',
                             '{}/video_{}.mp4'.format(output_path, vid_mod)]
            subprocess.call(command_set)
            print("Video generated at ", '{}/video_{}.mp4'.format(output_path, vid_mod))

robots = [{'name': 'robot1', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}]

floor_no = 1

no_robot = len(robots)
objects_locations1 = '/home/user/wzx/karma/memory/objects_locations.json'
# initialize n agents into the scene
c = Controller(
    agentMode="default",
    visibilityDistance=100,
    scene="FloorPlan1",

    # step sizes
    gridSize=0.25,
    snapToGrid=False,
    rotateStepDegrees=20,
    quality='Low',
    # image modalities
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    agentCount=no_robot,
    # camera properties
    width=1000,
    height=1000,
    fieldOfView=90
)
multi_agent_event = c.step(action="Done") 

# add a top view camera
event = c.step(action="GetMapViewCameraProperties")
event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

# get reachabel positions
reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

# randomize postions of the agents
for i in range (no_robot):
    init_pos = random.choice(reachable_positions_)
    c.step(dict(action="Teleport", position=init_pos, agentId=i))

#map for memory
memory_last_event=c.step(action="Done")    
first_map(memory_last_event)
first_map_for_next_time(memory_last_event)
# Get divided positions
centers = get_divided_positions(c)

# Get static objects in regions
regions = get_static_objects_in_regions(c, centers)
#保存long-term memory
save_regions_to_json(regions)

filename = '/home/user/wzx/karma/memory/longterm_memory.json'
sentences = extract_regions_from_json(filename)
for sentence in sentences:
    print(sentence)

action_queue = []

task_over = False

def exec_actions():
    # delete if current output already exist
    cur_path = os.path.dirname(__file__) + "/*/"
    for x in glob(cur_path, recursive = True):
        shutil.rmtree (x)
    
    # create new folders to save the images from the agents
    for i in range(no_robot):
        folder_name = "agent_" + str(i+1)
        folder_path = os.path.dirname(__file__) + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # create folder to store the top view images
    folder_name = "top_view"
    folder_path = os.path.dirname(__file__) + "/" + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    img_counter = 0
    #for short-term memory
    image_counter = 0 
    while not task_over:
        if len(action_queue) > 0:
            try:
                act = action_queue[0]
                if act['action'] == 'ObjectNavExpertAction':
                    multi_agent_event = c.step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                    next_action = multi_agent_event.metadata['actionReturn']

                    if next_action != None:
                        multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=True)
                
                elif act['action'] == 'MoveAhead':
                    multi_agent_event = c.step(action="MoveAhead", agentId=act['agent_id'])
                    
                elif act['action'] == 'MoveBack':
                    multi_agent_event = c.step(action="MoveBack", agentId=act['agent_id'])
                        
                elif act['action'] == 'RotateLeft':
                    multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])
                    
                elif act['action'] == 'RotateRight':
                    multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])
                    
                elif act['action'] == 'PickupObject':
                    multi_agent_event = c.step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True) 

                elif act['action'] == 'PutObject':
                    multi_agent_event = c.step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    second_map(multi_agent_event)
                    compare_objects_location('/home/user/wzx/karma/memory/objects_locations1.json', '/home/user/wzx/karma/memory/objects_locations2.json', '/home/user/wzx/karma/memory/memory3.json')
                    first_map(multi_agent_event)
                    #调整视角，用于拍摄short-term memory的图片
                    c.step(action='LookDown',degrees=20)
                    frame = multi_agent_event.frame
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    save_path = '/home/user/wzx/karma/memory/short_term'
                    filename = f"short_memory_{image_counter}.png"
                    save_agent_view(frame_bgr, save_path, filename)
                    image_counter += 1

                    task_description = load_task_description()
                    print(f"Executing task: {task_description}")
                    results = analyze_specified_image(directory_path, filename, task_description)
                    print("Analysis complete. Results saved to 'analysis_results.json'.")

                    # for filename, result in results.items():
                    #     print(f"{filename}: {result}")

                elif act['action'] == 'ToggleObjectOn':
                    multi_agent_event = c.step(action="ToggleObjectOn", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                
                elif act['action'] == 'ToggleObjectOff':
                    multi_agent_event = c.step(action="ToggleObjectOff", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)

                elif act['action'] == 'ThrowObject':
                    multi_agent_event = c.step(action="ThrowObject", moveMagnitude=7, agentId=act['agent_id'], forceAction=True) 

                elif act['action'] == 'PlaceObjectAtPoint':
                    multi_agent_event1 = c.step(action="PlaceObjectAtPoint", objectId="Apple|-00.47|+01.15|+00.48", position={"x": -1.35, "y": 1.0, "z": -2.3})
                    if multi_agent_event1.metadata['lastActionSuccess']:
                        print("Action was successful!")
                    else:
                        print("Action failed:", multi_agent_event1['errorMessage'])
                    second_map(multi_agent_event1)
                    compare_objects_location('/home/user/wzx/karma/memory/objects_locations1.json', '/home/user/wzx/karma/memory/objects_locations2.json', '/home/user/wzx/karma/memory/memory3.json')
                elif act['action'] == 'Done':
                    multi_agent_event = c.step(action="Done")
              
            except Exception as e:
                print (e)
              
            for i,e in enumerate(multi_agent_event.events):
                cv2.imshow('agent%s' % i, e.cv2img)
                f_name = os.path.dirname(__file__) + "/agent_" + str(i+1) + "/img_" + str(img_counter).zfill(5) + ".png"
                cv2.imwrite(f_name, e.cv2img)
            top_view_rgb = cv2.cvtColor(c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
            cv2.imshow('Top View', top_view_rgb)
            f_name = os.path.dirname(__file__) + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
            cv2.imwrite(f_name, e.cv2img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            img_counter += 1    
            action_queue.pop(0)
            
actions_thread = threading.Thread(target=exec_actions)
actions_thread.start()

# def GoToObject(robots, dest_obj):
#     print ("Going to ", dest_obj)
#     # check if robots is a list
#     # dest_obj_center = None
#     if not isinstance(robots, list):
#         # convert robot to a list
#         robots = [robots]
#     no_agents = len (robots)
#     # robots distance to the goal 
#     dist_goals = [10.0] * len(robots)
#     prev_dist_goals = [10.0] * len(robots)
#     count_since_update = [0] * len(robots)
#     clost_node_location = [0] * len(robots)
    
#     # list of objects in the scene and their centers
#     objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
#     objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])

#     # look for the location and id of the destination object
#     for idx, obj in enumerate(objs):
#         match = re.match(dest_obj, obj)
#         if match is not None:
#             dest_obj_id = obj
#             dest_obj_center = objs_center[idx]
#             break # find the first instance
        
#     dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
    
#     # closest reachable position for each robot
#     # all robots cannot reach the same spot 
#     # differt close points needs to be found for each robot
#     crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
#     goal_thresh = 0.3
#     # at least one robot is far away from the goal
    
#     while all(d > goal_thresh for d in dist_goals):
#         for ia, robot in enumerate(robots):
#             robot_name = robot['name']
#             agent_id = int(robot_name[-1]) - 1
            
#             # get the pose of robot        
#             metadata = c.last_event.events[agent_id].metadata
#             location = {
#                 "x": metadata["agent"]["position"]["x"],
#                 "y": metadata["agent"]["position"]["y"],
#                 "z": metadata["agent"]["position"]["z"],
#                 "rotation": metadata["agent"]["rotation"]["y"],
#                 "horizon": metadata["agent"]["cameraHorizon"]}
            
#             prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal
#             dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
#             dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
#             # print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
#             if dist_del < 0.2:
#                 # robot did not move 
#                 count_since_update[ia] += 1
#             else:
#                 # robot moving 
#                 count_since_update[ia] = 0
                
#             if count_since_update[ia] < 15:
#                 action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
#             else:    
#                 #updating goal
#                 clost_node_location[ia] += 1
#                 count_since_update[ia] = 0
#                 crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
#             time.sleep(0.5)

#     # align the robot once goal is reached
#     # compute angle between robot heading and object
#     metadata = c.last_event.events[agent_id].metadata
#     robot_location = {
#         "x": metadata["agent"]["position"]["x"],
#         "y": metadata["agent"]["position"]["y"],
#         "z": metadata["agent"]["position"]["z"],
#         "rotation": metadata["agent"]["rotation"]["y"],
#         "horizon": metadata["agent"]["cameraHorizon"]}
    
#     robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
#     y_axis = [0, 1]
#     unit_y = y_axis / np.linalg.norm(y_axis)
#     unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
    
#     angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
#     angle = 360*angle/(2*np.pi)
#     angle = (angle + 360) % 360
#     rot_angle = angle - robot_location['rotation']
    
#     if rot_angle > 0:
#         action_queue.append({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
#     else:
#         action_queue.append({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})
        
#     print ("Reached: ", dest_obj)
def GoToObject(robot, dest_obj):
    print("Going to ", dest_obj)
    # check if robot is a list
    if not isinstance(robot, list):
        robot = [robot]
    no_agents = len(robot)
    dist_goals = [10.0] * len(robot)
    prev_dist_goals = [10.0] * len(robot)
    count_since_update = [0] * len(robot)
    clost_node_location = [0] * len(robot)
    
    # list of objects in the scene and their centers
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])

    # look for the location and id of the destination object
    dest_obj_id = None
    dest_obj_center = None
    for idx, obj in enumerate(objs):
        match = re.match(dest_obj, obj)
        if match is not None:
            dest_obj_id = obj
            dest_obj_center = objs_center[idx]
            break # find the first instance
    
    if dest_obj_center is None:
        raise ValueError(f"Destination object '{dest_obj}' not found")

    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
    
    # closest reachable position for each robot
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 0.3
    # at least one robot is far away from the goal
    while all(d > goal_thresh for d in dist_goals):
        for ia, bot in enumerate(robot):
            robot_name = bot['name']
            agent_id = int(robot_name[-1]) - 1
            
            # get the pose of robot        
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            if dist_del < 0.2:
                count_since_update[ia] += 1
            else:
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 15:
                action_queue.append({'action': 'ObjectNavExpertAction', 'position': dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id': agent_id})
            else:    
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
            time.sleep(0.5)

    # align the robot once goal is reached
    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}
    
    robot_object_vec = [dest_obj_pos[0] - robot_location['x'], dest_obj_pos[2] - robot_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
    
    angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
    angle = 360 * angle / (2 * np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']
    
    if rot_angle > 0:
        action_queue.append({'action': 'RotateRight', 'degrees': abs(rot_angle), 'agent_id': agent_id})
    else:
        action_queue.append({'action': 'RotateLeft', 'degrees': abs(rot_angle), 'agent_id': agent_id})
        
    print("Reached: ", dest_obj)

def ExploreObject(robots, dest_obj, dest_obj2):
    print ("Explore ", dest_obj2)
    # check if robots is a list
    
    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    # robots distance to the goal 
    dist_goals = [10.0] * len(robots)
    dist_goals2 = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)
    
    # list of objects in the scene and their centers
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])

    # look for the location and id of the destination object
    for idx, obj in enumerate(objs):
        match = re.match(dest_obj2, obj)
        if match is not None:
            dest_obj_id2 = obj
            dest_obj_center2 = objs_center[idx]
            break # find the first instance   

    metadata = c.last_event.events[0].metadata
    location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}

    dest_obj_pos2 = [dest_obj_center2['x'], dest_obj_center2['y'], dest_obj_center2['z']] 
    dest_obj_pos =  [dest_obj[0], metadata["agent"]["position"]["y"], dest_obj[2]] 
    # closest reachable position for each robot
    # all robots cannot reach the same spot 
    # differt close points needs to be found for each robot
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 0.3
    # at least one robot is far away from the goal
    exit_flag = False
    exit_goto = False
    while all(d > goal_thresh for d in dist_goals):
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = int(robot_name[-1]) - 1
            
            # get the pose of robot        
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal

            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            dist_goals2[ia] =distance_pts([location['x'], location['y'], location['z']], dest_obj_pos2)
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            # print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
            if dist_goals2[ia] < 1.5:
                exit_flag = True
                break 
            if dist_del < 0.2:
                # robot did not move 
                count_since_update[ia] += 1
            else:
                # robot moving 
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 15:
                action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:    
                #updating goal
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
            time.sleep(0.5)
        if exit_flag == True:
            exit_goto = True
            print("find ",dest_obj2)
            break
    # align the robot once goal is reached
    # compute angle between robot heading and object
    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}
    
    robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
    
    angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    angle = 360*angle/(2*np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']
    
    if rot_angle > 0:
        action_queue.append({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    else:
        action_queue.append({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})
        
    print ("Reached: ", dest_obj)
    return exit_goto
def GoToObject_next_time(robots, dest_obj, json_file='/home/user/wzx/karma/memory/objects_locations.json', json_file2='/home/user/wzx/karma/memory/objects_locations2.json'):
    print ("Going to ", dest_obj)
    # check if robots is a list

    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    # robots distance to the goal 
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)
    
    with open(json_file, 'r') as f:
        objects = json.load(f)
    objs = {obj['objectId']: obj for obj in objects}

    with open(json_file2, 'r') as f1:
        objects2 = json.load(f1)
    objs2 = {obj2['objectId']: obj2 for obj2 in objects2}
    
    dest_obj_id, dest_obj_center = None, None
    dest_obj_id2, dest_obj_center2 = None, None
    for obj_id, obj_data in objs.items():
        if re.match(dest_obj, obj_id):
            dest_obj_id = obj_id
            dest_obj_center = obj_data['position']
            break
    for obj_id2, obj_data2 in objs2.items():
        if re.match(dest_obj, obj_id2):
            dest_obj_id2 = obj_id2
            dest_obj_center2 = obj_data2['position']
            break 
        
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
    dest_obj_pos2 = [dest_obj_center2['x'], dest_obj_center2['y'], dest_obj_center2['z']] 
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 0.3
    
    while all(d > goal_thresh for d in dist_goals):
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = int(robot_name[-1]) - 1
            
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia]
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            if dist_del < 0.2:
                count_since_update[ia] += 1
            else:
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 15:
                action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
            time.sleep(0.5)

    reach_flag = False
    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}
    
    robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
    real_distance = ((dest_obj_pos2[0]-robot_location['x']) ** 2 + (dest_obj_pos2[2]-robot_location['z']) ** 2) ** 0.5
    if real_distance <= 0.3:
        reach_flag = True
    elif real_distance > 0.3:
        reach_flag = False
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
    
    angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    angle = 360*angle/(2*np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']
    
    if rot_angle > 0:
        action_queue.append({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    else:
        action_queue.append({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    if reach_flag:
        print ("Reached: ", dest_obj)
    elif not reach_flag:
        print ("Failed to Reach: ", dest_obj)
    return reach_flag
def GoToObject_with_memory(robots, dest_obj, json_file='/home/user/wzx/karma/memory/memory3.json'):
    print ("Going to ", dest_obj)
    # check if robots is a list

    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)
    
    with open(json_file, 'r') as f:
        objects = json.load(f)
    objs = {obj['objectId']: obj for obj in objects}

    dest_obj_id, dest_obj_center = None, None
    dest_obj_id2, dest_obj_center2 = None, None
    for obj_id, obj_data in objs.items():
        if re.match(dest_obj, obj_id):
            dest_obj_id = obj_id
            dest_obj_center = obj_data['position']
            break
           
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 0.3
    
    while all(d > goal_thresh for d in dist_goals):
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = int(robot_name[-1]) - 1
            
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia]
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            if dist_del < 0.2:
                count_since_update[ia] += 1
            else:
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 15:
                action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
            time.sleep(0.5)

    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}
    
    robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
    
    angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    angle = 360*angle/(2*np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']
    
    if rot_angle > 0:
        action_queue.append({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    else:
        action_queue.append({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    
    print ("Reached: ", dest_obj)
    

def PickupObject(robot, pick_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(pick_obj, obj)
        if match is not None:
            pick_obj_id = obj
            break
        
    action_queue.append({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
    
def PutObject(robot, put_obj, recp):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    objs_dists = list([obj["distance"] for obj in c.last_event.metadata["objects"]])
    
    metadata = c.last_event.events[agent_id].metadata
    robot_location = [metadata["agent"]["position"]["x"], metadata["agent"]["position"]["y"], metadata["agent"]["position"]["z"]]
    dist_to_recp = 9999999 
    for idx, obj in enumerate(objs):
        match = re.match(recp, obj)
        if match is not None:
            dist = objs_dists[idx]
            if dist < dist_to_recp:
                recp_obj_id = obj
                dest_obj_center = objs_center[idx]
                dist_to_recp = dist
    action_queue.append({'action':'PutObject', 'objectId':recp_obj_id, 'agent_id':agent_id})
         
def SwitchOn(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break
    
    action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})      
        
def SwitchOff(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break
    
    action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})        

def OpenObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break
    
    action_queue.append({'action':'OpenObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    
def CloseObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break
    
    action_queue.append({'action':'CloseObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    
def BreakObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break
    
    action_queue.append({'action':'BreakObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    
def SliceObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break
    
    action_queue.append({'action':'SliceObject', 'objectId':sw_obj_id, 'agent_id':agent_id})      
  
def CleanObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break

    action_queue.append({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 

def ThrowObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break
    action_queue.append({'action':'ThrowObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    time.sleep(1)

def Explore(robot, sw_obj,available_positions):   
    exit_goto = False
    exit_goto_finish = False
    #定义随机探索的点位（根据地图：选择9个点，尽量覆盖地图的大部分区域）
    # available_positions = [
    #     (1.25, 0, -1.75),
    #     (-1.0, 0, 0),
    #     (-0.25, 0, -1.5),
    #     (-1.0, 0, -1.5),
    #     (0.5, 0, 1.5),
    #     (1.5, 0, -0.25),
    #     (1.5, 0, 1.0),
    #     (-2.0, 0, 2.0)
    # ]
    explore_point_count = 0
    # rand_position, available_positions = generate_random_position_from_list(available_positions)
    # 0: Task 1: Wash the Apple
    # 1: Go to the Apple.
    for positions in available_positions:
       if exit_goto_finish:
           break
       
       exit_goto=ExploreObject(robot, positions, sw_obj)
       explore_point_count += 1 

       if exit_goto:
            GoToObject(robot, sw_obj)
            exit_goto_finish = True
    print(explore_point_count)

def execute_tasks(robot):
    while True:
        task = task_queue.get()
        if task is None:
            break
        task_function = task['function']
        task_function(robot)  # 将 robot 传递给任务函数
        task_queue.task_done()

def add_task_to_queue(task_function, robot):
    task_queue.put({'function': task_function, 'robot': robot})

def parse_and_execute_task(robot):
    try:
        # Load the generated function name from the .json file
        with open('/home/user/wzx/karma/logs/generated_function_name.json', 'r') as file:
            data = json.load(file)
            function_name = data["function_name"]
        
        # Dynamically import the module and reload it
        module = importlib.import_module('task_functions')
        importlib.reload(module)
        task_function = getattr(module, function_name)
        
        # Add the task to the queue
        add_task_to_queue(task_function, robot)
    except ImportError as e:
        print(f"No task function matches the description, error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def parse_task(robot, task_description):
    # 使用传入的task_description来执行任务
    print(f"Executing task: {task_description}")
    # 添加执行任务的代码
    return 

def run_scripts():
    try:
        subprocess.run(['python', '/home/user/wzx/karma/scripts/query_with_short_term_memory.py'], check=True)
        subprocess.run(['python', '/home/user/wzx/karma/scripts/llm_as_planner.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running scripts: {e}")

task_executor_thread = threading.Thread(target=execute_tasks, args=(robots[0],))
task_executor_thread.start()


