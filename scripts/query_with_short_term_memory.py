import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def extract_task(description):
    # 找到冒号的位置
    colon_index = description.find(':')
    if colon_index != -1:
        # 提取冒号后面到第一个句号之间的部分
        task = description[colon_index + 1:].split('.')[0].strip()
        return task
    return None

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_to_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def update_memory_with_state(memory_file, analysis_file):
    # 读取 memory3.json 文件
    with open(memory_file, 'r', encoding='utf-8') as file:
        memory_data = json.load(file)
    
    # 读取 analysis_results.json 文件
    with open(analysis_file, 'r', encoding='utf-8') as file:
        analysis_data = json.load(file)
    
    # 提取 analysis_results.json 中最后一组数据的状态信息
    last_analysis_key = list(analysis_data.keys())[-1]
    state_data = {obj.lower(): state for obj, state in analysis_data[last_analysis_key].items()}

    # 更新 memory3.json 中的状态信息
    for item in memory_data:
        object_type = item.get('objectType', '').lower()
        if object_type in state_data:
            item['state'] = state_data[object_type]
    
    # 将更新后的数据写回 memory3.json 文件
    with open(memory_file, 'w', encoding='utf-8') as file:
        json.dump(memory_data, file, ensure_ascii=False, indent=4)

# 指定文件路径

analysis_file_path = '/home/user/wzx/karma/memory/analysis_results.json'

# 读取 memory3.json 中的物体数据
memory_file_path = '/home/user/wzx/karma/memory/memory3.json'
example_file_path = '/home/user/wzx/karma/experience/experience.json'
examples_output_path = '/home/user/wzx/karma/prompts/examples.txt'
# 更新memory3.json文件 with state
update_memory_with_state(memory_file_path, analysis_file_path)

items = load_json(memory_file_path)

# 读取并提取任务描述
instruction_file_path = '/home/user/wzx/karma/prompts/instruction.txt'
description = load_file(instruction_file_path)
extracted_task = extract_task(description)

# 输出提取的任务描述
if extracted_task:
    print(f"Extracted task: {extracted_task}")
else:
    print("No task found.")
    extracted_task = ""

# 如果提取到任务描述，则进行相似度计算
if extracted_task:
    item_texts = [item['objectType'] for item in items]
    
    # 初始化预训练的 transformer 模型
    model = SentenceTransformer('all-mpnet-base-v2')

    # 将物品名称转换为嵌入向量
    item_embeddings = model.encode(item_texts, convert_to_tensor=True)

    # 使用提取的任务描述作为查询
    query = extracted_task
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 使用余弦相似度计算查询与每个物品的相似度
    cosine_scores = util.pytorch_cos_sim(query_embedding, item_embeddings)[0]

    # 将PyTorch张量转换为NumPy数组
    cosine_scores = cosine_scores.cpu().numpy()

    # 获取相似度最高的一个物品
    top_result_idx = np.argsort(cosine_scores)[::-1][0]
    top_result_item = items[top_result_idx]

    # 打印最相似的物体
    print("Top matching item:")
    print(f"Object Type: {top_result_item['objectType']}, Position: {top_result_item['position']}, Score: {cosine_scores[top_result_idx]:.4f}")
 
    # 将最相似的物体保存到 /prompts/short_term_memory.txt 中
    short_term_memory_file_path = '/home/user/wzx/karma/prompts/short_term_memory.txt'
    object_type = top_result_item['objectType']
    position = top_result_item['position']
    formatted_content = f"{object_type} is at position ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})."
    save_to_file(short_term_memory_file_path, formatted_content)

    print(f"Top matching item has been saved to {short_term_memory_file_path}")
    

############ 匹配 相似的experience 中的物体

# 如果提取到任务描述，则进行相似度计算
if extracted_task:
    example_data = load_json(example_file_path)
    tasks = [example['task'] for example in example_data]

    # 初始化预训练的 transformer 模型
    model = SentenceTransformer('all-mpnet-base-v2')

    # 将任务描述转换为嵌入向量
    task_embeddings = model.encode(tasks, convert_to_tensor=True)

    # 使用提取的任务描述作为查询
    query_embedding = model.encode(extracted_task, convert_to_tensor=True)

    # 使用余弦相似度计算查询与每个任务的相似度
    cosine_scores = util.pytorch_cos_sim(query_embedding, task_embeddings)[0]

    # 将 PyTorch 张量转换为 NumPy 数组
    cosine_scores = cosine_scores.cpu().numpy()

    # 获取相似度最高的三个任务
    top_k_indices = np.argsort(cosine_scores)[::-1][:3]

    # 提取对应的 decomposition 内容
    top_decompositions = [example_data[idx]['decomposition'] for idx in top_k_indices]

    # 将提取的内容保存到 examples.txt 中
    with open(examples_output_path, 'w', encoding='utf-8') as file:
        for i, decomposition in enumerate(top_decompositions):
            file.write(f"Example {i+1} Decomposition:\n")
            file.write('\n'.join(decomposition))
            file.write('\n\n')

    print(f"Top 3 task decompositions have been saved to {examples_output_path}")
    # analysis_results_path = '/home/user/wzx/karma/memory/analysis_results.json'
    # analysis_results = load_json(analysis_results_path)

    # best_match = None
    # best_match_score = -1
    # for image, objects in analysis_results.items():
    #     for obj, state in objects.items():
    #         obj_embedding = model.encode(obj, convert_to_tensor=True)
    #         score = util.pytorch_cos_sim(query_embedding, obj_embedding)[0].cpu().numpy()
    #         if score > best_match_score:
    #             best_match_score = score
    #             best_match = (image, obj, state)
    
    # if best_match:
    #     image, obj, state = best_match
    #     analysis_content = f"{obj}'s state is {state}."
    #     formatted_content += f"\n{analysis_content}"

    # save_to_file(short_term_memory_file_path, formatted_content)

    # print(f"Top matching item and analysis result have been saved to {short_term_memory_file_path}")