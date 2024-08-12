import openai
import json
# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = 'your_key'
similarity_flag_path = '/home/user/wzx/karma/logs/similarity_flag.json'
messages_path = '/home/user/wzx/karma/logs/messages.json'

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# def insert_code_into_file(api_generated_code, target_file_path, line_number):
#     with open(target_file_path, 'r') as file:
#         file_content = file.readlines()
    
#     insert_index = line_number - 1
    
#     if insert_index > len(file_content):
#         insert_index = len(file_content)
    
#     file_content.insert(insert_index, api_generated_code + '\n')
    
#     with open(target_file_path, 'w') as file:
#         file.writelines(file_content)

def insert_code_into_file(new_code, target_file_path, line_number):
    with open(target_file_path, 'r') as file:
        lines = file.readlines()
    
    # 保留插入行号之前的内容
    before_lines = lines[:line_number - 1]
    
    # 组合新的文件内容：插入之前的内容 + 新代码
    new_lines = before_lines + new_code.splitlines(keepends=True)
    
    # 写回文件
    with open(target_file_path, 'w') as file:
        file.writelines(new_lines)

# example
api_generated_code = '''
def new_function():
    print("This is a new function")
'''
target_file_path = 'task_functions.py'
line_number = 6
insert_code_into_file(api_generated_code, target_file_path, line_number)

def write_to_file(file_path, content):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(content + '\n')

def load_similarity_flag():
    try:
        with open(similarity_flag_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get("similarity_flag", False)
    except FileNotFoundError:
        return False

use_short_term_memory = load_similarity_flag()
       
skills = load_file('/home/user/wzx/karma/prompts/skills.txt')
skills_ex = load_file('/home/user/wzx/karma/resources/actions.py')
role = load_file('/home/user/wzx/karma/prompts/role.txt')
examples = load_file('/home/user/wzx/karma/prompts/examples.txt')
emphasize = load_file('/home/user/wzx/karma/prompts/emphasize.txt')
instruction = load_file('/home/user/wzx/karma/prompts/instruction.txt')
short_term_memory = load_file('/home/user/wzx/karma/prompts/short_term_memory.txt')
long_term_memory = load_file('/home/user/wzx/karma/prompts/long_term_memory.txt')


messages = [
    {"role": "user", "content": skills},
    {"role": "user", "content": skills_ex},
    {"role": "system", "content": role},
    {"role": "user", "content": examples},
    {"role": "user", "content": emphasize},
    {"role": "user", "content": long_term_memory},
    {"role": "user", "content": instruction}
]

if use_short_term_memory:
    messages.insert(5, {"role": "user", "content": short_term_memory})

with open(messages_path, 'w', encoding='utf-8') as file:
    json.dump(messages, file, ensure_ascii=False, indent=4)    

response = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo-1106",
    model="gpt-4o-mini-2024-07-18",
    messages=messages,
    max_tokens=4096,
    temperature=0,
    stop=None
)

response_content = response.choices[0].message['content'].strip()

lines = response_content.split('\n')
code_lines = []
recording = False
function_name = None

for line in lines:
    if line.strip().startswith("def "):
        recording = True
        function_name = line.split('(')[0].split()[1]
    if recording:
        if line.strip() == '```':
            continue
        code_lines.append(line)

api_generated_code = '\n'.join(code_lines).strip()

print(api_generated_code)

target_file_path = 'task_functions.py'
line_number = 6 # 插入代码的行号
insert_code_into_file(api_generated_code, target_file_path, line_number)

# Save the function name to a file for later use
file_path = '/home/user/wzx/karma/logs/generated_function_name.json'
with open(file_path, 'w') as file:
    json.dump({"function_name": function_name}, file)





