import subprocess

def run_script(script_path):
    try:
        result = subprocess.run(['python', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output of {script_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:\n{e.stderr}")

# 先运行 llm_as_planner.py
print("Running llm_as_planner.py...")
run_script('/home/user/wzx/SMART-LLM1/scripts/llm_as_planner.py')

# 然后运行 execute_LLM_plan.py
print("Running execute_LLM_plan.py...")
run_script('/home/user/wzx/SMART-LLM1/scripts/execute_LLM_plan.py')
