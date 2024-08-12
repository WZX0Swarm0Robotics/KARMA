# import tkinter as tk
# from tkinter import messagebox
# from datetime import datetime
# import json
# import execute_LLM_plan

# history_file_path = '/home/user/wzx/karma/history_tasks/task_history.json'
# similarity_flag_path = '/home/user/wzx/karma/logs/similarity_flag.json'
# task_description_file_path = '/home/user/wzx/karma/logs/task_description.json'

# robots = [{'name': 'robot1', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}]

# objects_list = ["AlarmClock", "AluminumFoil", "Apple", "AppleSliced", "BaseballBat", "BasketBall", "Bathtub", 
#                 "BathtubBasin", "Bed", "Blinds", "book", "boots", "bottle", "bowl", "box", "Bread", "BreadSliced", 
#                 "ButterKnife", "Candle", "CD", "CellPhone", "Cloth", "CoffeeMachine", "CreditCard", "cup", "curtains", 
#                 "DeskLamp", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell", "egg", "EggCracked", "FloorLamp", 
#                 "Footstool", "fork", "GarbageBag", "HandTowel", "HandTowelHolder", "HousePlant", "Kettle", "KeyChain", 
#                 "Knife", "Ladle", "Laptop", "LaundryHamper", "Lettuce", "LettuceSliced", "LightSwitch", "Microwave", 
#                 "Mirror", "Mug", "Newspaper", "Ottoman", "Painting", "Pan", "PaperTowelRoll", "pen", "Pencil", 
#                 "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot", "Potato", "PotatoSliced", 
#                 "RemoteControl", "RoomDecor", "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", 
#                 "ShowerCurtain", "ShowerDoor", "ShowerGlass", "ShowerHead", "Sink", "SinkBasin", "SoapBar", 
#                 "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle", "Statue", "Stool", "StoveBurner", 
#                 "StoveKnob", "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", 
#                 "TissueBox", "Toaster", "Toilet", "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", 
#                 "Towel", "TowelHolder", "TVStand", "VacuumCleaner", "Vase", "Watch", "WateringCan", "Window", 
#                 "WineBottle"]

# def check_task_similarity(new_task, task_history, objects_list):
#     new_task_words = set(new_task.lower().split())
#     similarity_report = []
#     for task in task_history:
#         task_words = set(task.lower().split())
#         common_words = new_task_words & task_words
#         for word in common_words:
#             if any(obj.lower() == word for obj in objects_list):
#                 similarity_report.append(f"Memory: {word}")
#     similarity_flag = bool(similarity_report)
#     with open(similarity_flag_path, 'w', encoding='utf-8') as file:
#         json.dump({"similarity_flag": similarity_flag}, file)
#     return similarity_report, bool(similarity_report)

# def save_task():
#     task = task_entry.get().strip()
#     if task:
#         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         task_with_time = f"{task} ({current_time})"
#         content = f"Please help me decompose the following tasks: {task}. Please output only the generated code."
#         with open('/home/user/wzx/karma/prompts/instruction.txt', 'w', encoding='utf-8') as file:
#             file.write(content)
        
#         try:
#             with open(history_file_path, 'r', encoding='utf-8') as file:
#                 task_history = json.load(file)
#         except FileNotFoundError:
#             task_history = []
        
#         similarity_report, has_similarity = check_task_similarity(task, task_history, objects_list)
        
#         if similarity_report:
#             similarity_text.set("\n".join(similarity_report))
#         else:
#             similarity_text.set("No similar tasks found.")
        
#         task_history.append(task_with_time)
        
#         with open(history_file_path, 'w', encoding='utf-8') as file:
#             json.dump(task_history, file, ensure_ascii=False, indent=4)
        
#         with open(task_description_file_path, 'w', encoding='utf-8') as file:
#             json.dump({"task_description": task}, file, ensure_ascii=False, indent=4)
        
#         task_listbox.insert(tk.END, task_with_time)
        
#         messagebox.showinfo("Success", "Start the task!")
        
#         execute_LLM_plan.run_scripts()
        
#         # # 传递 task_description 参数
#         execute_LLM_plan.parse_and_execute_task(robots[0])
#     else:
#         messagebox.showwarning("Input Error", "Please enter a task.")

# def load_task_history():
#     try:
#         with open(history_file_path, 'r', encoding='utf-8') as file:
#             return json.load(file)
#     except FileNotFoundError:
#         return []

# def clear_task_history():
#     with open(history_file_path, 'w', encoding='utf-8') as file:
#         json.dump([], file, ensure_ascii=False, indent=4)
#     task_listbox.delete(0, tk.END)
#     similarity_text.set("")
#     messagebox.showinfo("Success", "Task history has been cleared!")

# def exit_application():
#     execute_LLM_plan.task_queue.put(None)
#     execute_LLM_plan.task_executor_thread.join()
#     messagebox.showinfo("Info", "All tasks have been executed.")
#     root.destroy()

# root = tk.Tk()
# root.title("Task Input")
# root.geometry("900x600")

# large_font = ("Helvetica", 16)

# task_label = tk.Label(root, text="Please enter the task you want the robot to help you complete:", font=large_font)
# task_label.pack(pady=20)

# task_entry = tk.Entry(root, width=100, font=large_font)
# task_entry.pack(pady=20)

# save_button = tk.Button(root, text="Start the task", font=large_font, command=save_task)
# save_button.pack(pady=20)

# task_listbox_label = tk.Label(root, text="Previously executed tasks:", font=large_font)
# task_listbox_label.pack(pady=10)

# task_listbox = tk.Listbox(root, width=100, height=10, font=large_font)
# task_listbox.pack(pady=10)

# task_history = load_task_history()
# for task in task_history:
#     task_listbox.insert(tk.END, task.strip())

# similarity_text = tk.StringVar()
# similarity_label = tk.Label(root, textvariable=similarity_text, font=large_font, fg="red")
# similarity_label.pack(pady=10)

# clear_button = tk.Button(root, text="Clear Task History", font=large_font, command=clear_task_history)
# clear_button.pack(pady=20)

# exit_button = tk.Button(root, text="Exit", font=large_font, command=exit_application)
# exit_button.pack(pady=20)

# root.mainloop()
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import json
import execute_LLM_plan

history_file_path = '/home/user/wzx/karma/history_tasks/task_history.json'
similarity_flag_path = '/home/user/wzx/karma/logs/similarity_flag.json'
task_description_file_path = '/home/user/wzx/karma/logs/task_description.json'
short_term_memory_path = '/home/user/wzx/karma/memory/memory3.json'

robots = [{'name': 'robot1', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}]

objects_list = ["AlarmClock", "AluminumFoil", "Apple", "AppleSliced", "BaseballBat", "BasketBall", "Bathtub", 
                "BathtubBasin", "Bed", "Blinds", "book", "boots", "bottle", "bowl", "box", "Bread", "BreadSliced", 
                "ButterKnife", "Candle", "CD", "CellPhone", "Cloth", "CoffeeMachine", "CreditCard", "cup", "curtains", 
                "DeskLamp", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell", "egg", "EggCracked", "FloorLamp", 
                "Footstool", "fork", "GarbageBag", "HandTowel", "HandTowelHolder", "HousePlant", "Kettle", "KeyChain", 
                "Knife", "Ladle", "Laptop", "LaundryHamper", "Lettuce", "LettuceSliced", "LightSwitch", "Microwave", 
                "Mirror", "Mug", "Newspaper", "Ottoman", "Painting", "Pan", "PaperTowelRoll", "pen", "Pencil", 
                "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot", "Potato", "PotatoSliced", 
                "RemoteControl", "RoomDecor", "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", 
                "ShowerCurtain", "ShowerDoor", "ShowerGlass", "ShowerHead", "Sink", "SinkBasin", "SoapBar", 
                "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle", "Statue", "Stool", "StoveBurner", 
                "StoveKnob", "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", 
                "TissueBox", "Toaster", "Toilet", "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", 
                "Towel", "TowelHolder", "TVStand", "VacuumCleaner", "Vase", "Watch", "WateringCan", "Window", 
                "WineBottle"]

def check_task_similarity(new_task, task_history, objects_list):
    new_task_words = set(new_task.lower().split())
    similarity_report = []
    for task in task_history:
        task_words = set(task.lower().split())
        common_words = new_task_words & task_words
        for word in common_words:
            if any(obj.lower() == word for obj in objects_list):
                similarity_report.append(f"Memory: {word}")
    similarity_flag = bool(similarity_report)
    with open(similarity_flag_path, 'w', encoding='utf-8') as file:
        json.dump({"similarity_flag": similarity_flag}, file)
    return similarity_report, bool(similarity_report)

def save_task():
    task = task_entry.get().strip()
    if task:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_with_time = f"{task} ({current_time})"
        content = f"Please help me decompose the following tasks: {task}. Please output only the generated code."
        with open('/home/user/wzx/karma/prompts/instruction.txt', 'w', encoding='utf-8') as file:
            file.write(content)
        
        try:
            with open(history_file_path, 'r', encoding='utf-8') as file:
                task_history = json.load(file)
        except FileNotFoundError:
            task_history = []
        
        similarity_report, has_similarity = check_task_similarity(task, task_history, objects_list)
        
        if similarity_report:
            similarity_text.set("\n".join(similarity_report))
        else:
            similarity_text.set("No similar tasks found.")
        
        task_history.append(task_with_time)
        
        with open(history_file_path, 'w', encoding='utf-8') as file:
            json.dump(task_history, file, ensure_ascii=False, indent=4)
        
        with open(task_description_file_path, 'w', encoding='utf-8') as file:
            json.dump({"task_description": task}, file, ensure_ascii=False, indent=4)
        
        task_listbox.insert(tk.END, task_with_time)
        
        messagebox.showinfo("Success", "Start the task!")
        
        execute_LLM_plan.run_scripts()
        
        # 传递 task_description 参数
        execute_LLM_plan.parse_and_execute_task(robots[0])
    else:
        messagebox.showwarning("Input Error", "Please enter a task.")

def load_task_history():
    try:
        with open(history_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def clear_task_history():
    with open(history_file_path, 'w', encoding='utf-8') as file:
        json.dump([], file, ensure_ascii=False, indent=4)
    task_listbox.delete(0, tk.END)
    similarity_text.set("")
    messagebox.showinfo("Success", "Task history has been cleared!")

def exit_application():
    execute_LLM_plan.task_queue.put(None)
    execute_LLM_plan.task_executor_thread.join()
    messagebox.showinfo("Info", "All tasks have been executed.")
    root.destroy()

def load_short_term_memory():
    try:
        with open(short_term_memory_path, 'r', encoding='utf-8') as file:
            memory_data = json.load(file)
            memory_text.delete(1.0, tk.END)
            memory_text.insert(tk.END, format_memory_data(memory_data))
    except FileNotFoundError:
        memory_text.delete(1.0, tk.END)
        memory_text.insert(tk.END, "No short-term memory found.")

def format_memory_data(memory_data):
    formatted_data = ""
    for item in memory_data:
        formatted_data += f"Object Type: {item['objectType']}\n"
        formatted_data += f"Position: x={item['position']['x']}, y={item['position']['y']}, z={item['position']['z']}\n"
        formatted_data += f"Object ID: {item['objectId']}\n\n"
    return formatted_data

def save_short_term_memory():
    memory_data = memory_text.get(1.0, tk.END).strip()
    try:
        # 这里假设输入的格式是正确的 JSON，可以进一步改进以检查和解析输入
        lines = memory_data.split('\n')
        json_data = []
        obj = {}
        for line in lines:
            if line.startswith("Object Type: "):
                if obj:
                    json_data.append(obj)
                obj = {"objectType": line.split("Object Type: ")[1]}
            elif line.startswith("Position: "):
                pos = line.split("Position: ")[1].split(', ')
                obj["position"] = {
                    "x": float(pos[0].split('=')[1]),
                    "y": float(pos[1].split('=')[1]),
                    "z": float(pos[2].split('=')[1])
                }
            elif line.startswith("Object ID: "):
                obj["objectId"] = line.split("Object ID: ")[1]
        if obj:
            json_data.append(obj)
        
        with open(short_term_memory_path, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)
        messagebox.showinfo("Success", "Short-term memory has been saved!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
root.title("Task Input")
root.geometry("900x600")

large_font = ("Helvetica", 16)

task_label = tk.Label(root, text="Please enter the task you want the robot to help you complete:", font=large_font)
task_label.pack(pady=20)

task_entry = tk.Entry(root, width=100, font=large_font)
task_entry.pack(pady=20)

save_button = tk.Button(root, text="Start the task", font=large_font, command=save_task)
save_button.pack(pady=20)

task_listbox_label = tk.Label(root, text="Previously executed tasks:", font=large_font)
task_listbox_label.pack(pady=10)

task_listbox = tk.Listbox(root, width=100, height=10, font=large_font)
task_listbox.pack(pady=10)

task_history = load_task_history()
for task in task_history:
    task_listbox.insert(tk.END, task.strip())

similarity_text = tk.StringVar()
similarity_label = tk.Label(root, textvariable=similarity_text, font=large_font, fg="red")
similarity_label.pack(pady=10)

clear_button = tk.Button(root, text="Clear Task History", font=large_font, command=clear_task_history)
clear_button.pack(pady=20)

exit_button = tk.Button(root, text="Exit", font=large_font, command=exit_application)
exit_button.pack(pady=20)

# 添加用于显示和编辑 short-term memory 的区域
memory_label = tk.Label(root, text="Short-term Memory:", font=large_font)
memory_label.pack(pady=10)

memory_text = tk.Text(root, width=100, height=10, font=large_font)
memory_text.pack(pady=10)

load_memory_button = tk.Button(root, text="Load Short-term Memory", font=large_font, command=load_short_term_memory)
load_memory_button.pack(pady=10)

save_memory_button = tk.Button(root, text="Save Short-term Memory", font=large_font, command=save_short_term_memory)
save_memory_button.pack(pady=10)

root.mainloop()



