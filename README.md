# **KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems**

Anonymous EMNLP submission.

Submitted to The 2024 Conference on Empirical Methods in Natural Language Processing(EMNLP).

## Setup

To get started, clone the KARMA repository:
```
git clone https://github.com/WZX0Swarm0Robotics/KARMA
```

Switch to the karma directory
```
cd KARMA
```

Create a conda environment (or virtualenv):
```
conda env create -f environment.yml
```

Activate the virtual environment
```
conda activate karma
```

## Creating OpenAI API Key
The code relies on OpenAI API. Create an API Key at https://platform.openai.com/.

In file /karma/scripts/llm_as_planner.py, change 'your_key' in line 4 of the code to your own OpenAI api key.

In file /karma/scripts/execute_LLM_plan.py, add the following code: api_key = 'your_key'

## Running Script
Run the following command to generate output execuate python scripts to perform the tasks in the given AI2Thor floor plans. 

Refer to https://ai2thor.allenai.org/demo for the layout of various AI2Thor floor plans.

```
python3 scripts/GUI_karma.py 
```
Note: You can enter the tasks you want the agent to perform in the GUI, for example："wash an apple and put it on the countertop", "slice an apple and place it on the plate". 

