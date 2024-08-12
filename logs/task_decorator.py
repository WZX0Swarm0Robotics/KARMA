import functools
from execute_LLM_plan import GoToObject, ExploreObject, PickupObject, PutObject, SwitchOn, SwitchOff, OpenObject, CloseObject, BreakObject, SliceObject, CleanObject
def replace_explore_with_custom(func):
    @functools.wraps(func)
    def wrapper(robot, *args, **kwargs):
        task_code = func.__code__
        task_globals = func.__globals__.copy()
        
        # 定义我们自定义的探索逻辑
        def custom_explore(robot, object_name):
            exit_goto = False
            exit_goto_finish = False
            available_positions = [
                (-1, 0.00, 0.0),
                (0.25, 0.00, -1.5),
                (-1.0, 0.00, -1.50),
                (-2.0, 0.00, 2.0),
                (1.5, 0.00, -1.5),
                (1.5, 0.00, -0.25),
                (0.5, 0.00, 1.5),
                (1.5, 0.00, 1.0)
            ]
            explore_point_count = 0
            for positions in available_positions:
                if exit_goto_finish:
                    break
                
                exit_goto = ExploreObject(robot, positions, object_name)
                explore_point_count += 1

                if exit_goto:
                    GoToObject(robot, object_name)
                    exit_goto_finish = True
            
            print(explore_point_count)
        
        # 替换Explore函数为custom_explore
        task_globals['Explore'] = custom_explore
        
        return func(robot, *args, **kwargs)
    return wrapper
