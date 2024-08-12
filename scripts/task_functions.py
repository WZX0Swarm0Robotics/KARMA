from execute_LLM_plan import GoToObject, ExploreObject, PickupObject, PutObject, SwitchOn, SwitchOff, OpenObject, CloseObject, BreakObject, SliceObject, CleanObject, Explore
# from task_decorator import replace_explore_with_custom

# @replace_explore_with_custom

def wash_apple(robot):
    # 0: SubTask 1: Go to the apple
    # 1: Navigate to the known position of the apple.
    GoToObject(robot, 'Apple')
    # 2: Pick up the apple.
    PickupObject(robot, 'Apple')
    # 3: Explore the Sink.
    available_positions = [
        (-1.0, 0.00, -1.5),  # Center with Sink
        (-0.25, 0.00, -1.5), # Center with Sink
        (-2, 0.00, 2.0),     # Center with Fridge and GarbageCan
        (1.25, 0.00, -1.75),  # Center with Drawer and Shelf
        (1.5, 0.00, -0.25),   # Center with LightSwitch
        (0.5, 0.00, 1.5),     # Empty center
        (-1, 0.00, 0.0),      # Center with CoffeeMachine and Drawers
        (1.5, 0.00, 1.0)      # Empty center
    ]
    Explore(robot, 'Sink', available_positions)
    # 4: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 5: Wash the apple in the Sink.
    CleanObject(robot, 'Apple')