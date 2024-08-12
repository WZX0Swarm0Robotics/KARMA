def GoToObject(robots, dest_obj):
    # Navigate to the object. 

    # If agent knows the location of object, the agent can use this function to navigates to the object. 
    # If agent does not know the location of object, the robot should use the Explore(robots, dest_obj) to find the object.

    # The function captures only those objects that are within the agent's line of sight.
    
    # Example: 
    # <Instruction> Go to the apple(The memory contains the location of apple). 
    # Python script: 
    # GoToObject(robot,'Apple')
    pass
    
def PickupObject(robot, pick_obj):
    # pickup the object. 
    # The function captures only those objects that are within the agent's line of sight.
    
    # Example: 
    # <Instruction> Go get the apple on the kitchen counter. 
    # Python script: 
    # Explore(robot,'CounterTop')
    # GoToObject(robot,'CounterTop')
    # PickupObject(robot,'CounterTop')        
    pass

def PutObject(robot, put_obj, recp): 
    # puts the current interactive object held by the agent in the designated location. 
    # This function assumes the object is already picked up.

    # Example: 
    # <Instruction> put the apple on the Sink. 
    # Python script: 
    # Explore(robot,'Sink')
    # GoToObject(robot,'Sink')
    # PutObject(robot,'Sink')  
    pass

def SwitchOn(robot, sw_obj):
    # Turn on a switch. 
    
    # Example: 
    # <Instruction> Turn on the light. 
    # Python script: 
    # SwitchOn(robot,'LightSwitch') 
    pass

def SwitchOff(robot, sw_obj):
    # Turn off a switch. 
    
    # Example: 
    # <Instruction> Turn off the light. 
    # Python script: 
    # SwitchOn(robot,'LightSwitch') 
    pass

def OpenObject(robot, sw_obj):
    # Open the interaction object.
    # This function assumes the object is already closed and the agent has already navigated to the object. 
    # Only some objects can be opened. Fridges, cabinets, and drawers are some example of objects that can be closed.

    #Example: 
    # <Instruction> Get the apple in the fridge. 
    # Python script: 
    # Explore(robot,'Fridge')
    # GoToObject(robot,'Fridge')
    # OpenObject(robot,'Fridge') 
    # PickupObject(robot,'apple') 
    pass
    
def CloseObject(robot, sw_obj):
    # Close the interaction object.
    # This function assumes the object is already open and the agent has already navigated to the object. 
    # Only some objects can be closed. Fridges, cabinets, and drawers are some example of objects that can be closed.
    pass
    
def BreakObject(robot, sw_obj):
    # Break the interactable object.
    pass
    
def SliceObject(robot, sw_obj):
    # Slice the interactable object.
    # Only some objects can be sliced. Apple, tomato, and bread are some example of objects that can be sliced.

    #Example: 
    # <Instruction> Slice an apple. 
    # Python script: 
    # Explore(robot,'Knife')
    # GoToObject(robot,'Knife')
    # PickupObject(robot,'Knife')
    # Explore(robot,'Apple')
    # GoToObject(robot,'Apple')
    # SliceObject(robot,'Apple') 
    pass

def ThrowObject(robot, sw_obj):
    # Throw away the object.
    # This function assumes the object is already picked up.
    pass
    
def Explore(robot, sw_obj, position):
    # Explore the environment in the given sequence of locations until the target object becomes visible in the robot's field of view.
    pass

def ToggleOn(robot, sw_obj):
    # Toggles on the interaction object.
    # This function assumes the interaction object is already off and the agent has navigated to the object. 
    # Only some landmark objects can be toggled on. Lamps, stoves, and microwaves are some examples of objects that can be toggled on.

    # Example:  
    # <Instruction> Turn on the lamp. 
    # Python script: 
    # Explore(robot,'Lamp')
    # GoToObject(robot,'Lamp')
    # ToggleOn(robot,'Lamp')
    pass
    
def ToggleOff(robot, sw_obj):
    # Toggles off the interaction object.
    pass