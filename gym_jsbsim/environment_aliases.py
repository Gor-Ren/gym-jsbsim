from gym_jsbsim.environment import JsbSimEnv, NoFGJsbSimEnv
from gym_jsbsim.tasks import SteadyLevelFlightTask, HeadingControlTask
from gym_jsbsim.deprecated_tasks import SteadyLevelFlightTask_v0, SteadyLevelFlightTask_v1


# convenience classes for specific task/aircraft combos for registration with OpenAI Gym
class SteadyLevelFlightCessnaEnv_v0(JsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask_v0, aircraft_name='c172p')


class SteadyLevelFlightCessnaEnv_NoFg_v0(NoFGJsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask_v0, aircraft_name='c172p')


class SteadyLevelFlightCessnaEnv_v1(JsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask_v1, aircraft_name='c172p')


class SteadyLevelFlightCessnaEnv_NoFg_v1(NoFGJsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask_v1, aircraft_name='c172p')


class SteadyLevelFlightCessnaEnv_v2(JsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask, aircraft_name='c172p')

class SteadyLevelFlightCessnaEnv_NoFg_v2(NoFGJsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask, aircraft_name='c172p')


class HeadingControlCessnaEnv_v0(JsbSimEnv):
    def __init__(self):
        super().__init__(task_type=HeadingControlTask, aircraft_name='c172p')


class HeadingControlCessnaEnv_NoFg_v0(NoFGJsbSimEnv):
    def __init__(self):
        super().__init__(task_type=HeadingControlTask, aircraft_name='c172p')
