from gym_jsbsim.environment import JsbSimEnv, NoFlightGearJsbSimEnv
from gym_jsbsim.tasks import SteadyLevelFlightTask
from gym_jsbsim.deprecated_tasks import SteadyLevelFlightTask_v0


# convenience classes for specific task/aircraft combos for registration with OpenAI Gym
class SteadyLevelFlightCessnaEnv_v0(JsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask_v0, aircraft_name='c172p')


class SteadyLevelFlightCessnaEnv_NoFg_v0(NoFlightGearJsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask_v0, aircraft_name='c172p')


class SteadyLevelFlightCessnaEnv_v1(JsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask, aircraft_name='c172p')


class SteadyLevelFlightCessnaEnv_NoFg_v1(NoFlightGearJsbSimEnv):
    def __init__(self):
        super().__init__(task_type=SteadyLevelFlightTask, aircraft_name='c172p')