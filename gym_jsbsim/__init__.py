import gym.envs.registration
from gym_jsbsim.tasks import Task, HeadingControlTask, TurnHeadingControlTask
from gym_jsbsim.aircraft import Aircraft, cessna172P
from typing import Type
Shaping = HeadingControlTask.Shaping


# This script registers all combinations of task, aircraft, shaping settings
# etc. with OpenAI Gym so that they can be instantiated with a gym.make(id)
# command.
#
# To get the ID of an environment with the desired configuration, use the
# get_env_id(...) function.

def get_env_id(task: Type[Task], aircraft: Aircraft, shaping: Shaping, enable_flightgear: bool) -> str:
    if enable_flightgear:
        fg_setting = 'FG'
    else:
        fg_setting = 'NoFG'
    return f'{task.__name__}-{aircraft.name}-{shaping}-{fg_setting}-v0'


entry_points = {True: 'gym_jsbsim.environment:JsbSimEnv',
                False: 'gym_jsbsim.environment:NoFGJsbSimEnv'}


for task in (HeadingControlTask, TurnHeadingControlTask):
    for plane in (cessna172P,):
        for shaping in (Shaping.OFF, Shaping.BASIC, Shaping.ADDITIVE):
            for enable_flightgear in (True, False):
                entry_point = entry_points[enable_flightgear]
                env_id = get_env_id(task, plane, shaping, enable_flightgear)
                kwargs = dict(task_type=task,
                              aircraft=plane,
                              shaping=shaping)
                gym.envs.registration.register(id=env_id,
                                               entry_point=entry_point,
                                               kwargs=kwargs)
