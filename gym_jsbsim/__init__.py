import gym.envs.registration
from gym_jsbsim.tasks import Task, HeadingControlTask, TurnHeadingControlTask
from gym_jsbsim.aircraft import Aircraft, cessna172P
from typing import Type, Tuple, Dict


# This script registers all combinations of task, aircraft, shaping settings
# etc. with OpenAI Gym so that they can be instantiated with a gym.make(id)
# command.
#
# To get the ID of an environment with the desired configuration, use the
# get_env_id(...) function.

def get_env_id(task_type: Type[Task], plane: Aircraft, shaping: HeadingControlTask.Shaping,
               enable_flightgear: bool) -> str:
    if enable_flightgear:
        fg_setting = 'FG'
    else:
        fg_setting = 'NoFG'
    return f'JSBSim-{task_type.__name__}-{plane.name}-{shaping}-{fg_setting}-v0'


def get_env_id_kwargs_map() -> Dict[str, Tuple[Task, Aircraft, HeadingControlTask.Shaping, bool]]:
    """ Returns all environment IDs mapped to tuple of (task, aircraft, shaping, flightgear) """
    map = {}
    for task_type in (HeadingControlTask, TurnHeadingControlTask):
        for plane in (cessna172P,):
            for shaping in (HeadingControlTask.Shaping.OFF, HeadingControlTask.Shaping.BASIC,
                    HeadingControlTask.Shaping.ADDITIVE):
                for enable_flightgear in (True, False):
                    id = get_env_id(task_type, plane, shaping, enable_flightgear)
                    assert id not in map
                    map[id] = (task_type, plane, shaping, enable_flightgear)
    return map


entry_points = {True: 'gym_jsbsim.environment:JsbSimEnv',
                False: 'gym_jsbsim.environment:NoFGJsbSimEnv'}

for env_id, (task, plane, shaping, enable_flightgear) in get_env_id_kwargs_map().items():
    entry_point = entry_points[enable_flightgear]
    kwargs = dict(task_type=task,
                  aircraft=plane,
                  shaping=shaping)
    gym.envs.registration.register(id=env_id,
                                   entry_point=entry_point,
                                   kwargs=kwargs)
