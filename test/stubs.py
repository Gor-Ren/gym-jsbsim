from tasks import TaskModule
from typing import Optional


class TaskStub(TaskModule):
    """ A minimal task module for testing. """
    task_state_variables = ()

    def _calculate_reward(self, _):
        return 0

    def _is_done(self, _):
        return False

    def get_initial_conditions(self):
        return None

    def __init__(self, task_name: Optional[str]='TaskStub'):
        super().__init__(task_name)


class SimStub(dict):
    def run(self):
        pass
