import random
from typing import Dict, Optional
from gym_jsbsim.tasks import TaskModule
from gym_jsbsim.simulation import Simulation


class SteadyLevelFlightTask_v0(TaskModule):
    """ A task in which the agent must perform steady, level flight. """
    task_state_variables = (dict(name='velocities/h-dot-fps',
                                 description='earth frame altitude change rate [ft/s]',
                                 high=2200, low=-2200),
                            )
    # target values: prop_name, target_value, gain
    TARGET_VALUES = (('velocities/h-dot-fps', 0, 1),
                     ('attitude/roll-rad', 0, 10),
                     )

    MAX_TIME_SECS = 15
    MIN_ALT_FT = 1000
    TOO_LOW_REWARD = -10
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8

    def __init__(self, task_name='SteadyLevelFlightTask'):
        super().__init__(task_name)

    def get_initial_conditions(self) -> Optional[Dict[str, float]]:
        """ Returns dictionary mapping initial episode conditions to values.

        The aircraft is initialised at a random orientation and velocity.

        :return: dict mapping string for each initial condition property to
            a float, or None to use Env defaults
        """
        initial_conditions = {'ic/u-fps': 150,
                              'ic/v-fps': 0,
                              'ic/w-fps': 0,
                              'ic/p-rad_sec': 0,
                              'ic/q-rad_sec': 0,
                              'ic/r-rad_sec': 0,
                              'ic/roc-fpm': 0,  # rate of climb
                              'ic/psi-true-deg': random.uniform(0, 360),  # heading
                              }
        return {**self.base_initial_conditions, **initial_conditions}

    def get_full_action_variables(self):
        """ Returns information defining all action variables for this task.

        For steady level flight the agent controls ailerons, elevator and rudder.
        Throttle will be set in the initial conditions and maintained at a
        constant value.

        :return: tuple of dicts, each dict having a 'source', 'name',
            'description', 'high' and 'low' key
        """
        all_action_vars = super().get_full_action_variables()
        # omit throttle from default actions
        action_vars = tuple(var for var in all_action_vars if var['name'] != 'fcs/throttle-cmd-norm')
        assert len(action_vars) == len(all_action_vars) - 1
        return action_vars

    def _calculate_reward(self, sim: Simulation):
        """ Calculates the reward from the simulation state.

        :param sim: Simulation, the environment simulation
        :return: a number, the reward for the timestep
        """
        reward = 0
        for prop, target, gain in self.TARGET_VALUES:
            reward -= abs(target - sim[prop]) * gain
        too_low = sim['position/h-sl-ft'] < self.MIN_ALT_FT
        if too_low:
            reward += self.TOO_LOW_REWARD
        return reward

    def _is_done(self, sim: Simulation):
        """ Determines whether the current episode should terminate.

        :param sim: Simulation, the environment simulation
        :return: True if the episode should terminate else False
        """
        time_out = sim['simulation/sim-time-sec'] > self.MAX_TIME_SECS
        too_low = sim['position/h-sl-ft'] < self.MIN_ALT_FT
        return time_out or too_low

    def _input_initial_controls(self, sim: Simulation):
        """ Sets control inputs for start of episode.

        :param sim: Simulation, the environment simulation
        """
        # start engines and trims for steady, level flight
        sim.start_engines()
        sim['fcs/throttle-cmd-norm'] = self.THROTTLE_CMD
        sim['fcs/mixture-cmd-norm'] = self.MIXTURE_CMD
        sim.trim(Simulation.FULL)
        SteadyLevelFlightTask_v0._transfer_pitch_trim_to_cmd(sim)

    @staticmethod
    def _transfer_pitch_trim_to_cmd(sim: Simulation):
        """
        Removes a pitch trim and adds it to the elevator command.

        JSBSim's trimming utility may stabilise pitch by using a trim, which is
        a constant offset from the commanded position. However, agents use the
        elevator command directly and trim is not reflected in their state
        representation. Therefore we prefer to remove the trim and reflect it
        directly in the elevator commanded position.

        :param sim: the Simulation instance
        """
        PITCH_CMD = 'fcs/elevator-cmd-norm'
        PITCH_TRIM = 'fcs/pitch-trim-cmd-norm'
        total_elevator_cmd = sim[PITCH_CMD] + sim[PITCH_TRIM]
        sim[PITCH_CMD] = total_elevator_cmd
        sim[PITCH_TRIM] = 0.0
