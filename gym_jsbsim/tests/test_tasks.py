import unittest
import math
import numpy as np
import sys
import gym_jsbsim.properties as prp
from gym_jsbsim import rewards, utils
from gym_jsbsim.assessors import Assessor, AssessorImpl
from gym_jsbsim.aircraft import Aircraft, cessna172P
from gym_jsbsim.tasks import Shaping, HeadingControlTask, TurnHeadingControlTask
from gym_jsbsim.tests.stubs import SimStub, TransitioningSimStub


class TestHeadingControlTask(unittest.TestCase):
    default_shaping = Shaping.STANDARD
    default_episode_time_s = 1
    default_step_frequency_hz = 1
    default_aircraft = cessna172P

    def setUp(self):
        self.task = self.make_task()
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)  # causes task to init new-episode attributes

        self.dummy_action = np.asarray([0 for _ in range(len(self.task.action_variables))])

    def get_class_under_test(self):
        return HeadingControlTask

    def make_task(self,
                  shaping_type: Shaping = default_shaping,
                  episode_time_s: float = default_episode_time_s,
                  step_frequency_hz: float = default_step_frequency_hz,
                  aircraft: Aircraft = default_aircraft) -> HeadingControlTask:
        task_class = self.get_class_under_test()
        return task_class(shaping_type=shaping_type,
                          episode_time_s=episode_time_s,
                          step_frequency_hz=step_frequency_hz,
                          aircraft=aircraft)

    def get_initial_sim_with_state(self,
                                   task: HeadingControlTask = None,
                                   time_terminal=False,
                                   track_deg=0.0,
                                   altitude_ft=None,
                                   roll_rad=0.0) -> SimStub:
        if task is None:
            task = self.task
        sim = SimStub.make_valid_state_stub(task)
        task.observe_first_state(sim)

        return self.modify_sim_to_state_(sim, task, time_terminal, track_deg,
                                         altitude_ft, roll_rad)

    def modify_sim_to_state_(self,
                             sim: SimStub,
                             task: HeadingControlTask = None,
                             time_terminal=False,
                             track_deg=0.0,
                             altitude_ft=None,
                             roll_rad=0.0,
                             sideslip_deg=0.0) -> SimStub:
        if task is None:
            task = self.task
        if time_terminal:
            sim[prp.sim_time_s] = task.max_time_s + 1
        else:
            sim[prp.sim_time_s] = task.max_time_s - 1

        if altitude_ft is None:
            sim[prp.altitude_sl_ft] = task.INITIAL_ALTITUDE_FT
        else:
            sim[prp.altitude_sl_ft] = altitude_ft

        v_east = math.sin(math.radians(track_deg))
        v_north = math.cos(math.radians(track_deg))
        sim[prp.v_east_fps] = v_east
        sim[prp.v_north_fps] = v_north
        sim[task.track_error_deg] = 0.0  # consistent with the velocities set
        sim[prp.roll_rad] = roll_rad
        sim[prp.sideslip_deg] = sideslip_deg
        return sim

    def get_initial_state_sim(self, task=None) -> SimStub:
        if task is None:
            task = self.task
        sim = SimStub.make_valid_state_stub(task)

        # set properties to reasonable initial episode values
        sim[prp.sim_time_s] = 0.0
        sim[prp.dist_travel_m] = 0.0
        sim[prp.heading_deg] = task.INITIAL_HEADING_DEG
        sim[prp.altitude_sl_ft] = task.get_initial_conditions()[prp.initial_altitude_ft]
        sim[prp.roll_rad] = 0
        return sim

    def get_perfect_state_sim(self, task=None, time_terminal=True) -> SimStub:
        if task is None:
            task = self.task
        sim = SimStub.make_valid_state_stub(task)
        task.observe_first_state(sim)

        perfect_track = sim[task.target_track_deg]
        perfect_altitude = task._get_target_altitude()
        perfect_roll = 0.0
        perfect_sideslip = 0.0

        self.modify_sim_to_state_(sim, task, time_terminal=time_terminal, track_deg=perfect_track,
                                  altitude_ft=perfect_altitude, roll_rad=perfect_roll,
                                  sideslip_deg=perfect_sideslip)

        return sim

    def get_transitioning_sim(self, task, time_terminal, altitude_ft, roll_rad,
                              track_error_deg, sideslip_deg) -> TransitioningSimStub:
        """ Makes a sim that transitions between two states.

        The first state is as configured by the input task.
        The second state's altitude, distance traveled, heading etc. changes as per inputs.
         """
        initial_state_sim = self.get_initial_sim_with_state(task, time_terminal=False)
        target_heading = initial_state_sim[self.task.target_track_deg]
        track_deg = target_heading + track_error_deg
        next_state_sim = initial_state_sim.copy()
        next_state_sim = self.modify_sim_to_state_(next_state_sim,
                                                   time_terminal=time_terminal,
                                                   altitude_ft=altitude_ft,
                                                   roll_rad=roll_rad,
                                                   track_deg=track_deg,
                                                   sideslip_deg=sideslip_deg)
        return TransitioningSimStub(initial_state_sim, next_state_sim)

    def get_target_track(self, sim: SimStub):
        return self.task.INITIAL_HEADING_DEG

    def test_init_shaping_standard(self):
        task = self.make_task(shaping_type=Shaping.STANDARD)

        self.assertIsInstance(task.assessor, Assessor)
        self.assertEqual(4, len(task.assessor.base_components))
        self.assertFalse(task.assessor.potential_components)  # assert empty

    def test_init_shaping_sequential_cont(self):
        task = self.make_task(shaping_type=Shaping.SEQUENTIAL_CONT)

        self.assertIsInstance(task.assessor, Assessor)
        self.assertEqual(4, len(task.assessor.base_components))
        self.assertEqual(0, len(task.assessor.potential_components))

    def test_get_intial_conditions_valid_target_heading(self):
        self.setUp()

        ics = self.task.get_initial_conditions()
        initial_heading = ics[prp.initial_heading_deg]

        self.assertLessEqual(prp.heading_deg.min, initial_heading)
        self.assertGreaterEqual(prp.heading_deg.max, initial_heading)

    def test_get_initial_conditions_contains_all_props(self):
        ics = self.task.get_initial_conditions()

        self.assertIsInstance(ics, dict)
        for prop, value in self.task.base_initial_conditions.items():
            self.assertAlmostEqual(value, ics[prop])

        steady_level_task_ic_properties = [prp.initial_u_fps,
                                           prp.initial_v_fps,
                                           prp.initial_w_fps,
                                           prp.initial_p_radps,
                                           prp.initial_q_radps,
                                           prp.initial_r_radps,
                                           prp.initial_heading_deg
                                           ]
        for prop in steady_level_task_ic_properties:
            self.assertIn(prop, ics.keys(),
                          msg='expected HeadingControlTask to set value for'
                              f'property {prop} but not found in ICs')

    def test_observe_first_state_returns_valid_state(self):
        sim = SimStub.make_valid_state_stub(self.task)

        first_state = self.task.observe_first_state(sim)

        self.assertEqual(len(first_state), len(self.task.state_variables))
        self.assertIsInstance(first_state, tuple)

    def test_task_first_observation_inputs_controls(self):
        dummy_sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(dummy_sim)

        # check engine as expected
        self.assertAlmostEqual(self.task.THROTTLE_CMD, dummy_sim[prp.throttle_cmd])
        self.assertAlmostEqual(self.task.MIXTURE_CMD, dummy_sim[prp.mixture_cmd])
        self.assertAlmostEqual(1.0, dummy_sim[prp.engine_running])

    def test_task_step_correct_return_types(self):
        sim = SimStub.make_valid_state_stub(self.task)
        steps = 1
        _ = self.task.observe_first_state(sim)

        state, reward, is_terminal, info = self.task.task_step(sim, self.dummy_action, steps)

        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), len(self.task.state_variables))

        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_terminal, bool)
        self.assertIsInstance(info, dict)

    def test_task_step_returns_reward_in_info(self):
        sim = SimStub.make_valid_state_stub(self.task)
        steps = 1
        _ = self.task.observe_first_state(sim)

        _, reward_scalar, _, info = self.task.task_step(sim, self.dummy_action, steps)
        reward_object = info['reward']

        self.assertIsInstance(reward_object, rewards.Reward)
        self.assertAlmostEqual(reward_object.agent_reward(), reward_scalar)

    def test_task_step_returns_non_terminal_time_less_than_max(self):
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)
        non_terminal_time = self.default_episode_time_s - 1
        sim[prp.sim_time_s] = non_terminal_time
        steps = 1

        _, _, is_terminal, _ = self.task.task_step(sim, self.dummy_action, steps)

        self.assertFalse(is_terminal)

    def test_task_step_returns_terminal_time_exceeds_max(self):
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)
        terminal_time = self.default_episode_time_s + 1
        sim[prp.sim_time_s] = terminal_time
        steps = 1

        _, _, is_terminal, _ = self.task.task_step(sim, self.dummy_action, steps)

        self.assertTrue(is_terminal)

    def test_task_step_returns_terminal_time_equals_max(self):
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)
        terminal_time = self.default_episode_time_s
        sim[prp.sim_time_s] = terminal_time
        steps = 1

        _, _, is_terminal, _ = self.task.task_step(sim, self.dummy_action, steps)

        self.assertTrue(is_terminal)

    def test_task_step_correct_terminal_reward_optimal_behaviour_no_shaping(self):
        self.setUp()
        task = self.make_task(shaping_type=Shaping.STANDARD,
                              episode_time_s=1.,
                              step_frequency_hz=1.)
        initial_state_sim = self.get_initial_state_sim(task)
        _ = task.observe_first_state(initial_state_sim)
        final_state_sim = self.get_perfect_state_sim(task, time_terminal=True)
        sim = TransitioningSimStub(initial_state_sim, final_state_sim)

        state, reward, done, info = task.task_step(sim, self.dummy_action, 1)

        # aircraft moved maximum distance on correct heading and maintained
        # altitude, so we expect reward of 1.0
        self.assertAlmostEqual(0., reward)

    def test_task_step_correct_non_terminal_reward_optimal_behaviour_no_shaping(self):
        self.setUp()
        task = self.make_task(shaping_type=Shaping.STANDARD)
        initial_state_sim = self.get_initial_state_sim(task)
        _ = task.observe_first_state(initial_state_sim)
        final_state_sim = self.get_perfect_state_sim(task, time_terminal=False)
        sim = TransitioningSimStub(initial_state_sim, final_state_sim)

        state, reward, done, info = task.task_step(sim, self.dummy_action, 1)

        # aircraft maintained correct altitude (1.0) but sim is non-terminal
        # so we expect no distance traveled reward (0.0) average to 0.5
        self.assertAlmostEqual(0.0, reward)

    def test_task_step_correct_terminal_reward_optimal_behaviour_shaping(self):
        self.setUp()
        for shaping in (Shaping.STANDARD, Shaping.SEQUENTIAL_CONT):
            task = self.make_task(shaping_type=shaping)
            initial_state_sim = self.get_initial_state_sim(task)
            _ = task.observe_first_state(initial_state_sim)
            final_state_sim = self.get_perfect_state_sim(task, time_terminal=True)
            sim = TransitioningSimStub(initial_state_sim, final_state_sim)

            _, _, _, info = task.task_step(sim, self.dummy_action, 1)
            reward_obj: rewards.Reward = info['reward']

            # aircraft moved maximum distance on correct heading and maintained
            # altitude, so we expect non-shaping reward of 1.0
            self.assertAlmostEqual(0., reward_obj.assessment_reward())

    def test_task_step_terrible_altitude_others_perfect(self):
        self.setUp()
        task = self.make_task(shaping_type=Shaping.STANDARD)
        bad_altitude = sys.float_info.max
        sim = self.get_transitioning_sim(task,
                                         time_terminal=True,
                                         altitude_ft=bad_altitude,
                                         roll_rad=0.,
                                         track_error_deg=0.,
                                         sideslip_deg=0.)

        _, _, _, info = task.task_step(sim, self.dummy_action, 1)
        reward_obj: rewards.Reward = info['reward']

        # we went from our initial position to an optimal distance and terrible altitude
        expected_reward = (0. * 3 - 1.) / 4
        self.assertAlmostEqual(expected_reward, reward_obj.agent_reward())

    def test_task_step_middling_track_travel_terrible_altitude(self):
        self.setUp()
        task = self.make_task(shaping_type=Shaping.STANDARD)
        bad_altitude = sys.float_info.max
        middling_track = HeadingControlTask.TRACK_ERROR_SCALING_DEG
        sim = self.get_transitioning_sim(task,
                                         time_terminal=True,
                                         altitude_ft=bad_altitude,
                                         roll_rad=0.,
                                         track_error_deg=middling_track,
                                         sideslip_deg=0., )

        _, _, _, info = task.task_step(sim, self.dummy_action, 1)
        reward_obj: rewards.Reward = info['reward']

        # track middling (-.5), altitude terrible (-1), sidelip perfect (0.), roll perfect (0.)
        expected_reward = (-.5 + -1 + 0. + 0.) / 4
        self.assertAlmostEqual(expected_reward, reward_obj.agent_reward())

    def test_task_step_reward_middling_everything(self):
        self.setUp()
        task = self.make_task(shaping_type=Shaping.STANDARD)
        # we get 0.5 reward at scaling distance from target altitude
        middling_altitude = task.INITIAL_ALTITUDE_FT + task.ALTITUDE_SCALING_FT
        middling_track = HeadingControlTask.TRACK_ERROR_SCALING_DEG
        middling_roll = HeadingControlTask.ROLL_ERROR_SCALING_RAD
        middling_sideslip = HeadingControlTask.SIDESLIP_ERROR_SCALING_DEG
        sim = self.get_transitioning_sim(task,
                                         time_terminal=True,
                                         altitude_ft=middling_altitude,
                                         roll_rad=middling_roll,
                                         track_error_deg=middling_track,
                                         sideslip_deg=middling_sideslip)

        _, _, _, info = task.task_step(sim, self.dummy_action, 1)
        reward_obj: rewards.Reward = info['reward']

        # we went from our initial position to an optimal distance and terrible altitude
        expected_reward = (-0.5 * 4) / 4
        [self.assertAlmostEqual(-0.5, comp) for comp in reward_obj.base_reward_elements]
        self.assertAlmostEqual(expected_reward, reward_obj.agent_reward())

    def test_observe_first_state_correct_track_error(self):
        self.setUp()
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)

        track_deg = prp.Vector2.from_sim(sim).heading_deg()
        target_track_deg = sim[self.task.target_track_deg]
        error_deg = track_deg - target_track_deg
        expected_acute_error_deg = utils.reduce_reflex_angle_deg(error_deg)
        actual_error_deg = sim[self.task.track_error_deg]

        self.assertAlmostEqual(expected_acute_error_deg, actual_error_deg)


class TestTurnHeadingControlTask(TestHeadingControlTask):

    def setUp(self):
        super().setUp()
        assert isinstance(self.task, TurnHeadingControlTask)

    def get_class_under_test(self):
        return TurnHeadingControlTask

    def get_target_track(self, sim: SimStub):
        return sim[TurnHeadingControlTask.target_track_deg]

    def test_observe_first_state_creates_desired_heading_in_expected_range(self):
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)

        desired_heading = sim[HeadingControlTask.target_track_deg]
        self.assertGreaterEqual(desired_heading, 0)
        self.assertLessEqual(desired_heading, 360)

    def test_observe_first_state_changes_desired_heading(self):
        sim = SimStub.make_valid_state_stub(self.task)
        _ = self.task.observe_first_state(sim)
        desired_heading = sim[HeadingControlTask.target_track_deg]

        _ = self.task.observe_first_state(sim)
        new_desired_heading = sim[HeadingControlTask.target_track_deg]

        self.assertNotEqual(desired_heading, new_desired_heading)
