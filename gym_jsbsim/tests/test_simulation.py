import unittest
import jsbsim
import multiprocessing
import time
from gym_jsbsim.simulation import Simulation
from gym_jsbsim import aircraft
import gym_jsbsim.properties as prp


class TestSimulation(unittest.TestCase):
    sim: Simulation = None

    def setUp(self):
        if self.sim:
            self.sim.close()
        self.sim = Simulation()

    def tearDown(self):
        self.sim = None

    def test_init_jsbsim(self):
        self.assertIsInstance(self.sim.jsbsim, jsbsim.FGFDMExec,
                              msg=f'Expected Simulation.jsbsim to hold an '
                              'instance of JSBSim.')

    def test_load_model(self):
        plane = aircraft.a320
        self.sim = None
        self.sim = Simulation(aircraft=plane)
        actual_name = self.sim.get_loaded_model_name()

        self.assertEqual(plane.jsbsim_id, actual_name,
                         msg=f'Unexpected aircraft model name after loading.')

    def test_load_bad_aircraft_id(self):
        bad_name = 'qwertyuiop'
        bad_aircraft = aircraft.Aircraft(bad_name, '', '', 100.)

        with self.assertRaises(RuntimeError):
            self.sim = None
            self.sim = Simulation(aircraft=bad_aircraft)

    def test_get_property(self):
        self.setUp()
        # we expect certain values specified in the IC config XML file
        expected_values = {
            prp.initial_u_fps: 328.0,
            prp.initial_v_fps: 0.0,
            prp.initial_w_fps: 0.0,
            prp.u_fps: 328.0,
            prp.v_fps: 0.0,
            prp.w_fps: 0.0,
        }

        for prop, expected in expected_values.items():
            actual = self.sim[prop]
            self.assertAlmostEqual(expected, actual)

    def test_get_bad_property(self):
        self.setUp()
        bad_prop = prp.BoundedProperty("bad_prop_name", "", 0, 0)
        with self.assertRaises(KeyError):
            _ = self.sim[bad_prop]

    def test_set_property(self):
        self.setUp()
        set_values = {
            prp.altitude_sl_ft: 1000,
            prp.aileron_cmd: 0.2,
            prp.elevator_cmd: 0.2,
            prp.rudder_cmd: 0.2,
            prp.throttle_cmd: 0.2,
        }

        for prop, value in set_values.items():
            self.sim[prop] = value

        for prop, expected in set_values.items():
            actual = self.sim[prop]
            self.assertAlmostEqual(expected, actual)

    def test_initialise_conditions_basic_config(self):
        plane = aircraft.f15

        # manually reset JSBSim instance with new initial conditions
        if self.sim:
            self.sim.close()
        sim_frequency = 2
        self.sim = Simulation(sim_frequency_hz=sim_frequency, aircraft=plane, init_conditions=None)

        self.assertEqual(self.sim.get_loaded_model_name(), plane.jsbsim_id,
                         msg='JSBSim did not load expected aircraft model: ' +
                         self.sim.get_loaded_model_name())

        # check that properties are as we expected them to be
        expected_values = {
            prp.initial_u_fps: 328.0,
            prp.initial_v_fps: 0.0,
            prp.initial_w_fps: 0.0,
            prp.u_fps: 328.0,
            prp.v_fps: 0.0,
            prp.w_fps: 0.0,
            prp.BoundedProperty('simulation/dt', '', None, None): 1 / sim_frequency
        }

        for prop, expected in expected_values.items():
            actual = self.sim[prop]
            self.assertAlmostEqual(expected, actual)

    def test_initialise_conditions_custom_config(self):
        """ Test JSBSimInstance initialisation with custom initial conditions. """

        plane = aircraft.f15
        init_conditions = {
            prp.initial_u_fps: 1000.0,
            prp.initial_v_fps: 0.0,
            prp.initial_w_fps: 1.0,
            prp.initial_altitude_ft: 5000,
            prp.initial_heading_deg: 12,
            prp.initial_r_radps: -0.1,
        }
        # map JSBSim initial condition properties to sim properties
        init_to_sim_conditions = {
            prp.initial_u_fps: prp.u_fps,
            prp.initial_v_fps: prp.v_fps,
            prp.initial_w_fps: prp.w_fps,
            prp.initial_altitude_ft: prp.altitude_sl_ft,
            prp.initial_heading_deg: prp.heading_deg,
            prp.initial_r_radps: prp.r_radps,
        }
        sim_frequency = 10

        # manually reset JSBSim instance
        if self.sim:
            self.sim.close()
        self.sim = Simulation(sim_frequency, plane, init_conditions)

        # check JSBSim initial condition and simulation properties
        for init_prop, expected in init_conditions.items():
            sim_prop = init_to_sim_conditions[init_prop]

            init_actual = self.sim[init_prop]
            sim_actual = self.sim[sim_prop]
            self.assertAlmostEqual(expected, init_actual,
                                   msg=f'wrong value for property {init_prop}')
            self.assertAlmostEqual(expected, sim_actual,
                                   msg=f'wrong value for property {sim_prop}')

        self.assertAlmostEqual(1.0 / sim_frequency, self.sim[prp.sim_dt])

    def test_multiprocess_simulations(self):
        """
        JSBSim segfaults when multiple instances are run on one process.

        Let's confirm that we can launch multiple processes each with 1 instance.
        """
        processes = 4
        with multiprocessing.Pool(processes) as pool:
            # N.B. basic_task is a top level function that inits JSBSim
            future_results = [pool.apply_async(basic_task) for _ in range(processes)]
            results = [f.get() for f in future_results]

        good_exit_code = 0
        expected = [good_exit_code] * processes
        self.assertListEqual(results, expected,
                             msg="multiprocess execution of JSBSim failed")


def basic_task():
    """ A simple task involving initing a JSBSimInstance to test multiprocessing. """
    model = aircraft.cessna172P
    time.sleep(0.05)
    fdm = Simulation(aircraft=model)
    fdm.run()
    time.sleep(0.05)

    return 0


if __name__ == '__main__':
    unittest.main()
