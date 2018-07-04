import unittest
import jsbsim
import multiprocessing
import time
from simulation import Simulation


class TestJsbSimWrapper(unittest.TestCase):
    sim: Simulation = None

    def setUp(self):
        if self.sim:
            self.sim.close()
        self.sim = Simulation()

    def tearDown(self):
        self.sim = None

    def test_init_jsbsim(self):
        self.assertIsInstance(self.sim.sim, jsbsim.FGFDMExec,
                              msg=f'Expected Simulation.sim to hold an '
                              'instance of JSBSim.')

    def test_load_model(self):
        # make fresh sim instance with "X15" plane
        model_name = 'X15'
        self.sim = None
        self.sim = Simulation(aircraft_model_name=model_name)
        actual_name = self.sim.get_model_name()

        self.assertEqual(model_name, actual_name,
                         msg=f'Unexpected aircraft model name after loading.')

    def test_load_bad_aircraft_name(self):
        bad_name = 'qwertyuiop'

        with self.assertRaises(RuntimeError):
            self.sim = None
            self.sim = Simulation(aircraft_model_name=bad_name)

    def test_get_property(self):
        self.setUp()
        # we expect certain values specified in the IC config XML file
        expected_values = {
            'ic/u-fps': 328.0,
            'ic/v-fps': 0.0,
            'ic/w-fps': 0.0,
            'velocities/u-fps': 328.0,
            'velocities/v-fps': 0.0,
            'velocities/w-fps': 0.0,
        }

        for prop, expected in expected_values.items():
            actual = self.sim[prop]
            self.assertAlmostEqual(expected, actual)

    def test_get_bad_property(self):
        self.setUp()
        bad_prop = 'bad'
        with self.assertRaises(KeyError):
            _ = self.sim[bad_prop]

    def test_set_property(self):
        self.setUp()
        set_values = {
            'ic/u-fps': 200.0,
            'ic/v-fps': 5.0,
            'ic/w-fps': 5,
            'position/h-sl-meters': 1000,
            'fcs/aileron-cmd-norm': 0.2,
            'fcs/elevator-cmd-norm': 0.2,
            'fcs/rudder-cmd-norm': 0.2,
            'fcs/throttle-cmd-norm': 0.2,
        }

        for prop, value in set_values.items():
            self.sim[prop] = value

        for prop, expected in set_values.items():
            actual = self.sim[prop]
            self.assertAlmostEqual(expected, actual)

    def test_initialise_conditions_basic_config(self):
        aircraft = '737'

        # manually reset JSBSim instance with new initial conditions
        if self.sim:
            self.sim.close()
        self.sim = Simulation(dt=0.5, aircraft_model_name=aircraft, init_conditions=None)

        self.assertEqual(self.sim.get_model_name(), aircraft,
                         msg='JSBSim did not load expected aircraft model: ' +
                         self.sim.get_model_name())

        # check that properties are as we expected them to be
        expected_values = {
            'ic/u-fps': 328.0,
            'ic/v-fps': 0.0,
            'ic/w-fps': 0.0,
            'velocities/u-fps': 328.0,
            'velocities/v-fps': 0.0,
            'velocities/w-fps': 0.0,
            'simulation/dt': 0.5
        }

        for prop, expected in expected_values.items():
            actual = self.sim[prop]
            self.assertAlmostEqual(expected, actual)

    def test_initialise_conditions_custom_config(self):
        """ Test JSBSimInstance initialisation with custom initial conditions. """

        aircraft = 'f15'
        init_conditions = {
            'ic/u-fps': 1000.0,
            'ic/v-fps': 0.0,
            'ic/w-fps': 1.0,
            'ic/h-sl-ft': 5000,
            'ic/phi-deg': 12,
            'ic/theta-deg': -5,
            'ic/psi-true-deg': 45,
        }
        # map JSBSim initial condition properties to sim properties
        init_to_sim_conditions = {
            'ic/u-fps': 'velocities/u-fps',
            'ic/v-fps': 'velocities/v-fps',
            'ic/w-fps': 'velocities/w-fps',
            'ic/h-sl-ft': 'position/h-sl-ft',
            'ic/phi-deg': 'attitude/phi-deg',
            'ic/theta-deg': 'attitude/theta-deg',
            'ic/psi-true-deg': 'attitude/psi-deg',
        }
        dt = 0.1

        # manually reset JSBSim instance
        if self.sim:
            self.sim.close()
        self.sim = Simulation(dt, aircraft, init_conditions)

        # check JSBSim initial condition and simulation properties
        for init_prop, expected in init_conditions.items():
            sim_prop = init_to_sim_conditions[init_prop]

            init_actual = self.sim[init_prop]
            sim_actual = self.sim[sim_prop]
            self.assertAlmostEqual(expected, init_actual,
                                   msg=f'wrong value for property {init_prop}')
            self.assertAlmostEqual(expected, sim_actual,
                                   msg=f'wrong value for property {sim_prop}')

        self.assertAlmostEqual(dt, self.sim['simulation/dt'])

    def test_multiprocess_simulations(self):
        """ JSBSim segfaults when multiple instances are run on one process.

        Lets confirm that we can launch multiple processes each with 1 instance.
        """
        processes = 4
        with multiprocessing.Pool(processes) as pool:
            # N.B. basic_task is a top level function that inits JSBSim
            future_results = [pool.apply_async(basic_task) for _ in range(processes)]
            results = [f.get() for f in future_results]

        expected = [0] * processes  # each process should return 0
        self.assertListEqual(results, expected,
                             msg="multiprocess execution of JSBSim did not work")


def basic_task():
    """ A simple task involving initing a JSBSimInstance to test multiprocessing. """
    time.sleep(0.05)
    fdm = Simulation(aircraft_model_name='c172x')
    fdm.run()
    time.sleep(0.05)

    return 0


if __name__ == '__main__':
    unittest.main()
