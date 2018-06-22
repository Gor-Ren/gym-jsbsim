import unittest
import jsbsim
from JsbSimInstance import JsbSimInstance


class TestJsbSimWrapper(unittest.TestCase):

    def setUp(self):
        self.sim: JsbSimInstance = None  # make sure any old sim instance is deallocated
        self.sim = JsbSimInstance()

    def init_model(self, aircraft='c172x'):
        """
        Initialises a fresh JSBSim instance with an aircraft loaded.
        :param aircraft:
        :return:
        """
        self.setUp()
        self.sim.initialise(model_name=aircraft)

    def tearDown(self):
        self.sim = None

    def test_init_jsbsim(self):
        self.assertIsInstance(self.sim.sim, jsbsim.FGFDMExec,
                              msg=f'Expected JsbSimInstance.sim to hold an '
                              'instance of JSBSim.')

    def test_load_model(self):
        # make fresh sim instance
        self.setUp()

        # we expect simulation to init with no aircraft loaded
        self.assertIsNone(self.sim.get_model_name())

        # load an "X15" plane
        model_name = 'X15'
        self.sim.load_model(model_name)
        actual_name = self.sim.get_model_name()

        self.assertEqual(model_name, actual_name,
                         msg=f'Unexpected aircraft model name after loading.')

    def test_load_bad_aircraft_name(self):
        bad_name = 'qwertyuiop'
        with self.assertRaises(RuntimeError):
            self.sim.load_model(bad_name)

    def test_get_property(self):
        self.init_model()
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

    def test_set_property(self):
        self.init_model()
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
        self.setUp()
        aircraft = 'c172x'
        self.sim.initialise(model_name=aircraft, init_conditions=None)

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

        self.setUp()
        self.sim.initialise(model_name=aircraft, init_conditions=init_conditions)

        # check JSBSim initial condition and simulation properties
        for init_prop, expected in init_conditions.items():
            sim_prop = init_to_sim_conditions[init_prop]

            init_actual = self.sim[init_prop]
            sim_actual = self.sim[sim_prop]
            self.assertAlmostEqual(expected, init_actual,
                                   msg=f'wrong value for property {init_prop}')
            self.assertAlmostEqual(expected, sim_actual,
                                   msg=f'wrong value for property {sim_prop}')


if __name__ == '__main__':
    unittest.main()
