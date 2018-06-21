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
        self.assertTrue(False, msg='implement this test!')

    def test_initialise_conditions(self):
        self.setUp()
        aircraft = 'c172x'
        self.sim.initialise(model_name=aircraft)

        self.assertEqual(self.sim.get_model_name(), aircraft,
                         msg='JSBSim did not load expected aircraft model: ' +
                         self.sim.get_model_name())

        # check that properties are as we expected them to be
        self.assertTrue(False, msg='implement this test!')


if __name__ == '__main__':
    unittest.main()
