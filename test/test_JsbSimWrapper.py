import unittest
import jsbsim
from wrapper import JsbSimInstance

class TestJsbSimWrapper(unittest.TestCase):

    def setUp(self):
        self.sim = None  # make sure any old sim instance is deallocated; can only run 1 JSBSim instance in a process
        self.sim = JsbSimInstance()

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



if __name__ == '__main__':
    unittest.main()
