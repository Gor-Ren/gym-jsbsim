import unittest
import jsbsim
from wrapper import JsbSimWrapper

class TestJsbSimWrapper(unittest.TestCase):

    def test_init_jsbsim(self):
        wrapped_sim = JsbSimWrapper()
        self.assertIsInstance(wrapped_sim.sim, jsbsim.FGFDMExec,
                              msg=f'Expected JsbSimWrapper.sim to hold an '
                              'instance of JSBSim.')

    def test_load_aircraft(self):
        wrapped_sim = JsbSimWrapper()
        # we expect simulation to init with no aircraft loaded
        tmp = wrapped_sim.get_model_name()
        self.assertIsNone(wrapped_sim.get_model_name())

        # load an "X15" plane
        model_name = 'X15'
        wrapped_sim.load_model(model_name)
        actual_name = wrapped_sim.get_model_name()

        self.assertEqual(model_name, actual_name,
                         msg=f'Unexpected aircraft model name after loading.')


if __name__ == '__main__':
    unittest.main()
