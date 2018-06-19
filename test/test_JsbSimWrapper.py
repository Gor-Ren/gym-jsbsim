import unittest
import jsbsim
from wrapper import JsbSimWrapper

class TestJsbSimWrapper(unittest.TestCase):

    def test_init_jsbsim(self):
        wrapped_sim = JsbSimWrapper()
        self.assertIsInstance(wrapped_sim.sim, jsbsim.FGFDMExec,
                              msg=f"Expected JsbSimWrapper.sim to hold an "
                              f"instance of JSBSim. Actual: {type(sim.sim)}")


if __name__ == '__main__':
    unittest.main()
