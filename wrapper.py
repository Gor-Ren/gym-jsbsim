import jsbsim
import os

class JsbSimWrapper(object):
    """
    A class which wraps an instance of JSBSim and manages communication with it.
    """
    encoding = 'utf-8'

    def __init__(self):
        root_dir = os.path.abspath("/home/gordon/Apps/jsbsim-code")
        self.sim = jsbsim.FGFDMExec(root_dir=root_dir)

    def load_model(self, aircraft: str) -> None:
        """
        Loads the specified aircraft config into the simulation.

        The root JSBSim directory aircraft folder is searched for the aircraft
        XML config file.

        :param aircraft: string, the aircraft name
        """
        self.sim.load_model(model=aircraft)

    def get_model_name(self) -> str:
        """ Gets the name of the aircraft model currently loaded in JSBSim.

        :return: string, the name of the aircraft model if one is loaded, or
            None if no model is loaded.
        """
        name: str = self.sim.get_model_name().decode(self.encoding)
        if name:
            return name
        else:
            # name is empty string, no model is loaded
            return None