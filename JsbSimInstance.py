import jsbsim
import os


class JsbSimInstance(object):
    """
    A class which wraps an instance of JSBSim and manages communication with it.
    """
    encoding = 'utf-8'  # encoding of bytes returned by JSBSim Cython funcs

    def __init__(self):
        root_dir = os.path.abspath("/home/gordon/Apps/jsbsim-code")
        self.sim = jsbsim.FGFDMExec(root_dir=root_dir)

    def __getitem__(self, key: str):
        """
        Retrieves specified simulation property.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        :param key: string, the property to be retrieved
        :return: object?, property value
        """
        return self.sim[key]

    def __setitem__(self, key: str, value):
        """
        Sets simulation property to specified value.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        :param key: string, the property to be retrieved
        :param value: object?, the value to be set
        """
        self.sim[key] = value

    def load_model(self, model_name: str) -> None:
        """
        Loads the specified aircraft config into the simulation.

        The root JSBSim directory aircraft folder is searched for the aircraft
        XML config file.

        :param model_name: string, the aircraft name
        """
        load_success = self.sim.load_model(model=model_name)

        if not load_success:
            raise RuntimeError('JSBSim could not find specified model_name model: '
                               + model_name)

    def get_model_name(self) -> str:
        """
        Gets the name of the aircraft model currently loaded in JSBSim.

        :return: string, the name of the aircraft model if one is loaded, or
            None if no model is loaded.
        """
        name: str = self.sim.get_model_name().decode(self.encoding)
        if name:
            return name
        else:
            # name is empty string, no model is loaded
            return None

    def initialise(self, model_name: str) -> None:
        """
        Loads an aircraft and initialises simulation conditions.

        Initial conditions are specified in an XML config file. It is intended
        that a dummy config file will be used, and then new conditions
        specified programmatically.

        TODO: investigate whether loading an IC config and calling RunIC() is
        strictly necessary.
        """
        ic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basic_ic.xml')
        self.sim.load_ic(ic_path, useStoredPath=False)
        self.load_model(model_name)
        success = self.sim.run_ic()

        if not success:
            raise RuntimeError('JSBSim failed to launch with initial conditions.')
