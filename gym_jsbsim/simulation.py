import jsbsim
import os
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # req'd for 3d plotting
from typing import Dict, Union


class Simulation(object):
    """
    A class which wraps an instance of JSBSim and manages communication with it.
    """
    encoding = 'utf-8'  # encoding of bytes returned by JSBSim Cython funcs
    ROOT_DIR = os.path.abspath('/home/gordon/apps/jsbsim')
    OUTPUT_FILE = 'flightgear.xml'

    def __init__(self,
                 sim_dt: float= 1.0 / 120.0,
                 aircraft_model_name: str='c172p',
                 init_conditions: Dict[str, Union[int, float]]=None):
        """
        Constructor. Creates an instance of JSBSim and sets initial conditions.

        :param sim_dt: float, the JSBSim integration timestep in seconds. Defaults
            to 1/120, i.e. 120 Hz
        :param aircraft_model_name: string, name of aircraft to be loaded.
            JSBSim looks for file \model_name\model_name.xml from root dir.
        :param init_conditions: dict mapping properties to their initial values.
            Defaults to None, causing a default set of initial props to be used.
        """
        self.sim = jsbsim.FGFDMExec(root_dir=self.ROOT_DIR)
        self.sim.set_debug_level(0)
        self.properties = None
        output_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.OUTPUT_FILE)
        self.sim.set_output_directive(output_config_path)
        self.sim_dt = sim_dt
        self.initialise(self.sim_dt, aircraft_model_name, init_conditions)
        self.sim.disable_output()
        self.wall_clock_dt = None

    def __getitem__(self, key: str):
        """
        Retrieves specified simulation property.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        :param key: string, the property to be retrieved
        :return: object?, property value
        :raises KeyError: if key is not a valid parameter
        """

        if key in self.properties:
            # TODO: can remove guard once JSBSim updated; JSBSim will check this
            #   alternatively leave in, and bypass JSBSim check by using .get_property_value()
            return self.sim[key]
        else:
            raise KeyError('property not found:' + key)

    def __setitem__(self, key: str, value) -> None:
        """
        Sets simulation property to specified value.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        Warning: JSBSim will create new properties if the specified one exists.
        If the property you are setting is read-only in JSBSim the operation
        will silently fail.

        :param key: string, the property to be retrieved
        :param value: object?, the value to be set
        :raises KeyError: if key is not a valid parameter
        """
        if key in self.properties:
            self.sim[key] = value
        else:
            raise KeyError('property not found: ' + key)

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

    def get_sim_time(self) -> float:
        """ Gets the simulation time from JSBSim, a float. """
        return self.sim['simulation/sim-time-sec']

    def initialise(self, dt: float, model_name: str,
                   init_conditions: Dict[str, Union[int, float]]=None) -> None:
        """
        Loads an aircraft and initialises simulation conditions.

        JSBSim creates an InitialConditions object internally when given an
        XML config file. This method either loads a basic set of ICs, or
        can be passed a dictionary with ICs. In the latter case a minimal IC
        XML file is loaded, and then the dictionary values are fed in.

        This method sets the self.properties set of valid property names.

        :param dt: float, the JSBSim integration timestep in seconds
        :param model_name: string, name of aircraft to be loaded
        :param init_conditions: dict mapping properties to their initial values
        """
        if init_conditions is not None:
            # if we are specifying conditions, load a minimal file
            ic_file = 'minimal_ic.xml'
        else:
            ic_file = 'basic_ic.xml'

        ic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ic_file)
        self.sim.load_ic(ic_path, useStoredPath=False)
        self.load_model(model_name)
        self.sim.set_dt(dt)
        # extract set of legal property names for this aircraft
        # TODO: can remove the .split(" ")[0] once JSBSim bug has been fixed (in progress)
        self.properties = set([prop.split(" ")[0] for prop in self.sim.query_property_catalog('')])

        # now that IC object is created in JSBSim, specify own conditions
        if init_conditions:
            for prop, value in init_conditions.items():
                self[prop] = value

        success = self.sim.run_ic()
        if not success:
            raise RuntimeError('JSBSim failed to init simulation conditions.')

    def run(self) -> bool:
        """
        Runs a single timestep in the JSBSim simulation.

        JSBSim monitors the simulation and detects whether it thinks it should
        end, e.g. because a simulation time was specified. False is returned
        if JSBSim termination criteria are met.

        :return: bool, False if sim has met JSBSim termination criteria else True.
        """
        result = self.sim.run()
        if self.wall_clock_dt is not None:
            time.sleep(self.wall_clock_dt)
        return result

    def enable_flightgear_output(self):
        self.sim.enable_output()

    def disable_flightgear_output(self):
        self.sim.disable_output()

    def close(self):
        """ Closes the simulation and any plots. """
        if self.sim:
            self.sim = None

    def set_simulation_time_factor(self, time_factor):
        """ Specifies a factor, relative to realtime, for simulation to run at.

        The simulation runs at realtime for time_factor = 1. It runs at double
        speed for time_factor=2, and half speed for 0.5.

        :param time_factor: int or float, nonzero, sim speed relative to realtime
        """
        if time_factor is not None:
            self.wall_clock_dt = self.sim_dt / time_factor
        else:
            self.wall_clock_dt = None

    def start_engines(self):
        """ Sets all engines running. """
        for engine_no in range(self.sim.propulsion_get_num_engines()):
            self.sim.propulsion_init_running(engine_no)
