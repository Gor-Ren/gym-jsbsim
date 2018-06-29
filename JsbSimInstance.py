import jsbsim
import os
import math
import matplotlib.pyplot as plt
from typing import Dict, Union


class JsbSimInstance(object):
    """
    A class which wraps an instance of JSBSim and manages communication with it.
    """
    encoding = 'utf-8'  # encoding of bytes returned by JSBSim Cython funcs
    properties = None
    props_to_plot: Dict = dict(x=dict(name='position/lat-gc-deg', label='geocentric latitude [deg]'),
                               y=dict(name='position/long-gc-deg', label='geocentric longitude [deg]'),
                               z=dict(name='position/h-sl-ft', label='altitude above MSL [ft]'),
                               v_x=dict(name='velocities/v-north-fps', label='velocity true north [ft/s]'),
                               v_y=dict(name='velocities/v-east-fps', label='velocity east [ft/s]'),
                               v_z=dict(name='velocities/v-down-fps', label='velocity downwards [ft/s]'))
    FT_PER_DEG_LAT: int = 365228
    ft_per_deg_lon: int = None  # calc at reset(), depends on longitude
    figure: plt.Figure = None
    velocity_arrow = None

    def __init__(self,
                 dt: float=1.0/120.0,
                 aircraft_model_name: str='c172x',
                 init_conditions: Dict[str, Union[int, float]]=None):
        """
        Constructor. Creates an instance of JSBSim and sets initial conditions.

        :param dt: float, the JSBSim integration timestep in seconds. Defaults
            to 1/120, i.e. 120 Hz
        :param aircraft_model_name: string, name of aircraft to be loaded.
            JSBSim looks for file \model_name\model_name.xml in root dir.
        :param init_conditions: dict mapping properties to their initial values
        """
        root_dir = os.path.abspath("/home/gordon/apps/jsbsim")
        self.sim = jsbsim.FGFDMExec(root_dir=root_dir)
        self.initialise(dt, aircraft_model_name, init_conditions)

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

    def initialise(self, dt: float, model_name: str,
                   init_conditions: Dict[str, Union[int, float]]=None) -> None:
        """
        Loads an aircraft and initialises simulation conditions.

        JSBSim creates an InitialConditions object internally when given an
        XML config file. This method either loads a basic set of ICs, or
        can be passed a dictionary with ICs. In the latter case a minimal IC
        XML file is loaded, and then the dictionary values are fed in.

        This method sets the self.properties set of valid property names.

        TODO: investigate whether we can specify ICs without loading XML

        :param dt: float, the JSBSim integration timestep in seconds
        :param model_name: string, name of aircraft to be loaded
        :param init_conditions: dict mapping properties to their initial values
        """
        if init_conditions:
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

        # ft per deg. longitude is distance at equator * cos(lon)
        # attribution: https://www.colorado.edu/geography/gcraft/warmup/aquifer/html/distance.html
        lon = self.sim[self.props_to_plot['y']['name']]
        self.ft_per_deg_lon = self.FT_PER_DEG_LAT * math.cos(math.radians(lon))

    def run(self) -> bool:
        """
        Runs a single timestep in the JSBSim simulation.

        JSBSim monitors the simulation and detects whether it thinks it should
        end, e.g. because a simulation time was specified. False is returned
        if JSBSim termination criteria are met.

        :return: bool, False if sim has met JSBSim termination criteria else True.
        """
        return self.sim.run()

    def close(self):
        """ Closes the simulation and any plots. """
        if self.sim:
            self.sim = None
        if self.figure:
            plt.close(self.figure)
            self.figure = None

    def plot(self) -> None:
        """
        Creates or updates a 3D plot of the episode aircraft trajectory.
        """
        STATE_VARS_TO_PLOT = ('x', 'y', 'z', 'v_x', 'v_y', 'v_z')
        x, y, z, v_x, v_y, v_z = [self.sim[self.props_to_plot[var]['name']] for var in STATE_VARS_TO_PLOT]

        if not self.figure:
            plt.ion()  # interactive mode allows dynamic updating of plot
            self.figure = plt.figure()
            self.figure.add_subplot(1, 1, 1, projection='3d')
            self.figure.gca().set_xlabel(self.props_to_plot['x']['label'])
            self.figure.gca().set_ylabel(self.props_to_plot['y']['label'])
            self.figure.gca().set_zlabel(self.props_to_plot['z']['label'])
            plt.show()
            plt.pause(0.001)  # voodoo pause needed for figure to appear

        ax = self.figure.gca()
        if self.velocity_arrow:
            # get rid of previous timestep velocity arrow
            self.velocity_arrow.pop().remove()
        # get coords from scaled velocities for drawing velocity line
        x2 = x + v_x / self.FT_PER_DEG_LAT
        y2 = y + v_y / self.ft_per_deg_lon
        z2 = z - v_z  # negative because v_z is positive down
        self.velocity_arrow = ax.plot([x, x2], [y, y2], [z, z2], 'r-')
        # draw trajectory point
        ax.scatter([x], [y], zs=[z], c='k', s=10)
        plt.pause(0.001)
