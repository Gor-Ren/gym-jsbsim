import jsbsim
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # req'd for 3d plotting
from typing import NamedTuple, Dict, Union


class AxesTuple(NamedTuple):
    """ Holds references to figure subplots (axes) """
    axes_state: Axes3D
    axes_stick: plt.Axes
    axes_throttle: plt.Axes
    axes_rudder: plt.Axes


class JsbSimInstance(object):
    """
    A class which wraps an instance of JSBSim and manages communication with it.
    """
    #AxesTuple = namedtuple('AxesTuple',
    #                      ['axes_state', 'axes_stick', 'axes_throttle', 'axes_rudder'])

    encoding = 'utf-8'  # encoding of bytes returned by JSBSim Cython funcs
    properties = None
    props_to_plot: Dict = dict(x=dict(name='position/lat-gc-deg', label='geocentric latitude [deg]'),
                               y=dict(name='position/long-gc-deg', label='geocentric longitude [deg]'),
                               z=dict(name='position/h-sl-ft', label='altitude above MSL [ft]'),
                               u=dict(name='velocities/v-north-fps', label='velocity true north [ft/s]'),
                               v=dict(name='velocities/v-east-fps', label='velocity east [ft/s]'),
                               w=dict(name='velocities/v-down-fps', label='velocity downwards [ft/s]'),
                               ail=dict(name='fcs/left-aileron-pos-norm', label='left aileron position, [-]'),
                               ele=dict(name='fcs/elevator-pos-norm', label='elevator position, [-]'),
                               thr=dict(name='fcs/throttle-pos-norm', label='throttle position, [-]'),
                               rud=dict(name='fcs/rudder-pos-norm', label='rudder position, [-]'))
    FT_PER_DEG_LAT: int = 365228
    ft_per_deg_lon: int = None  # calc at reset(), depends on longitude
    figure: plt.Figure = None
    axes: AxesTuple = None
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
        :param init_conditions: dict mapping properties to their initial values.
            Defaults to None, causing a default set of initial props to be used.
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
        Creates or updates a 3D plot of the episode.
        """
        if not self.figure:
            self.figure, self.axes = self._plot_configure()

        self._plot_state(self.axes, self.props_to_plot)

    def _plot_configure(self):
        """
        Creates a figure with subplots for states and actions.

        TODO: return params (refs to fig and its axes)
        """
        plt.ion()  # interactive mode allows dynamic updating of plot
        figure = plt.figure(figsize=(6, 12))

        spec = plt.GridSpec(nrows=3,
                            ncols=2,
                            width_ratios=[5, 1],  # second column very thin
                            height_ratios=[6, 5, 1],  # bottom row very short
                            wspace=0.3)

        # create subplots
        axes_state: Axes3D = figure.add_subplot(spec[0, 0:], projection='3d')
        axes_stick = figure.add_subplot(spec[1, 0])
        axes_throttle = figure.add_subplot(spec[1, 1])
        axes_rudder = figure.add_subplot(spec[2, 0])

        # config subplot for state
        axes_state.set_xlabel(self.props_to_plot['x']['label'])
        axes_state.set_ylabel(self.props_to_plot['y']['label'])
        axes_state.set_zlabel(self.props_to_plot['z']['label'])

        # config subplot for stick (aileron and elevator control in x/y axes)
        axes_stick.set_xlabel('ailerons [-]', )
        axes_stick.set_ylabel('elevator [-]')
        # set limits
        axes_stick.set_xlim(left=-1, right=1)
        axes_stick.set_ylim(bottom=-1, top=1)
        # move axis labels outside subplot
        axes_stick.xaxis.set_label_coords(0.5, 1.08)
        axes_stick.yaxis.set_label_coords(-0.05, 0.5)
        # make axes cross at origin
        axes_stick.spines['left'].set_position('zero')
        axes_stick.spines['bottom'].set_position('zero')
        # only show ticks at extremes of range
        axes_stick.set_xticks([-1, 1])
        axes_stick.xaxis.set_ticks_position('bottom')
        axes_stick.set_yticks([-1, 1])
        axes_stick.yaxis.set_ticks_position('left')
        axes_stick.tick_params(which='both', direction='inout')
        # show minor ticks throughout
        minor_locator = plt.MultipleLocator(0.2)
        axes_stick.xaxis.set_minor_locator(minor_locator)
        axes_stick.yaxis.set_minor_locator(minor_locator)
        # hide unneeded spines
        axes_stick.spines['right'].set_visible(False)
        axes_stick.spines['top'].set_visible(False)

        # config subplot for throttle: a 1D vertical plot
        axes_throttle.set_ylabel('throttle [-]')
        axes_throttle.set_ylim(bottom=0, top=1)
        axes_throttle.yaxis.set_label_coords(0.5, 0.5)
        axes_throttle.set_yticks([0, 0.5, 1])
        axes_throttle.yaxis.set_minor_locator(minor_locator)
        axes_throttle.tick_params(axis='y', which='both', direction='inout')
        # hide horizontal x-axis and related spines
        axes_throttle.xaxis.set_visible(False)
        for spine in ['right', 'bottom', 'top']:
            axes_throttle.spines[spine].set_visible(False)

        # config rudder subplot: 1D horizontal plot
        axes_rudder.set_xlabel('rudder [-]')
        axes_rudder.set_xlim(left=-1, right=1)
        axes_rudder.xaxis.set_label_coords(0.5, -0.5)
        axes_rudder.set_xticks([-1, 0, 1])
        axes_rudder.xaxis.set_minor_locator(minor_locator)
        axes_rudder.tick_params(axis='x', which='both', direction='inout')
        axes_rudder.get_yaxis().set_visible(False)  # only want a 1D subplot
        for spine in ['left', 'right', 'top']:
            axes_rudder.spines[spine].set_visible(False)

        all_axes = AxesTuple(axes_state=axes_state,
                             axes_stick=axes_stick,
                             axes_throttle=axes_throttle,
                             axes_rudder=axes_rudder)

        plt.show()
        plt.pause(0.001)  # voodoo pause needed for figure to appear

        return figure, all_axes

    def _plot_state(self, all_axes: AxesTuple, props: Dict):
        """
        Plots the state of the simulation on input axes.

        State is given by three translational coords (x, y, z) and three
        linear velocities (v_x, v_y, v_z).

        The dict 'props' provides a mapping of these variable names to a
        dict specifying their 'name', the property to be retrieved from
        JSBSim.

        :param figure: plt.Axes, the axes (subplot) to plot on
        :param props: dict, mapping strs x, y, z, v_x, v_y, v_z to a dict
            containing a 'name' field for the property to be retrieved from
            JSBSim
        """
        x, y, z = [self.sim[props[var]['name']] for var in 'xyz']
        u, v, w = [self.sim[props[var]['name']] for var in 'uvw']
        control_surfaces = ['ail', 'ele', 'thr', 'rud']
        ail, ele, thr, rud = [self.sim[props[var]['name']] for var in control_surfaces]
        # get velocity vector coords using scaled velocity
        x2 = x + u / self.FT_PER_DEG_LAT
        y2 = y + v / self.ft_per_deg_lon
        z2 = z - z    # negative because v_z is positive down

        # plot aircraft position and velocity
        all_axes.axes_state.scatter([x], [y], zs=[z], c='k', s=10)
        if self.velocity_arrow:
            # get rid of previous timestep velocity arrow
            self.velocity_arrow.pop().remove()
        # scale velocity and get coords for arrow
        x2 = x + v_x / self.FT_PER_DEG_LAT
        y2 = y + v_y / self.ft_per_deg_lon
        z2 = z - v_z    # negative because v_z is positive down
        self.velocity_arrow = axes.plot([x, x2], [y, y2], [z, z2], 'r-')
        plt.pause(0.001)


