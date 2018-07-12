import gym
import math
import subprocess
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # req'd for 3d plotting
from gym_jsbsim.simulation import Simulation
from typing import NamedTuple, Dict


class AxesTuple(NamedTuple):
    """ Holds references to figure subplots (axes) """
    axes_state: Axes3D
    axes_stick: plt.Axes
    axes_throttle: plt.Axes
    axes_rudder: plt.Axes


class FigureVisualiser(object):
    """ Class for manging a matplotlib Figure displaying agent state and actions """
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
    ft_per_deg_lon: int = None  # calc at reset() - it depends on the longitude value
    PLOT_PAUSE_SECONDS = 0.0001

    def __init__(self, sim: Simulation, is_plot_position=True):
        """
        Constructor.

        The main attribute set here is ft_per_deg_lon, which depends
        dynamically on the aircraft's longitude (because of the conversion
        between geographic and Euclidean coordinate systems). We retrieve
        longitude from the simulation and assume it is constant thereafter.

        :param sim: Simulation that will be plotted
        :param is_plot_position: aircraft position and velocity is plotted if True
        """
        self.is_plot_position = is_plot_position
        self.figure: plt.Figure = None
        self.axes: AxesTuple = None
        self.velocity_arrow = None

        # ft per deg. longitude is distance at equator * cos(lon)
        # attribution: https://www.colorado.edu/geography/gcraft/warmup/aquifer/html/distance.html
        lon = sim[self.props_to_plot['y']['name']]
        self.ft_per_deg_lon = self.FT_PER_DEG_LAT * math.cos(math.radians(lon))

    def plot(self, sim: Simulation) -> None:
        """
        Creates or updates a 3D plot of the episode.

        :param sim: Simulation that will be plotted
        """
        if not self.figure:
            self.figure, self.axes = self._plot_configure()

        # delete old control surface data points
        for subplot in self.axes[1:]:
            # pop and remove all data points
            while subplot.lines:
                data = subplot.lines.pop()
                del data

        if self.is_plot_position:
            self._plot_position_state(sim, self.axes)
        self._plot_control_states(sim, self.axes)
        self._plot_control_commands(sim, self.axes)
        plt.pause(self.PLOT_PAUSE_SECONDS)  # voodoo pause needed for figure to update

    def close(self):
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.axes = None
            self.velocity_arrow = None

    def _plot_configure(self):
        """
        Creates a figure with subplots for states and actions.

        :return: (figure, axes) where:
            figure: a matplotlib Figure with subplots for state and controls
            axes: an AxesTuple object with references to all figure subplot axes
        """
        plt.ion()  # interactive mode allows dynamic updating of plot
        figure = plt.figure(figsize=(6, 11))

        spec = plt.GridSpec(nrows=3,
                            ncols=2,
                            width_ratios=[5, 1],  # second column very thin
                            height_ratios=[6, 5, 1],  # bottom row very short
                            wspace=0.3)

        # create subplots
        if self.is_plot_position:
            axes_state: Axes3D = figure.add_subplot(spec[0, 0:], projection='3d')
        else:
            axes_state = None
        axes_stick = figure.add_subplot(spec[1, 0])
        axes_throttle = figure.add_subplot(spec[1, 1])
        axes_rudder = figure.add_subplot(spec[2, 0])

        if self.is_plot_position:
            # config subplot for state
            axes_state.set_xlabel(self.props_to_plot['x']['label'])
            axes_state.set_ylabel(self.props_to_plot['y']['label'])
            axes_state.set_zlabel(self.props_to_plot['z']['label'])
            green_rgba = (0.556, 0.764, 0.235, 0.8)
            axes_state.w_zaxis.set_pane_color(green_rgba)

        # config subplot for stick (aileron and elevator control in x/y axes)
        axes_stick.set_xlabel('ailerons [-]', )
        axes_stick.set_ylabel('elevator [-]')
        axes_stick.set_xlim(left=-1, right=1)
        axes_stick.set_ylim(bottom=-1, top=1)
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
        axes_throttle.set_xlim(left=0, right=1)
        axes_throttle.spines['left'].set_position('zero')
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
        axes_rudder.set_ylim(bottom=0, top=1)
        axes_rudder.xaxis.set_label_coords(0.5, -0.5)
        axes_stick.spines['bottom'].set_position('zero')
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

        # create figure-wide legend
        cmd_entry = (plt.Line2D([], [], color='b', marker='o', ms=10, linestyle='', fillstyle='none'),
                     'Commanded Position, normalised')
        pos_entry = (plt.Line2D([], [], color='r', marker='+', ms=10, linestyle=''),
                     'Current Position, normalised')
        figure.legend((cmd_entry[0], pos_entry[0]),
                      (cmd_entry[1], pos_entry[1]),
                      loc='lower center')

        plt.show()
        plt.pause(self.PLOT_PAUSE_SECONDS)  # voodoo pause needed for figure to appear

        return figure, all_axes

    def _plot_position_state(self, sim: Simulation, all_axes: AxesTuple):
        """
        Plots the state of the simulation on input axes.

        State is given by three translational coords (x, y, z) and three
        linear velocities (v_x, v_y, v_z).

        The dict 'props' provides a mapping of these variable names to a
        dict specifying their 'name', the property to be retrieved from
        JSBSim.

        :param all_axes: AxesTuple, collection of axes of subplots to plot on
        """
        x, y, z = [sim[self.props_to_plot[var]['name']] for var in 'xyz']
        u, v, w = [sim[self.props_to_plot[var]['name']] for var in 'uvw']

        # get velocity vector coords using scaled velocity
        x2 = x + u / self.FT_PER_DEG_LAT
        y2 = y + v / self.ft_per_deg_lon
        z2 = z - w    # negative because v_z is positive down

        # plot aircraft position and velocity
        all_axes.axes_state.scatter([x], [y], zs=[z], c='k', s=10)
        if self.velocity_arrow:
            # get rid of previous timestep velocity arrow
            self.velocity_arrow.pop().remove()
        self.velocity_arrow = all_axes.axes_state.plot([x, x2], [y, y2], [z, z2], 'r-')

        # rescale the top z-axis limit, but keep bottom limit at 0
        all_axes.axes_state.set_autoscalez_on(True)
        all_axes.axes_state.set_zlim3d(bottom=0, top=None)

    def _plot_control_states(self, sim: Simulation, all_axes: AxesTuple):
        control_surfaces = ['ail', 'ele', 'thr', 'rud']
        ail, ele, thr, rud = [sim[self.props_to_plot[prop]['name']] for prop in control_surfaces]
        # plot aircraft control surface positions
        all_axes.axes_stick.plot([ail], [ele], 'r+', mfc='none', markersize=10, clip_on=False)
        all_axes.axes_throttle.plot([0], [thr], 'r+', mfc='none', markersize=10, clip_on=False)
        all_axes.axes_rudder.plot([rud], [0], 'r+', mfc='none', markersize=10, clip_on=False)

    def _plot_control_commands(self, sim: Simulation, all_axes: AxesTuple):
        """
        Plots agent-commanded actions on the environment figure.

        :param sim: Simulation to plot control commands from
        :param all_axes: AxesTuple, collection of axes of subplots to plot on
        """
        ail_cmd = sim['fcs/aileron-cmd-norm']
        ele_cmd = sim['fcs/elevator-cmd-norm']
        thr_cmd = sim['fcs/throttle-cmd-norm']
        rud_cmd = sim['fcs/rudder-cmd-norm']

        all_axes.axes_stick.plot([ail_cmd], [ele_cmd], 'bo', mfc='none', markersize=10, clip_on=False)
        all_axes.axes_throttle.plot([0], [thr_cmd], 'bo', mfc='none', markersize=10, clip_on=False)
        all_axes.axes_rudder.plot([rud_cmd], [0], 'bo', mfc='none', markersize=10, clip_on=False)


class FlightGearVisualiser(object):
    """ Class for visualising aircraft using the FlightGear simulator.

     This visualiser launches FlightGear and (by default) waits for it to
     launch. A Figure is also displayed (by creating its own FigureVisualiser)
     which is used to display the agent's actions.
     """
    TYPE = 'socket'
    DIRECTION = 'in'
    RATE = 60
    SERVER = ''
    PORT = 5550
    PROTOCOL = 'udp'
    LOADED_MESSAGE = 'loading cities done'
    FLIGHTGEAR_TIME_FACTOR = 5

    def __init__(self, sim: Simulation, block_until_loaded=True):
        """ Constructor

        Launches FlightGear in a subprocess and starts a figure for plotting
        actions.

        :param sim: Simulation that will be visualised
        :param block_until_loaded: visualiser will block until it detects that
            FlightGear has loaded if True.
        """
        self.configure_simulation(sim)
        self.flightgear_process = self._launch_flightgear(sim.get_model_name())
        self.figure = FigureVisualiser(sim, is_plot_position=False)
        if block_until_loaded:
            self._block_until_flightgear_loaded()

    def plot(self, sim: Simulation) -> None:
        """
        Updates a 3D plot of agent actions.
        """
        self.figure.plot(sim)

    @staticmethod
    def _launch_flightgear(aircraft_name: str):
        cmd_line_args = FlightGearVisualiser._create_cmd_line_args(aircraft_name)
        gym.logger.info(f'Subprocess: "{cmd_line_args}"')
        flightgear_process = subprocess.Popen(
            cmd_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        gym.logger.info('Started FlightGear')
        return flightgear_process

    def configure_simulation(self, sim: Simulation):
        sim.enable_flightgear_output()
        sim.set_simulation_time_factor(self.FLIGHTGEAR_TIME_FACTOR)

    @staticmethod
    def _create_cmd_line_args(aircraft_name: str):
        # FlightGear doesn't have a 172X model, use the P instead
        if aircraft_name == 'c172x':
            aircraft_name = 'c172p'

        flightgear_cmd = 'fgfs'
        aircraft_arg = f'--aircraft={aircraft_name}'
        flight_model_arg = '--native-fdm=' + f'{FlightGearVisualiser.TYPE},' \
                                             f'{FlightGearVisualiser.DIRECTION},' \
                                             f'{FlightGearVisualiser.RATE},' \
                                             f'{FlightGearVisualiser.SERVER},' \
                                             f'{FlightGearVisualiser.PORT},' \
                                             f'{FlightGearVisualiser.PROTOCOL}'
        flight_model_type_arg = '--fdm=' + 'external'
        disable_ai_arg = '--disable-ai-traffic'
        return (flightgear_cmd, aircraft_arg, flight_model_arg,
                flight_model_type_arg, disable_ai_arg)

    def _block_until_flightgear_loaded(self):
        while True:
            msg_out = self.flightgear_process.stdout.readline().decode()
            if self.LOADED_MESSAGE in msg_out:
                gym.logger.info('FlightGear loading complete; entering world')
                break
            else:
                time.sleep(0.001)

    def close(self):
        if self.flightgear_process:
            self.flightgear_process.kill()
            timeout_secs = 3
            self.flightgear_process.wait(timeout=timeout_secs)
