import math
import collections


class Property(collections.namedtuple('Property', ['name', 'description', 'min', 'max'])):
    pass


class InitialProperty(collections.namedtuple('InitialProperty', ['name', 'description'])):
    pass


# aircraft state
altitude_ft = Property('position/h-sl-ft', 'altitude above mean sea level [ft]', -1400, 85000)
pitch_rad = Property('attitude/pitch-rad', 'pitch [rad]', -0.5 * math.pi, 0.5 * math.pi)
roll_rad = Property('attitude/roll-rad', 'roll [rad]', -math.pi, math.pi)
heading_deg = Property('attitude/psi-deg', 'heading [deg]', 0, 360)
u_fps = Property('velocities/u-fps', 'body frame x-axis velocity [ft/s]', -2200, 2200)
v_fps = Property('velocities/v-fps', 'body frame y-axis velocity [ft/s]', 2200, -2200)
w_fps = Property('velocities/w-fps', 'body frame z-axis velocity [ft/s]', -2200, 2200)
p_radps = Property('velocities/p-rad_sec', 'roll rate [rad/s]', -2 * math.pi, 2 * math.pi)
q_radps = Property('velocities/q-rad_sec', 'pitch rate [rad/s]', -2 * math.pi, 2 * math.pi)
r_radps = Property('velocities/r-rad_sec', 'yaw rate [rad/s]', -2 * math.pi, 2 * math.pi)
# controls state
aileron_left = Property('fcs/left-aileron-pos-norm', 'left aileron position, normalised', -1, 1)
aileron_right = Property('fcs/right-aileron-pos-norm', 'right aileron position, normalised', -1, 1)
elevator = Property('fcs/elevator-pos-norm', 'elevator position, normalised', -1, 1)
rudder = Property('fcs/rudder-pos-norm', 'rudder position, normalised', -1, 1)
throttle = Property('fcs/throttle-pos-norm', 'throttle position, normalised', 0, 1)
# controls command
aileron_cmd = Property('fcs/aileron-cmd-norm', 'aileron commanded position, normalised', -1.0, 1.0)
elevator_cmd = Property('fcs/elevator-cmd-norm', 'elevator commanded position, normalised', -1.0, 1.0)
rudder_cmd = Property('fcs/rudder-cmd-norm', 'rudder commanded position, normalised', -1.0, 1.0)
throttle_cmd = Property('fcs/throttle-cmd-norm', 'throttle commanded position, normalised', 0, 1)
mixture_cmd = Property('fcs/mixture-cmd-norm', 'engine mixture setting, normalised', 0, 1)
# initial conditions
initial_altitude_ft = InitialProperty('ic/h-sl-ft', 'initial altitude MSL [ft]')
initial_terrain_altitude_ft = InitialProperty('ic/terrain-elevation-ft', 'initial terrain alt [ft]')
initial_longitude_geoc_deg = InitialProperty('ic/long-gc-deg', 'initial geocentric longitude [deg]')
initial_latitude_geod_deg = InitialProperty('ic/lat-geod-deg', 'initial geodesic latitude [deg]')
initial_u_fps = InitialProperty('ic/u-fps', 'body frame x-axis velocity; positive forward [ft/s]')
initial_v_fps = InitialProperty('ic/v-fps', 'body frame y-axis velocity; positive right [ft/s]')
initial_w_fps = InitialProperty('ic/w-fps', 'body frame z-axis velocity; positive down [ft/s]')
initial_p_radps = InitialProperty('ic/p-rad_sec', 'roll rate [rad/s]')
initial_q_radps = InitialProperty('ic/q-rad_sec', 'pitch rate [rad/s]')
initial_r_radps = InitialProperty('ic/r-rad_sec', 'yaw rate [rad/s]')
initial_roc_fpm = InitialProperty('ic/roc-fpm', 'initial rate of climb [ft/min]')
initial_heading_deg = InitialProperty('ic/psi-true-deg', 'initial (true) heading [deg]')