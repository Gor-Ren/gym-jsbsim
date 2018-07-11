"""
A minimal example for a bug where JSBSim prints output despite having print
disabled.
"""
import jsbsim
import os


# init JSBSim and load basic config
root_dir = "/home/gordon/apps/jsbsim"

fdm = jsbsim.FGFDMExec(root_dir=root_dir)
fdm.set_debug_level(0)
fdm.load_model('c172x')
fdm.load_ic('reset00', useStoredPath=True)

# set socket output
output_file = "data_output/flightgear.xml"
output_config_path = os.path.join(root_dir, output_file)
fdm.set_output_directive(output_config_path)

fdm.run_ic()
pass
fdm.reset_to_initial_conditions(0)
pass
fdm.reset_to_initial_conditions(0)
pass