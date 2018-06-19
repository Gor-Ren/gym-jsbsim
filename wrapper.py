import jsbsim

class JsbSimWrapper(object):
    """
    A class which wraps an instance of JSBSim and manages communication with it.
    """
    def __init__(self):
        root_dir = os.path.abspath("/home/gordon/Apps/jsbsim-code")
        self.sim = jsbsim.FGFDMExec(root_dir=root_dir)
