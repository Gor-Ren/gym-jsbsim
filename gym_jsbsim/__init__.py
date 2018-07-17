from gym.envs.registration import register

# register as OpenAI Gym environments
register(
    id='SteadyLevelFlightCessna-v0',
    entry_point='gym_jsbsim.environment_aliases:SteadyLevelFlightCessnaEnv_v0'
)

register(
    id='SteadyLevelFlightCessna-NoFG-v0',
    entry_point='gym_jsbsim.environment_aliases:SteadyLevelFlightCessnaEnv_NoFg_v0'
)

register(
    id='SteadyLevelFlightCessna-v1',
    entry_point='gym_jsbsim.environment_aliases:SteadyLevelFlightCessnaEnv_v1'
)

register(
    id='SteadyLevelFlightCessna-NoFG-v1',
    entry_point='gym_jsbsim.environment_aliases:SteadyLevelFlightCessnaEnv_NoFg_v1'
)
