from gym.envs.registration import register

# register as OpenAI Gym environments
register(
    id='SteadyLevelFlightCessna-v0',
    entry_point='gym_jsbsim.environment:SteadyLevelFlightCessnaEnv'
)

register(
    id='SteadyLevelPitchControlCessnaEnv-v0',
    entry_point='gym_jsbsim.environment:SteadyLevelPitchControlCessnaEnv'
)


