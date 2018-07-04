from gym.envs.registration import register

# register as an OpenAI Gym environment
register(
    id='SteadyLevelFlightCessna-v0',
    entry_point='gym_jsbsim.environment:SteadyLevelFlightCessnaEnv'
)