from gym.envs.registration import register

register(
    id='twoDNavigation-v0',
    entry_point='gym_twoDNavigation.envs:TwoDNavigationEnv',
)
register(
    id='twoDNavigation-oracle-v0',
    entry_point='gym_twoDNavigation.envs:TwoDNavigationOracleEnv',
)
