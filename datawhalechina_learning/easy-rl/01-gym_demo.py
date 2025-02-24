import gym

env = gym.make('MountainCar-v0')
env.reset()
env.seed(3)

print('-----------------------------------')
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')
print(f'Observation space range: {env.observation_space.low} - {env.observation_space.high}')
print('-----------------------------------')

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    env.step(action)
    
env.close()