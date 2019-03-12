import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    print("In init")
    
  def step(self, action):
    return
  
  def reset(self):
    return
  
  def render(self, mode='human'):
    return 
  
  def close(self):
    return
