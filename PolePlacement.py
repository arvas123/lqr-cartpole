import gym
import numpy as np
from gym import wrappers
from gym import spaces
from matplotlib import pyplot
import math
video_path = "./gym-results"
x_init = np.array([0.5, 0.05, 0.5, 0.3])
x_s = np.array([0, 0, 0, 0], dtype=np.double)
u_s = np.array([0], dtype=np.double)
T = 500
A = np.array([[0,    1.0000,         0,         0],
[0,         0,   -0.7171,         0],
[0,         0,         0,    1.0000],
[0         ,0   ,15.7756         ,0]])
B = [[0],[0.9756],[0],[-1.4634]]
K = [[-3.4265  , -5.1813,  -41.8399 , -10.6976]]
a = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
env = gym.make("env:CartPoleControlEnv-v0")
env = wrappers.Monitor(env, video_path, force = True)
observation = env.reset(state = x_init)
states=[]
for _ in range(300):
    env.render()
    input()
    action = -(K@observation)
    observation, cost, done, info = env.step(action)
    if observation[0]<0.02:
        break
    states.append(observation)
    if done:
        break
env.close()
xs=[-1]
ys=[-1]
for i in states:
    x=i[0] + math.sin(i[2])# 0.5 * l
    y=0.2 + math.cos(i[2])
    xs.append(x)
    ys.append(y)
    

pyplot.scatter(xs,ys)
pyplot.axis([0,2,1,1.25])
pyplot.show()