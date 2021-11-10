import gym
import numpy as np
from cartpole_controller import LocalLinearizationController
from gym import wrappers
import math
from matplotlib import pyplot
video_path = "./gym-results"
x_init = np.array([0.5, 0.05, 0.5, 0.3])
x_s = np.array([0, 0, 0, 0], dtype=np.double)
u_s = np.array([0], dtype=np.double)
T = 500

env = gym.make("env:CartPoleControlEnv-v0")
controller = LocalLinearizationController(env)
policies = controller.compute_local_policy(x_s, u_s, T)

# For testing, we use a noisy environment which adds small Gaussian noise to
# state transition. Your controller only need to consider the env without noise.
# env = gym.make("env:NoisyCartPoleControlEnv-v0")

env = wrappers.Monitor(env, video_path, force = True)
total_cost = 0
observation = env.reset(state = x_init)
print(len(policies))
cnt=0
states=[]
for (K,k) in policies:
    cnt+=1
    print(cnt)
    # if cnt==300: break
    env.render()
    action = (K @ observation + k)
    observation, cost, done, info = env.step(action)
    states.append(observation)
    total_cost += cost
    if done: # When the state is out of the range, the cost is set to inf and done is set to True
        break
env.close()
print("cost = ", total_cost)
xs=[-1]
ys=[-1]
for i in states:
    x=i[0] + math.sin(i[2])# 0.5 * l
    y=0.2 + math.cos(i[2])
    xs.append(x)
    ys.append(y)
pyplot.scatter(xs,ys)
pyplot.axis([-1,5,0.75,1.5])
pyplot.show()
