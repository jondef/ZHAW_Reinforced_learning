"""
!pip install pyvirtualdisplay
!pip install ipywidgets
!pip install matplotlib
!pip install box2d-py

!brew install --cask xquartz
!export PATH=$PATH:/opt/X11/bin
"""

import gym

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from sys import exit

# ref: https://xusophia.github.io/DataSciFinalProj/
visualize = False

##################################
# UNDERSTAND THE ENVIRONMENT
##################################

env = gym.make("LunarLander-v2")
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action

# https://www.gymlibrary.dev/environments/box2d/lunar_lander/

##################################
# CREATE THE MODEL
##################################

# vectorized environment (a method for stacking multiple independent environments into a single environment) of 16 environments
env = make_vec_env('LunarLander-v2', n_envs=32)

# We have studied our environment and we understood the problem: **being able to land the Lunar Lander to the Landing Pad correctly by controlling left, right and main orientation engine**. Now let's build the algorithm we're going to use to solve this Problem.
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            ent_coef=0.001,
            vf_coef=0.5,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            batch_size=64
            )


##################################
# TRAIN THE MODEL
##################################

# Let's train our DQN agent for 1,000,000 timesteps, don't forget to use GPU on Colab. It will take approximately ~20min, but you can use fewer timesteps if you just want to try it out.
model.learn(total_timesteps=1_000)


##################################
# EVALUATE THE MODEL
##################################

# Create a new environment for evaluation
eval_env = Monitor(gym.make("LunarLander-v2"))

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

# Print the results
print(f"{mean_reward:.2f} +/- {std_reward:.2f}")
# An episode is considered successful if the agent scores more than 200 points.

##################################
# VISUALIZE THE MODEL
##################################
if not visualize:
    exit()


def show_animation(frames):
    fig, ax = plt.subplots(figsize=(10, 6))
    patch = ax.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig=fig, func=animate, frames=len(frames), interval=20)
    plt.show(block=True)
    return anim  # to prevent anim object from being garbage collected


def get_frame():
    frame = env.render()
    return Image.fromarray(frame)


for _ in range(100):
    frames = []
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    observation = env.reset()[0]

    for _ in range(1000):  # Run for 1000 steps or till the episode ends
        frames.append(get_frame())
        y = model.predict(observation, deterministic=True)[0]  # Get the action predicted by the agent
        action = y
        observation, _, done, _, _ = env.step(action)
        if done:
            env.reset()  # Reset the environment if the episode ends
            break

    env.close()
    anim = show_animation(frames)


