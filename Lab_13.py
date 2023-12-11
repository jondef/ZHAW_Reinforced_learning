"""
!pip install pyvirtualdisplay
!pip install ipywidgets
!pip install matplotlib
!pip install box2d-py

!brew install --cask xquartz
!export PATH=$PATH:/opt/X11/bin
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from Lab_13_DQN import DQN

# ref: https://xusophia.github.io/DataSciFinalProj/
visualize = True

##################################
# UNDERSTAND THE ENVIRONMENT
##################################

# https://www.gymlibrary.dev/environments/box2d/lunar_lander/
# vectorized environment (a method for stacking multiple independent environments into a single environment) of 16 environments
env = make_vec_env('LunarLander-v2', n_envs=4)

print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())  # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())  # Take a random action

##################################
# CREATE THE MODEL
##################################

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

model = DQN("MlpPolicy", env, 0.9, 0.1)

##################################
# TRAIN THE MODEL
##################################

# Let's train our DQN agent for 1,000,000 timesteps, don't forget to use GPU on Colab. It will take approximately ~20min, but you can use fewer timesteps if you just want to try it out.
model.learn(total_timesteps=2)

##################################
# EVALUATE THE MODEL
##################################
print("EVALUATE THE MODEL")

custom = True
if custom:
    mean_reward, std_reward = evaluate_policy(model, model.env, n_eval_episodes=10)

    print("RESULTS:")
    print(f"{mean_reward:.2f} +/- {std_reward:.2f}")
    # An episode is considered successful if the agent scores more than 200 points.
else:
    # get the obtained rewards in every episode
    model.sumRewardsEpisode
    model.onlineNetwork.summary()
    model.onlineNetwork.save("trained_model_temp")

##################################
# VISUALIZE THE MODEL
##################################

if visualize:
    print("VISUALIZE THE MODEL")
    vec_env = model.env
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
env.close()
print("DONE")