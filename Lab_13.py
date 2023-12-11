"""
!pip install pyvirtualdisplay
!pip install ipywidgets
!pip install matplotlib
!pip install box2d-py

!brew install --cask xquartz
!export PATH=$PATH:/opt/X11/bin
"""
from PIL import Image
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from Lab_13_DQN import DQN


# ref: https://xusophia.github.io/DataSciFinalProj/


def main():
    ##################################
    # UNDERSTAND THE ENVIRONMENT
    ##################################

    # https://www.gymlibrary.dev/environments/box2d/lunar_lander/
    # vectorized environment (a method for stacking multiple independent environments into a single environment) of 16 environments
    env = make_vec_env('LunarLander-v2', n_envs=1, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))

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

    model = DQN("MlpPolicy", env, 0.99, 0.1)

    ##################################
    # TRAIN THE MODEL
    ##################################

    # Let's train our DQN agent for 1,000,000 timesteps, don't forget to use GPU on Colab. It will take approximately ~20min, but you can use fewer timesteps if you just want to try it out.
    # 10k time steps about 7 episodes
    model.learn(total_timesteps=1_000)

    ##################################
    # EVALUATE THE MODEL
    ##################################
    print("EVALUATE THE MODEL")

    mean_reward, std_reward = evaluate_policy(model, model.env, n_eval_episodes=10)

    print("RESULTS:")
    print("An episode is considered successful if the agent scores more than 200 points.")
    print(f"{mean_reward:.2f} +/- {std_reward:.2f}")

    model.onlineNetwork.summary()
    model.onlineNetwork.save("lab_13_trained_model_temp")
    #model.onlineNetwork.load("lab_13_trained_model_temp")


    ##################################
    # VISUALIZE THE MODEL
    ##################################
    print("VISUALIZE THE MODEL")

    visualize = True

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(model.sumRewardsEpisode, marker='o')
    plt.title('Sum of Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    if visualize: plt.show()
    plt.savefig('lab_13_sum_rewards_episode.png')
    # create gif
    frames = []
    vec_env = model.env
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if visualize: vec_env.render("human")
        frames.append(Image.fromarray(vec_env.render()))
    frames[0].save('lab_13_lunar_lander.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

    env.close()
    print("DONE")


if __name__ == '__main__':
    main()



