"""
!pip install pyvirtualdisplay
!pip install ipywidgets
!pip install matplotlib
!pip install box2d-py

!brew install --cask xquartz
!export PATH=$PATH:/opt/X11/bin
"""



from pyvirtualdisplay import Display

display = Display(visible=False, size=(1400, 900))
display.start()


import gym
env = gym.make('LunarLander-v2', render_mode='rgb_array')


from IPython import display as ipythondisplay
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
from sys import exit




def show_animation(frames):
    fig, ax = plt.subplots(figsize=(10, 6))
    patch = ax.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig=fig, func=animate, frames=len(frames), interval=20)
    plt.show()
    return anim  # to prevent anim object from being garbage collected

def get_frame():
    frame = env.render()
    return Image.fromarray(frame)



frames = []

# Reset the environment and get the initial observation
observation = env.reset()

for _ in range(1000):  # Run for 1000 steps or till the episode ends
    frames.append(get_frame())
    action = env.action_space.sample()  # Take a random action
    observation, _, done, _, _= env.step(action)
    if done:
        env.reset()  # Reset the environment if the episode ends
        break

env.close()
#frames[0].save('frames.png')
anim = show_animation(frames)


