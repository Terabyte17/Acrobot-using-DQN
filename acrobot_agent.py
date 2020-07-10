import gym
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from acrobot_DQN import DQN

env=gym.make("Acrobot-v1")
dqn_agent=DQN(env)
num_episodes=1000
ep_rewards=[]    #list to store rewards in each episode
timesteps=[]     #list to store timesteps each episode takes

def correct_shape(state):
	return np.reshape(state, [1,env.observation_space.shape[0]])		#correct the shape so that it can be input in the NN for prediction

for episodes in range(num_episodes):
	state=env.reset()
	state=correct_shape(state)
	action=dqn_agent.act(state)
	step=0
	while True:
		step+=1
		if step%100==0:
			print(step)
		env.render()
		next_state,reward,done,_=env.step(action)
		if done is True:
			reward=10000/step											#this reward was chosen as the reward system of the env. was very weak
		next_state=correct_shape(next_state)
		next_action=dqn_agent.act(next_state)  
		dqn_agent.add_experience(state,action,reward,next_state,done)
		state=next_state
		action=next_action
		if done:
			if episodes%10==0:
				dqn_agent.save_model(episodes)
			dqn_agent.target_train()
			print("Episode score is " + str(reward))
			ep_rewards.append(reward)
			timesteps.append(step)
			break
		dqn_agent.experience_replay()

plt.plot(ep_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
plt.savefig(r"C:\Users\yashs\OneDrive\Desktop\Episode_rewards.png")
cv2.waitKey(0)
plt.clf()

plt.plot(timesteps)
plt.title("TimeSteps")
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.show()
plt.savefig(r"C:\Users\yashs\OneDrive\Desktop\Timesteps.png")
cv2.waitKey(0)
plt.clf()



		
		
		
		
		
