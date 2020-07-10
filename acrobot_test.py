import gym
import numpy as np
import random
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from acrobot_DQN import DQN

env=gym.make("Acrobot-v1")
dqn_agent=DQN(env)
env.render()
num_episodes=100
DQN.model=load_model(r"C:\Users\yashs\OneDrive\Desktop\Acrobot using DQN\Acrobot_weights\weights1200.h5")  

def correct_shape(state):
	return np.reshape(state, [1,env.observation_space.shape[0]])

total_reward=0

for episode in range(num_episodes):
		state=env.reset()
		env.render()
		state=correct_shape(state)
		Q_values=DQN.model.predict(state)
		action=np.argmax(Q_values[0])
		done=False
		step=0
		while True:
			env.render()
			step+=1
			next_state,reward,done,_=env.step(action)
			next_state=correct_shape(next_state)
			Q_values=DQN.model.predict(next_state)
			next_action=np.argmax(Q_values[0])
			state=next_state
			action=next_action
			if done is True:
				reward=10000/step
				total_reward+=reward
				print("Epoch number "+ str(episode) + " reward: " + str(reward))
				break

print("Total Average Reward "+ str(total_reward/num_episodes))


				

			
		

