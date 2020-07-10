import gym
import numpy as np
import random
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

MEMORY_SIZE=1000000
LR=1e-4

class DQN:
	def __init__(self,env):
		self.env=env
		self.memory=deque(maxlen=MEMORY_SIZE)
		self.gamma=0.95
		self.epsilon=0.95
		self.epsilon_decay=0.95
		self.epsilon_final=0.1
		self.batch_size=32
		
		self.model=self.create_q_network()
		self.target_model=self.create_q_network()
		
	def create_q_network(self):							#neural network for value function approximation

		model=Sequential()
		
		model.add(Dense(self.env.observation_space.shape[0],input_shape=(self.env.observation_space.shape[0],),activation="relu"))
		model.add(Dense(16,activation="relu"))
		model.add(Dense(8,activation="relu"))
		model.add(Dense(self.env.action_space.n,activation="linear"))
		
		model.compile(loss="mse",optimizer=Adam(lr=LR))
		return model
		
	def act(self,state):								#using an epsilon-greedy implicit policy w.r.t the action-value function
		q_values=self.model.predict(state)
		policy=(np.ones(q_values.shape[1],dtype=float)/q_values.shape[1])*self.epsilon
		best_action=np.argmax(q_values[0])
		policy[best_action]+=1-self.epsilon
		action=np.random.choice(np.arange(len(policy)),p=policy)
		return action
		
	def add_experience(self,state,action,reward,next_state,done):		#adding experience for experience replay
		self.memory.append((state,action,reward,next_state,done))  
		
	def experience_replay(self):
		if len(self.memory)<self.batch_size:
			return										#if memory is very less, no experience replay
		
		samples=random.sample(self.memory,self.batch_size)				#sampling random experiences from queue
		for sample in samples:
			state,action,reward,next_state,done=sample
			target_values=self.target_model.predict(next_state)
			if done:
				q_update=reward
			else:
				q_update=reward+self.gamma*np.max(target_values)		#taking max over all the action value functions given by the target model(to ensure stable targets)
			q_values=self.model.predict(state)							
			q_values[0][action]=q_update								#taking max over all action value functions may overestimate the action value function
			self.model.fit(state, q_values, verbose=0)
		self.epsilon*=self.epsilon_decay		
		self.epsilon=max(self.epsilon,self.epsilon_final)
		
			
	def target_train(self):
		self.target_model.set_weights(self.target_model.get_weights())   #updating the weights of the target model with the model we have trained
		
	def save_model(self,epoch_num):
		self.model.save(r"C:\Users\yashs\OneDrive\Desktop/Acrobot_weights/weights"+str(epoch_num+1000)+".h5")			
	
	
	
		
