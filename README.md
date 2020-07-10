# Acrobot Environment Solved using DQN
### Deep Q-Networks (DQN)
A Deep Q-Network is a neural network which acts as a value function approximator for environments which have very large state space and action space, thereby allowing us to easily predict the action-value function or state-value function for any state. Combined with the Q-learning Reinforcement Learning Algorithm, it can be used to solve complex environments.

### Acrobot Environment
The Acrobot Environment is an open-source Reinforcement Learning environment provided by OpenAI in their Gym. The acrobot system includes two joints and two links, where the joint between the two links is actuated. Initially, the links are hanging downwards, and the goal is to swing the end of the lower link up to a given height. This height is represented by a horizontal line. Lesser the timesteps taken to achieve the goal, better is our model.

### Results
The model was trained for 1000 episodes and the best reward was achieved with the weights after episode 200.
<p align="center">
 <img  width="600" height="400" src="https://github.com/Terabyte17/Acrobot-using-DQN/blob/master/media/ezgif-3-635b42c5fd90.gif">
</p>

The results on the last few episodes of testing are as follows:-

Episode number 89 reward: 76.92307692307692

Episode number 90 reward: 68.02721088435374

Episode number 91 reward: 83.33333333333333

Episode number 92 reward: 78.125

Episode number 93 reward: 67.56756756756756

Episode number 94 reward: 70.4225352112676

Episode number 95 reward: 69.93006993006993

Episode number 96 reward: 70.92198581560284

Episode number 97 reward: 75.75757575757575

Episode number 98 reward: 65.78947368421052

Episode number 99 reward: 78.125

Total Average Reward 73.32780033318028

Accuracy of Solving in Under 500 Timesteps: 0.97

### Episode Rewards (Training)
<p align="center">
 <img  width="600" height="400" src="https://github.com/Terabyte17/Acrobot-using-DQN/blob/master/media/Figure_1.png">
</p>

### TimeSteps per Episode (Training)
<p align="center">
 <img  width="600" height="400" src="https://github.com/Terabyte17/Acrobot-using-DQN/blob/master/media/Figure_2.png">
</p>

### Note
The training files can be found in acrobot_DQN.py and acrobot_agent.py. Testing file - acrobot_test.py. Model for Q-network - weights1200.h5
The model was trained locally by using keras==2.1.2 and tensorflow==1.14.0.
