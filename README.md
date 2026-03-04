# Deep-Reinforcement-Learning
My attempt to create a single AI that can play several games.

Inspired entirely by this paper https://arxiv.org/abs/1312.5602 about playing Atari games with the technique named Deep Q Learning.
I also started my code by studying and directly editing the project at https://gist.github.com/EderSantana/c7222daa328f0e885093.

In addition to the experience replay mechanism, I have added target networks and double DQN methods to the Deep Q-learning implemented here. Dueling DQN and Prioritized Experience Replay are potential next additions.

Currently this implementation successfully learns the simple catch game from the original project as well as Pong. I previously had a functioning implementation of tic-tac-toe and a functional game of snake (which was having more difficulty), but haven't updated their structure to fit the current one yet. Unfortunately lack of time (mostly my PhD and subsequent job) has delayed this and changes to Tensorflow mean coming back to the project will take even more time.

In the meantime here is the single gif I have of the trained pong game from before my environment stopped working.
![pong_target_1000](https://github.com/user-attachments/assets/607124eb-6a9e-4694-9862-0206e1fc183e)

