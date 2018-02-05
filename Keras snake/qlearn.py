#Gonna have what the ai sees be the board with no other information besides the score/reward

import json
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten
from keras.optimizers import Adam,sgd

class Snake(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    directionDict={0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}

    def getNewApple(self):
        s=set(self.snake)
        x=random.randint(0,self.grid_size-1)
        y=random.randint(0,self.grid_size-1)
        while (x,y) in s:
            x=random.randint(0,self.grid_size-1)
            y=random.randint(0,self.grid_size-1)
        return (x,y)

    def _gameover(self):
        if self.bugFlag:
            return 1
        if len(self.snake)!=len(set(self.snake)):
            return 1
        head=self.snake[0]
        if  head[0]<0 or head[0]>=self.grid_size or head[1]<0 or head[1]>=self.grid_size:
            return 2
        return 0

    def _is_over(self):
        return not self._gameover()==0

    #action is 0 or 1 or 2 indicating left, nothing, or right respectively
    def _update_state(self, action):
        #assert action==0 or action==1 or action==2
        #self.direction=(self.direction+(action-1))%4

        assert action==0 or action==1 or action==2 or action==3
        #hack fix for the bug that the snake can sometimes move through itself.
        if (self.direction==0 and action==2) or (self.direction==2 and action==0) or (self.direction==1 and action==3) or (self.direction==3 and action==1):
            self.bugFlag=True
        self.direction=action

        head=self.snake[0]
        move=self.directionDict[self.direction]
        newHead=(head[0]+move[0],head[1]+move[1])
        self.snake=[newHead]+self.snake

        #no need to remove the tail if I ate an apple
        if newHead!=self.apple:
            self.snake.pop(-1)
        
        #getting a new apple is handled by the _get_reward function

        #self collision and wall collision are handled by the _is_over function
    
    def _draw_state(self):
        im_size = (self.grid_size,)*2
        canvas = np.zeros(im_size)
        canvas[self.apple[0], self.apple[1]] = 1  #draw apple
        for i,sp in enumerate(self.snake):
            if not (sp[0]<0 or sp[0]>=self.grid_size or sp[1]<0 or sp[1]>=self.grid_size):#check if it is on grid
                if i==0:
                    canvas[sp[0],sp[1]] = 3  #draw head
                else:
                    canvas[sp[0],sp[1]] = 2  #draw snake

        canvas=np.transpose(canvas)#makes x and y match up if the array is printed directly
        return canvas

    def _get_reward(self):
        if self.snake[0]==self.apple:
            self.score+=1

            self.apple=self.getNewApple()#new apple
            return 1

        g=self._gameover()
        if g>0:
            return -1
        
        #manhattenDist=abs(self.snake[0][0]-self.apple[0])+abs(self.snake[0][1]-self.apple[1])
        return 0#-1*manhattenDist/10
    
    def observe(self):
        canvas = self._draw_state()
        return np.array([canvas.reshape((self.grid_size, self.grid_size, 1))])

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        self.bugFlag=False

        self.score=0

        self.snake=[(self.grid_size//2,self.grid_size//2)]
        self.apple=self.getNewApple()
        self.direction=0#0=up, 1=right, 2=down, 3=left
        


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        #env_dim = self.memory[0][0][0].shape[1]
        #in_shape=(min(len_memory, batch_size), env_dim, env_dim, 1)
        env_shape = self.memory[0][0][0].shape[1:]
        in_shape=tuple([min(len_memory, batch_size)]+ list(env_shape))

        inputs = np.zeros(in_shape)
        targets = np.zeros((in_shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=in_shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i] = state_t
            # There should be no target values for actions not taken.
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


"""
#test snake in command prompt
game=Snake(5)
print(game._draw_state())
actions={1:3,2:2,3:1,5:0}
while not game._is_over():
    a=input("move direction (1,2,3,5): ")
    state,reward,game_over=game.act(actions[int(a)])
    print(game._draw_state())
    print("reward: "+str(reward)+", game over: "+str(game_over))
"""

if __name__ == "__main__":
    # parameters
    epsilon = .1  #exploration

    alpha=.1
    num_actions = 4  #[up, right, down, left]
    epoch = 1000

    max_memory = 1000
    batch_size = 100
    gamma = 0.9

    hidden_size = 128
    grid_size = 6

    model = Sequential()
    #first attempt
    #model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    #model.add(Dense(hidden_size, activation='relu'))
    #model.add(Dense(num_actions))
    model.add(Conv2D(8,(4,4), strides=2, activation='relu', input_shape=(grid_size,grid_size,1)))
    model.add(Conv2D(16,(2,2), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=alpha), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    model.load_weights("model.h5")

    # Define environment/game
    env = Snake(grid_size)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory, discount=gamma)

    # Train
    best_score = 0
    num_times = 0
    score = 0
    last_move_random=False

    #ending types
    rand_wall=0
    wall=0
    rand_self=0
    hit_self=0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                last_move_random=True

                action = random.randint(0,num_actions-1)
            else:
                last_move_random=False

                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
        
        score=env.score
        if score>best_score:
            num_times=1
            best_score=score
        elif score==best_score:
            num_times+=1
        print("Epoch {:03d}/{} | Loss {:.4f} | Apples {} | Best Score {}x{}".format(e, epoch-1, loss, score, best_score, num_times))

        #death type
        g=env._gameover()
        if g==1:
            if last_move_random:
                rand_self+=1
            else:
                hit_self+=1
        elif g==2:
            if last_move_random:
                rand_wall+=1
            else:
                wall+=1
        rand_tot=rand_self+rand_wall
        tot=rand_tot+wall+hit_self
        if e%10==0:
            print("Random Death {:.1f}% | Random wall (of tot rand) {:.1f}% | Non-random wall death {:.1f}%".format(100*rand_tot/tot,100*rand_wall/rand_tot if rand_tot!=0 else 0,100*wall/(wall+hit_self) if (wall+hit_self)!=0 else 0))
    
    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
