#need to write it all from scratch to learn.
#start with tic tac toe game then do learning. model it after the catch game.
import json
import numpy as np
import random
from copy import deepcopy
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

class TicTacToe(object):
    #if start is true then the player is starting instead of the trainer
    #the trainer is specified here as well. Can be "Random","Perfect","Middle"
    #the trainer takes moves by placing a -1 and the player takes moves by placing a 1.
    def __init__(self,start,trainer):
        self.dumbMove=0

        self.state=[0]*9
        self.trainer=trainer
        if not start:
            ind=self.getTrainerMove(self.trainer)
            assert self.state[ind]==0
            self.state[ind]=-1

    ############################################################
    #Get board state functions
    ############################################################
    def boardFull(self):
        return np.count_nonzero(self.state)==9

    #player is -1 or 1
    winInd=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    def hasWon(self,board,player):
        for r in self.winInd:
            row=[board[x] for x in r]
            if row==[player]*3:
                return True

        return False

    #return values:1=player 1 won, -1=player -1 won, 0 game still going, 2 cat game
    def gameOver(self):
        if self.hasWon(self.state,1):
            return 1
        if self.hasWon(self.state,-1):
            return -1
        if self.boardFull():
            return 2
        return 0

    ############################################################
    #Perfect player logic
    ############################################################

    def getWinMoves(self,board,player):
        winMoves=[]
        canWin=sorted([player,player,0])
        for r in self.winInd:
            row=sorted([board[x] for x in r])
            if row==canWin:
                i=0
                ind=r[i]
                while board[ind]!=0:
                    i+=1
                    ind=r[i]

                winMoves.append(ind)

        return winMoves

    def getForkPositions(self,board,player):
        board=deepcopy(board)

        forks=[]
        for ind,val in enumerate(board):
            if val==0:
                board[ind]=player
                c=len(self.getWinMoves(board,player))
                if c>=2:
                    forks.append(ind)
                board[ind]=0
        return forks


    def getPerfectMove(self):
        #Win
        myWinMoves=self.getWinMoves(self.state,-1)
        if len(myWinMoves)>0:
            return random.choice(myWinMoves)
        #Block
        playerWinMoves=self.getWinMoves(self.state,1)
        if len(playerWinMoves)>0:
            return playerWinMoves[0]
        #Fork
        myForks=self.getForkPositions(self.state,-1)
        if len(myForks)>0:
            return random.choice(myForks)
        #Block Forks/make row of 2 so other player cant get fork
        playerForks=self.getForkPositions(self.state,1)
        if len(playerForks)>0:
            possible=list((set(range(9))-set(playerForks))-set(np.nonzero(self.state)[0].tolist()))
            for p in possible:
                self.state[p]=-1
                wm=self.getWinMoves(self.state,-1)
                s=list(set(wm)-set(playerForks))
                self.state[p]=0
                if len(s)>0:
                    return random.choice(s)
            return random.choice(playerForks)
        #Center
        if self.state[4]==0:
            return 4
        #Opposite Corner
        temp=[(0,8),(2,6),(8,0),(6,2)]
        random.shuffle(temp)
        for t in temp:
            if self.state[t[0]]==1 and self.state[t[1]]==0:
                return t[1]
        """
        if self.state[0]==1 and self.state[8]==0:
            return 8
        if self.state[2]==1 and self.state[6]==0:
            return 6
        if self.state[8]==1 and self.state[0]==0:
            return 0
        if self.state[6]==1 and self.state[2]==0:
            return 2
        """
        #Empty Corner
        corners=[0,2,6,8]
        random.shuffle(corners)
        for c in corners:
            if self.state[c]==0:
                return c
        #Empty Side
        sides=[1,3,5,7]
        random.shuffle(sides)
        for s in sides:
            if self.state[s]==0:
                return s

        print("Error: typo or no empty spaces")
        return None

    ############################################################
    ############################################################

    def getTrainerMove(self,trainer):
        trainer=trainer.lower()
        #random player
        possible=list(set(range(9))-set(np.nonzero(self.state)[0].tolist()))
        if trainer=="random":
            return random.choice(possible)

        #perfect player
        p=self.getPerfectMove()
        assert p in possible
        if trainer=="perfect":
            return p

        #medium player
        if random.random()>.75:
            return random.choice(possible)
        return p

    def printBoard(self):
        s=""
        for i in range(3):
            for j in range(3):
                c=self.state[i*3+j]
                if c==1:
                    s+=" 1"
                elif c==0:
                    s+="  "
                else:
                    s+="-1"

                if j<2:
                    s+="|"

            s+="\n"
            if i<2:
                s+="-"*8

            s+="\n"
        print(s)


    ############################################################
    ############################################################
    def _is_over(self):
        return self.gameOver()!=0

    #action is a number from 0 to 8 indicating the place that the player wants to move
    #returns True if the action is allowed, returns False if the action is not allowed
    def _update_state(self, action):
        if self.state[action]!=0:
            return False

        self.state[action]=1

        #trainer move
        if not self._is_over():
            ind=self.getTrainerMove(self.trainer)
            assert self.state[ind]==0
            self.state[ind]=-1
        return True
    
    def _get_reward(self):
        g=self.gameOver()
        if g==2:
            return 0
        return g
    
    def observe(self):
        y=[x for x in self.state]
        return np.array(y).reshape((1, -1))

    def act(self, action):
        valid = self._update_state(action)
        reward = self._get_reward()
        if not valid:
            reward=-5
            self.dumbMove+=1

        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self,start,trainer):
        self.state=[0]*9
        self.trainer=trainer
        if not start:
            ind=self.getTrainerMove(self.trainer)
            assert self.state[ind]==0
            self.state[ind]=-1


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
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

#For command prompt play against the trainer
"""####################################################################
#game=TicTacToe(True,"Random")
game=TicTacToe(True,"Perfect")
#game=TicTacToe(False,"Medium")
game.printBoard()
while game.gameOver()==0:
    a=input("move index: ")
    state,reward,game_over=game.act(int(a))
    game.printBoard()
    print("reward: "+str(reward)+", game over: "+str(game_over))
print("winner is: "+str(game.gameOver()))
"""####################################################################



if __name__ == "__main__":
    # parameters
    epsilon = .1  #exploration
    num_actions = 9  #tic tac toe board numbered left to right then top to bottom
    epoch = 1000
    max_memory = 500
    hidden_size = 50
    batch_size = 50
    grid_size = 3

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    model.load_weights("model.h5")

    # Define environment/game
    start=False
    trainer="Perfect"
    env = TicTacToe(start,trainer)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    lose_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset(start,trainer)
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                possible=list(set(range(9))-set(np.nonzero(env.state)[0].tolist()))
                action = random.choice(possible)
                #action = np.random.randint(0, num_actions, size=1)
                #action=action[0]
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1
            if game_over and reward==-1:
                lose_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
        #env.printBoard()
        print("Epoch {:03d}/{} | Loss {:.4f} | Win count {} | Lose count {}".format(e, epoch-1, loss, win_cnt, lose_cnt))

    print("Number of dumb moves: "+str(env.dumbMove))
    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
