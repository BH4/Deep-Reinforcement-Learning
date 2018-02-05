# will need to pass in multiple (try 4) frames so that the ai can see the motion of the ball
# position on which the ball hits the paddle should affect angle it travels at

# import json
import numpy as np
import random
# from keras.models import Sequential
# from keras.layers import Dense,Conv2D,Flatten
# from keras.optimizers import sgd


class Pong(object):
    def __init__(self, grid_size, maxV=2, paddleV=1):
        self.grid_size = grid_size
        self.maxV = maxV
        self.paddleV = paddleV  # not sure why I would ever change this from 1

        self.paddleSize = 4
        self.reset()

    def _gameover(self):
        # player 2 scores
        if self.ballPos[0] <= 0:
            return 2
        # player 1 scores
        if self.ballPos[0] >= self.grid_size-1:
            return 1
        return 0

    def _is_over(self):
        return not self._gameover() == 0

    def _trainer_move(self):
        if self.ballPos[1] < self.p2[1]:
            return 0
        if self.ballPos[1] > self.p2[1]:
            return 2
        return 1

    # action is 0 or 1 or 2 indicating up, nothing, and down respectively
    def _update_state(self, action):
        assert action == 0 or action == 1 or action == 2

        # update paddle positions
        self.p1[1] += (action-1)*self.paddleV
        self.p2[1] += (self._trainer_move()-1)*self.paddleV

        # update ball position
        self.ballPos[0] += self.ballVel[0]
        self.ballPos[1] += self.ballVel[1]

        # hit wall (make sure ball is reset if it goes over any wall. only reverse velocity if it hits bottom or top)

        # hit paddle (change angle based on place on paddle that is hit)

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
        self.score=0

        self.p1=(0,self.grid_size//2)
        self.p2=(self.grid_size-1,self.grid_size//2)
        self.ballPos=(self.grid_size//2,self.grid_size//2)

        #don't want ball to have 0 x velocity
        xv=random.randint(-self.maxV,self.maxV)
        while xv==0:
            xv=random.randint(-self.maxV,self.maxV)

        self.ballVel=(xv,random.randint(-self.maxV,self.maxV))
