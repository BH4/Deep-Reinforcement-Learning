import numpy as np
import random


class Pong(object):
    def __init__(self, width, height, trainer, simpleOutput=False, maxV=2, paddleV=1, paddleSize=4):
        self.grid_size = (width, height)
        self.maxV = maxV
        self.paddleV = paddleV  # not sure why I would ever change this from 1

        self.paddleSize = paddleSize
        assert paddleSize <= self.grid_size[1]

        self.trainer = trainer

        self.simpleOutput = simpleOutput

        self.p1 = [0, self.grid_size[1]//2]
        self.p2 = [self.grid_size[0]-1, self.grid_size[1]//2]

        # These are reset by reset_score()
        self.wins = 0

        # These will be set by reset()
        self.ballPos = None
        self.ballVel = None
        self.IHitBall = False
        self.reset()

    def _gameover(self):
        # player 2 scores
        if self.ballPos[0] <= 0:
            return 2
        # player 1 scores
        if self.ballPos[0] >= self.grid_size[0]-1:
            return 1
        return 0

    def _is_over(self):
        return not self._gameover() == 0

    # Types: Perfect, Medium, Random. Perfect is assumed if not allowed choice.
    def _trainer_move(self):
        if self.trainer == "Random":
            return random.choice([0, 1, 2])

        if self.trainer == "Medium":
            # Maybe play perfect maybe play random.
            r = random.random()
            if r > .75:
                return random.choice([0, 1, 2])

        if self.ballPos[1] < self.p2[1]:
            return 0
        if self.ballPos[1] > self.p2[1]:
            return 2
        return 1

    # action is 0 or 1 or 2 indicating up, nothing, and down respectively
    def _update_state(self, action):
        assert action == 0 or action == 1 or action == 2

        # update paddle positions
        self.p1[1] = min(max(self.p1[1]+(action-1)*self.paddleV, self.paddleSize//2), self.grid_size[1]-1-(self.paddleSize-1)//2)
        self.p2[1] = min(max(self.p2[1]+(self._trainer_move()-1)*self.paddleV, self.paddleSize//2), self.grid_size[1]-1-(self.paddleSize-1)//2)

        # update ball position
        self.ballPos[0] += self.ballVel[0]
        self.ballPos[1] += self.ballVel[1]

        # hit top or bottom
        if self.ballPos[1] < 0:
            self.ballPos[1] *= -1
            self.ballVel[1] *= -1
        elif self.ballPos[1] >= self.grid_size[1]:
            self.ballPos[1] = 2*(self.grid_size[1]-1) - self.ballPos[1]
            self.ballVel[1] *= -1

        # hit paddle (change angle based on place on paddle that is hit?)
        if self.ballPos[0] < 1:
            crossY = self.ballPos[1] + (self.ballVel[1]/self.ballVel[0])*(1-self.ballPos[0])

            if self.p1[1]-self.paddleSize//2 <= crossY <= self.p1[1]+self.paddleSize-1-self.paddleSize//2:
                # Ball did hit paddle
                self.ballVel[0] *= -1
                self.ballPos[0] = 2-self.ballPos[0]  # reflect off front of paddle, not wall

                self.IHitBall = True
        elif self.ballPos[0] >= self.grid_size[0]-1:
            crossY = self.ballPos[1]+(self.ballVel[1]/self.ballVel[0])*(self.grid_size[0]-1-self.ballPos[0])

            if self.p2[1]-self.paddleSize//2 <= crossY <= self.p2[1]+self.paddleSize-1-self.paddleSize//2:
                # Ball did hit paddle
                self.ballVel[0] *= -1
                self.ballPos[0] = 2*(self.grid_size[0]-2)-self.ballPos[0]  # reflect off front of paddle, not wall

    def _get_reward(self):
        g = self._gameover()
        if g == 2:
            return -1
        if g == 1:
            self.wins += 1
            return 1
        if self.IHitBall:
            return 1
        return 0

    def statistics(self):
        return "Wins {}".format(self.wins)

    def reset_score(self):
        self.wins = 0

    def draw_state(self):
        im_size = self.grid_size
        canvas = np.zeros(im_size)
        if 0 <= self.ballPos[0] < self.grid_size[0]:
            canvas[self.ballPos[0], self.ballPos[1]] = 1  # draw ball

        # draw paddles
        p1x = self.p1[0]
        p2x = self.p2[0]
        for i in range(self.paddleSize):
            p1y = self.p1[1]+i-self.paddleSize//2
            p2y = self.p2[1]+i-self.paddleSize//2
            canvas[p1x][p1y] = 1
            canvas[p2x][p2y] = 1

        # makes x and y match up if the array is printed directly
        canvas = np.transpose(canvas)
        return canvas

    def gameOver(self):
        return self._gameover()

    def observe(self):
        if self.simpleOutput:
            # paddle1, paddle2, ballx, bally, ballvx, ballvy
            return np.array([[self.p1[1], self.p2[1],
                              self.ballPos[0], self.ballPos[1],
                              self.ballVel[0], self.ballVel[1]]])
        else:
            # Full board
            canvas = self.draw_state()
            return np.array([canvas.reshape((self.grid_size[0],
                                             self.grid_size[1], 1))])

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        self.IHitBall = False
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        self.IHitBall = False

        self.ballPos = [self.grid_size[0]//2, self.grid_size[1]//2]

        # don't want ball to have 0 x velocity
        xv = random.randint(-self.maxV, self.maxV)
        while xv == 0:
            xv = random.randint(-self.maxV, self.maxV)

        # Actually just make it negative to start.
        # xv = random.randint(-self.maxV, -1)

        yv = random.randint(-self.maxV, self.maxV)
        while yv == 0:
            yv = random.randint(-self.maxV, self.maxV)

        self.ballVel = [xv, yv]
