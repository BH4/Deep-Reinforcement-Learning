"""
Would like to add some kind of debug mode. Maybe print the current board state
sometimes in this mode?
"""

import numpy as np
import json


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
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


class Training(object):
    def __init__(self, env, model, name, max_memory, batch_size):
        self.env = env
        self.model = model
        self.name = name

        self.batch_size = batch_size  # Should batch_size be an input to the ER object?

        self.exp_replay = ExperienceReplay(max_memory=max_memory)

    def train(self, epsilon, num_epoch):
        # Could have instead made this an input
        num_actions = len(self.model.predict(self.env.observe()))

        for e in range(num_epoch):
            loss = 0.
            self.env.reset()
            game_over = False
            # get initial input
            input_t = self.env.observe()

            while not game_over:
                input_tm1 = input_t
                # get next action
                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, num_actions, size=1)
                else:
                    q = self.model.predict(input_tm1)
                    action = np.argmax(q[0])

                # apply action, get rewards and new state
                input_t, reward, game_over = self.env.act(action)

                # store experience
                self.exp_replay.remember([input_tm1, action, reward, input_t],
                                         game_over)

                # adapt model
                inputs, targets = self.exp_replay.get_batch(self.model, batch_size=self.batch_size)

                loss += self.model.train_on_batch(inputs, targets)

            env_output = self.env.statistics()
            print("Epoch {}/{} | Loss {:.4f} | {}".format(e, num_epoch-1, loss,
                                                          env_output))

        # Save trained model weights and architecture, this will be used by the visualization code
        self.model.save_weights(self.name+".h5", overwrite=True)
        with open(self.name+".json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)
