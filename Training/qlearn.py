"""
Read these two:
https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8
https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682


(Much easier to understand. Will in the future have deuling and double currently has some others I haven't heard of) https://becominghuman.ai/beat-atari-with-deep-reinforcement-learning-part-2-dqn-improvements-d3563f665a2c

target network: Implemented but not properly tested
double dqn: Implemented
dueling dqn: 



















These are all in my text book on desktop
keras-rl has this stuff already. Probably copy it from there (Remember to cite it well)
https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py







Need to have the option of:
a separate target network (how is this different than double dqn?),
double dqn (Training 2 networks on symmetrically on separate experiences (https://en.wikipedia.org/wiki/Q-learning#Double_Q-learning)),
dueling dqn (separate the action value and the state value),
prioritized experience replay,
and any combination.
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df

If doing it with any combination is too difficult just add them all one at a
time to make sure they do improve performance.
"""

import numpy as np
import json
from keras.models import clone_model


class ExperienceReplay(object):
    def __init__(self, enable_double, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.enable_double = enable_double

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, target_model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = target_model.output_shape[-1]
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
            targets[i] = target_model.predict(state_t)[0]
            if self.enable_double:
                # Action to be taken chosen by model
                a = np.argmax(model.predict(state_tp1)[0])
                # Still use target model to get Q values
                Q_sa = target_model.predict(state_tp1)[0][a]
            else:
                Q_sa = np.max(target_model.predict(state_tp1)[0])

            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


class ExperienceReplay_double(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, target_model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = target_model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t

            targets[i] = target_model.predict(state_t)[0]
            Q_sa = target_model.predict(state_tp1)[0][np.argmax(model.predict(state_tp1)[0])]
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


class Training(object):
    def __init__(self, env, model, name, max_memory, batch_size,
                 target_model_update, enable_double=False,
                 enable_dueling=False, dueling_type='avg'):
        self.env = env
        self.model = model
        self.name = name

        if target_model_update < 1:
            raise ValueError('target_model_update must be >= 1 since soft updates are not implemented.')
        else:
            self.target_model_update = int(target_model_update)

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model)
        self.target_model.compile(optimizer='sgd', loss='mse')

        self.batch_size = batch_size

        self.exp_replay = ExperienceReplay(enable_double, max_memory=max_memory)

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, epsilon, num_epoch):
        num_actions = self.model.output_shape[-1]

        step = 0
        for e in range(num_epoch):
            loss = 0.
            self.env.reset()
            game_over = False
            # get initial input
            input_t = self.env.observe()

            while not game_over:
                step += 1
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
                inputs, targets = self.exp_replay.get_batch(self.model, self.target_model, batch_size=self.batch_size)

                loss += self.model.train_on_batch(inputs, targets)

                if step % self.target_model_update == 0:
                    self.update_target_model_hard()

            env_output = self.env.statistics()
            print("Epoch {}/{} | Loss {:.4f} | {}".format(e, num_epoch-1, loss,
                                                          env_output))

        # Save trained model weights and architecture, this will be used by the visualization code
        self.model.save_weights(self.name+".h5", overwrite=True)
        with open(self.name+".json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)
