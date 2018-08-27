import json
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

from Environments.Catch import Catch
from Training.qlearn import Training
from Testing.test import Testing


if __name__ == "__main__":
    load_model = True
    train = True
    test = True

    # model inputs
    num_actions = 3  # [move_left, stay, move_right]
    grid_size = 10
    num_inputs = grid_size**2
    learning_rate = .1
    hidden_size1 = num_inputs
    hidden_size2 = num_inputs//2

    # Training time inputs
    epsilon = .1  # exploration
    epoch = 2000

    # Trainer inputs
    max_memory = 500
    batch_size = 50
    model_name = "models\\catch_model_small"

    # Testing inputs
    num_trials = 1000
    num_games = 10
    gif_name = "gifs\\catch_small_model"

    # Define environment
    env = Catch(grid_size)

    # Define Model
    if load_model:
        with open(model_name+".json", "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(model_name+".h5")
        model.compile(sgd(lr=learning_rate), "mse")
    else:
        model = Sequential()
        model.add(Dense(hidden_size1, input_shape=(num_inputs,),
                        activation='relu'))
        model.add(Dense(hidden_size2, activation='relu'))
        model.add(Dense(num_actions))
        model.compile(sgd(lr=learning_rate), "mse")

    # Training and/or testing
    if train:
        trainer = Training(env, model, model_name, max_memory, batch_size)
        trainer.train(epsilon, epoch)

    if test:
        tester = Testing(env, model)
        # tester.statistics(num_trials)
        tester.gif(num_games, gif_name)
