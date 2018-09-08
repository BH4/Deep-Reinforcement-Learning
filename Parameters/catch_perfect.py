from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

from Environments.Catch import Catch


def setup():
    name = "catch_perfect"

    # model inputs
    num_actions = 3  # [move_left, stay, move_right]
    grid_size = 10
    num_inputs = grid_size**2
    learning_rate = .1
    hidden_size1 = num_inputs
    hidden_size2 = num_inputs//2

    # Trainer inputs
    max_memory = 1000
    batch_size = 100
    target_model_update = 1000
    enable_double = False
    enable_dueling = False
    dueling_type = 'avg'

    # Training time inputs
    epsilon = .1  # exploration
    epoch = 2000

    # Testing inputs
    num_trials = 10000
    num_games = 50

    # Define environment
    env = Catch(grid_size)

    # Define Model
    model = Sequential()
    model.add(Dense(hidden_size1, input_shape=(num_inputs,),
                    activation='relu'))
    model.add(Dense(hidden_size2, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=learning_rate), "mse")

    training_input = (epsilon, epoch, max_memory, batch_size,
                      target_model_update, enable_double, enable_dueling,
                      dueling_type)

    testing_input = (num_trials, num_games)

    return env, model, name, training_input, testing_input
