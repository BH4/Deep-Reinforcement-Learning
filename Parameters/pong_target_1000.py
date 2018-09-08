from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

from Environments.Pong import Pong


def setup():
    name = "pong_target_1000"  # ~87% after 1000 training epochs

    # ai = "Random"
    ai = "Medium"
    # ai = "Perfect"

    # model inputs
    num_actions = 3  # [up, nothing, down]
    hidden_size1 = 20
    hidden_size2 = 10
    height = 5
    width = 10
    num_inputs = 6
    learning_rate = .03

    # Trainer inputs
    max_memory = 3000
    batch_size = 100
    target_model_update = 1000
    enable_double = False
    enable_dueling = False
    dueling_type = 'avg'

    # Training time inputs
    epsilon = .1  # exploration
    epoch = 1000

    # Testing inputs
    num_trials = 10000
    num_games = 10

    # Define environment
    env = Pong(width, height, ai, simpleOutput=True, paddleSize=2)

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
