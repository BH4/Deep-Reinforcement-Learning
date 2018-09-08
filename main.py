# import json
# from keras.models import model_from_json

# from keras.optimizers import sgd

import os.path

from Training.qlearn import Training
from Testing.test import Testing

# Change what is being trained/tested here.
from Parameters import pong_target_1000 as param

if __name__ == "__main__":
    # These four lines can be modified.
    load_model = True
    train = True
    test = True
    make_gif = True

    env, model, name, training_input, testing_input = param.setup()

    (epsilon, epoch, max_memory, batch_size, target_model_update,
        enable_double, enable_dueling, dueling_type) = training_input

    (num_trials, num_games) = testing_input

    model_name = "models\\"+name
    gif_name = "gifs\\"+name

    # Define Model
    if load_model:
        # Only need to load weights over the current model. Assumes the model
        # defined in param is not changed from a previous run.
        if os.path.isfile(model_name+".h5"):
            # with open(model_name+".json", "r") as jfile:
            #    model = model_from_json(json.load(jfile))
            model.load_weights(model_name+".h5")
            # model.compile(sgd(lr=learning_rate), "mse")
        else:
            print("File was not found. Continuing with defined model.")

    # Training and/or testing
    if train:
        trainer = Training(env, model, model_name, max_memory, batch_size,
                           target_model_update, enable_double=enable_double,
                           enable_dueling=enable_dueling,
                           dueling_type=dueling_type)
        trainer.train(epsilon, epoch)

    if test:
        tester = Testing(env, model)
        tester.statistics(num_trials, kill_limit=100)
        if make_gif:
            tester.gif(num_games, gif_name)
