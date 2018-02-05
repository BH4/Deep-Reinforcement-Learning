import json
#import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from qlearn import TicTacToe


if __name__ == "__main__":
    # Make sure this grid size matches the value used from training
    grid_size = 3

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    start=False
    trainer="Perfect"
    env = TicTacToe(start,trainer)
    wins=0
    losses=0
    for e in range(1000):
        loss = 0.
        env.reset(start,trainer)
        game_over = False
        # get initial input
        input_t = env.observe()
        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if game_over and reward==1:
                wins+=1
            if game_over and reward==-1:
                losses+=1
            #env.printBoard()
            #print("="*50)

    print(str(wins)+" wins")
    print(str(losses)+" losses")