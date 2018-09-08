import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os


class Testing(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def statistics(self, num_trials, kill_limit=1000):
        """
        Allows the model to play the game num_trial times then outputs the
        statistics from env. Will stop a game if it lasts more than
        kill_limit actions.
        """
        self.env.reset_score()
        killed_games = 0
        p = 10
        for e in range(num_trials):
            if int(e*100/num_trials) >= p:
                print("Testing {}% compleate".format(p))
                p += 10

            c = 0
            self.env.reset()
            game_over = False

            # get initial input
            input_t = self.env.observe()

            while not game_over and c < kill_limit:
                # get next action
                q = self.model.predict(input_t)
                action = np.argmax(q[0])

                # apply action, get rewards and new state
                input_t, reward, game_over = self.env.act(action)

                c += 1

            if c >= kill_limit:
                killed_games += 1

        print("Total trials {} | {} | Killed games {}".format(num_trials, self.env.statistics(), killed_games))

    def gif(self, num_games, gif_name,
            slow_mult=2, delete_pics=True, kill_limit_per_game=1000):
        """
        Allows the model to play the game num_games times then creates a gif
        from them. Will stop a game if it lasts more than kill_limit images.
        """
        slow_mult = int(slow_mult)

        try:
            os.remove(gif_name+'.gif')
        except Exception:
            pass

        kill_limit = kill_limit_per_game * num_games

        c = 0
        e = 0
        while c < kill_limit and e < num_games:
            self.env.reset()
            game_over = False
            # get initial input
            input_t = self.env.observe()

            plt.imshow(self.env.draw_state(),
                       interpolation='none', cmap='gray')
            plt.savefig("gifs\\%d.png" % c)
            plt.close()
            c += 1
            while not game_over and c < kill_limit:
                input_tm1 = input_t

                # get next action
                q = self.model.predict(input_tm1)
                action = np.argmax(q[0])

                # apply action, get rewards and new state
                input_t, reward, game_over = self.env.act(action)

                plt.imshow(self.env.draw_state(),
                           interpolation='none', cmap='gray')
                plt.savefig("gifs\\%d.png" % c)
                plt.close()
                c += 1

            e += 1

        # Making a temporary gif and slowing it down seems to be the only way I
        # can make a slower gif. For some reason the command works in cmd but
        # not here so i guess I am stuck with fast gifs.
        """
        call1 = ['ffmpeg', '-i', '%d.png', gif_name+'_temp.gif']
        subprocess.call(call1)
        call2 = ['ffmpeg', '-i', gif_name+'_temp.gif', '-filter:v',
                 '"setpts={}.0*PTS"'.format(slow_mult), gif_name+'.gif']
        subprocess.call(call2, shell=True)
        # ffmpeg -i catch_small_model.gif -filter:v "setpts=3.0*PTS" catch_small_model_slow.gif
        print(call2)
        try:
            os.remove(gif_name+'_temp.gif')
        except Exception as e:
            print(e)
        """
        subprocess.call(['ffmpeg', '-i', 'gifs\\%d.png', gif_name+'.gif'])

        if delete_pics:
            for i in range(c):
                try:
                    os.remove("gifs\\%d.png" % i)
                except Exception as e:
                    print(e)
