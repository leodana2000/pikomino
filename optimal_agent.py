'''
An agent that does optimal play at 1-turn pikomino.

Observation: throw, nb_dice_left, tiles=[available tiles, your tiles, their tiles]
'''

import pikomino_1_turn as p1t
import numpy as np
import torch as t

class One_turn_agent:

    def __init__(self, reward : list[float], pen : float, max_dice : int, min_score : int, max_score : int, seed : int = 42):
        self.pen = pen
        self.reward = reward
        self.max_dice = max_dice
        self.min_score = min_score
        self.max_score = max_score
        self.rng = np.random.default_rng(seed)

        print("Initializing the agent.")
        self.tables = p1t.init_table(
            reward, 
            pen,
            max_dice,
            max_dice,
            min_score, 
            max_score, 
        )
        self.cond_list = [p1t.init_cond(i) for i in range(0, 6+1)]

    def play(self) -> None:
        roll_again = True
        ind_throw = 0
        cond = 0
        nb_cond = 0
        sum = 0
        throw = []
        summary = []
        dice_left = self.max_dice

        while roll_again:
            throw = p1t.throw_dice(dice_left, self.rng)
            ind_throw += 1 
            print("Throw n°{} is {}.".format(ind_throw, throw))

            action = p1t.get_opt_action(
                throw,
                self.tables,
                self.pen,
                cond,
                nb_cond,
                self.cond_list,
                sum, 
                dice_left,
                self.max_dice,
            )
            
            sym_throw = [throw.count(roll) for roll in range(1, 1+6)]
            nb_dice = sym_throw[action[1]]
            dice = action[1]+1

            roll_again = action[0] == 0
            if roll_again:
                print("The best action is to choose the dice {}, and then re-roll.".format(dice))
            else:
                print("The best action is to choose the dice {}, and then to stop and take the tile.".format(dice))

            # Updtaing the quantities
            sum += nb_dice*dice
            cond = p1t.add_condition(self.cond_list, nb_cond, dice, cond)
            nb_cond += 1
            dice_left -= nb_dice
            summary += [dice]*nb_dice

            if dice_left == 0:
                print("We don't have any more dice to use. The final sum is {}.".format(sum))
            elif nb_cond == 6:
                print("We have used all possible numbers. The final sum is {}.".format(sum))
            else:
                print("We have still {} dices and a total sum of {}.".format(dice_left, sum))

        t_reward = t.tensor(self.reward, dtype=t.float)
        if sum < self.min_score:
            tile = None
            points = self.pen
        else:
            if p1t.contains(self.cond_list[nb_cond], cond, 6):
                ind_max = min(sum, self.max_score) - self.min_score
                tile = t.argmax(t_reward[:ind_max])+1 + self.min_score
                points = self.reward[tile - self.min_score]
                print("We selected the dices {}, for a final sum of {}. \n We select the tile {} and get {} points.".format(summary, sum, tile, points))
            else:
                print("We selected the dices {}, for a final sum of {}. \n But the dice 6 was not selected, so we get {} points.".format(summary, sum, pen))


min_score = 21
max_score = 36
reward = [float(i) for i in range(min_score, max_score+1)]
pen = 0
max_dice = 8

agent = One_turn_agent(
    reward, 
    pen, 
    max_dice,
    min_score, 
    max_score,
)

agent.play()