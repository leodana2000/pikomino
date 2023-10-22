import numpy as np

class Dice:

    def __init__(self):
        self.rng = np.random.default_rng()

    def get_roll(self, nb_dice):
        return [self.rng.integers(1, 6+1) for _ in range(nb_dice)]
    
class State:

    def __init__(self, sum : int, previous_dices : list[int], nb_dice : int):
        self.sum = sum
        self.previous_dices = previous_dices
        self.nb_dice = nb_dice
        self.fault = False

    def update(self, throw, dice):
        self.fault = not (dice in self.previous_dice)
        for roll in throw:
            if roll == dice:
                self.nb_dice -= 1
                self.sum += dice
        self.previous_dices.append(dice)

class Throw:
    pass

class Action:
    pass

def trans_state_throw(state : State, action : Action, throw : Throw):
    if action.re_roll:
        return 1/(6**state.nb_dice)
    else:
        return 1

def trans_throw_state(throw : State, action : Action, state : Throw):
    return 1