'''
Implements optimal strategy for pikomino in one turn.
We use dynamic programming to find the optimal solution.
'''

from itertools import product
from tqdm import tqdm
import torch
import numpy as np
import math
from time import time

def init_table(reward_vect, pen, max_dice = 8, nb_dice = 8, 
               min_score = 21, max_score = 36, nb_cond = 0, 
               cond_list = None, limit=False):
    '''
    The table has shape (accept_r, (6 nb_cond), max_sum-min_sum)
    '''

    # To reduce complexity, we keep track of the minimal and maximal possible sum of dice.
    dice_used = max_dice-nb_dice
    min_sum = dice_used
    max_sum = 6*(dice_used) + 1
    cond_range = math.comb(6, nb_cond)
    ind_cond = nb_cond-1
    table = torch.zeros((2, cond_range, max_sum-min_sum))

    # When initiating the function, we first compute all the condition tables.
    if dice_used == 0:
        cond_list = [init_cond(i) for i in range(0, 6+1)]

    # Init the table, in case you pick the reward and you have picked a 6 before.
    # Otherwise you get the pen.
    for r in range(max_sum - min_sum):
        for cond in range(cond_range):
            if contains(cond_list[nb_cond], cond, 6):
                if r < min_score-min_sum:
                    table[1, cond, r] = pen
                elif r > max_score-min_sum:
                    if limit:
                        table[1, cond, r] = pen
                    else:
                        table[1, cond, r] = reward_vect[max_score-min_score]
                else:
                    table[1, cond, r] = reward_vect[r-min_score]
            else:
                table[1, cond, r] = pen

    # We initialize the tables backward in order of decresing and nb_dice and then decreasing nb_cond.
    if nb_cond <= 1:
        if nb_dice == 0:
            # If there is not dice left to play and one condition, then start the tables.
            table[0] += pen
            return [[table]]
        
        else:
            # Otherwise, we just hit the minimum number of conditions, so we go back to max conditions with one dice lower.
            tables = init_table(reward_vect, pen, max_dice = max_dice, nb_dice = nb_dice - 1, 
                                min_score = min_score, max_score = max_score, 
                                nb_cond = min(6, dice_used+1) ,cond_list=cond_list) 
    
    else:
        # Otherwise, we ask for one less conditions.
        tables = init_table(reward_vect, pen, max_dice = max_dice, nb_dice = nb_dice, 
                            min_score = min_score, max_score = max_score, 
                            nb_cond = nb_cond-1, cond_list=cond_list) 
        
        if nb_dice == 0 or nb_cond == 6:
            # And in the case of 0 nb_dice or 6 nb_cond, we don't have to compute the table since we cannot throw dice. 
            table[0] += pen
            tables[nb_dice].append(table)
            return tables


    ## Now, we compute the table backward.


    # Get the symetric throws and all possible conditions.
    sym_throws, probas = get_throws(nb_dice)

    for sym_throw, proba in zip(sym_throws, probas):
        for cond in range(cond_range):
            for sum in range(max_sum-min_sum):

                # Computes the reward of all actions. 
                Qs = []
                for i, nb_d in enumerate(sym_throw):
                    dice = i+1

                    # Impossible actions get very negative reward.
                    if contains(cond_list[nb_cond], cond, dice) or nb_d == 0:
                        Qs.append(pen)

                    # Otherwise, add the condition and get the previous reward.
                    else:
                        new_cond = add_condition(cond_list, nb_cond, dice, cond)
                        Qs.append(max(tables[nb_dice-nb_d][ind_cond+1][:, new_cond, sum + nb_d*dice - nb_d]).item())

                table[0, cond, sum] += max(Qs)*proba
    
    # If we hit the minimal number of conditions, we need to initiate the list at nb_dice.
    if nb_cond <= 1:
        tables.append([table])
    else:
        tables[nb_dice].append(table)
    return tables


def contains(cond_list, cond, dice):
    '''
    Returns if the dice we want to choose was already picked.
    '''
    return dice in cond_list[cond]


def add_condition(cond_list, nb_cond, dice, cond):
    '''
    Transform the condition number with nb_cond, to a condition number with nb_cond+1 conditions.
    '''
    next_cond = cond_list[nb_cond+1]

    # Finds what is the list of dice corresponding to that condition. !in-place operation!
    dice_cond = list(np.copy(cond_list[nb_cond][cond]))

    # Adds the current dice at the right place in the condition.
    flag = True
    for i, d in enumerate(dice_cond):
        if d<dice and flag:
            dice_cond.insert(i, dice)
            flag = False
            break
    if flag:
        dice_cond.append(dice)

    # Finds the indice of this new condition.
    return next_cond.index(dice_cond)


def init_cond(nb_cond, dices = [1, 2, 3, 4, 5, 6]):
    '''
    A function that initializes the list of conditions: all decreasingly ordered tuples of nb_cond dices.
    '''

    cond_list = []

    if nb_cond == 0:
        # No condition, no dices.
        cond_list.append([])
        return cond_list
    
    for i, dice in enumerate(dices):
        if len(dices[i+1:]) >= nb_cond-1:
            pre_list = init_cond(nb_cond-1, dices = dices[i+1:])
            for pre in pre_list:
                pre.append(dice)
                cond_list.append(pre)
    return cond_list


def get_throws(nb_dice, dice = [1, 2, 3, 4, 5, 6]):
    '''
    Construct the powerset of a set and then order into independent powerset.
    '''

    throws = []
    syms = []
    probas = []
    dices = [dice for _ in range(nb_dice)]
    proba = 6**-nb_dice
    
    for d in product(*dices):
        # Enumerate all throws
        throws.append(list(d))

        # Keep only the meaningfully different throws: permutation indepenent.
        sym_throw = [throws[-1].count(roll) for roll in range(1, 1+6)]
        if sym_throw in syms:
            probas[syms.index(sym_throw)] += proba 
        else:
            syms.append(sym_throw)
            probas.append(0)
            probas[-1] += proba

    return syms, probas


def throw_dice(nb_dice, seed = 42):
    '''
    Computes a random sample of nb_dice, and return the symetric throw form.
    '''
    rng = np.random.default_rng(seed=seed)
    throw = [rng.integers(1, 6+1) for _ in range(nb_dice)]
    sym_throw = [throw.count(roll) for roll in range(1, 1+6)]
    return sym_throw


def get_opt_action(tables, throw, cond, nb_cond, pen, sum = 0, nb_dice = 8):
    '''
    Given a throw, nb_dice and a condition, return what is the best dice to pick.
    '''
    sym_throw = [throw.count(roll) for roll in range(1, 1+6)]
    cond_list = [init_cond(i) for i in range(0, 6+1)]
    ind_cond = nb_cond-1

    Qs = []
    for i, nb_d in enumerate(sym_throw):
        dice = i+1

        # Impossible actions get very negative reward.
        if contains(cond_list[nb_cond], cond, dice) or nb_d == 0:
            Qs.append(pen)

        # Otherwise, add the condition and get the previous reward.
        else:
            new_cond = add_condition(cond_list, nb_cond, dice, cond)
            Qs.append(max(tables[nb_dice-nb_d][ind_cond+1][:, new_cond, sum + nb_d*dice - nb_d]).item())

    print(Qs)
    return torch.argmax(torch.Tensor(Qs), dim = -1).item() + 1