'''
Implements optimal strategy for pikomino in one turn.
We use dynamic programming to find the optimal solution.
'''

from itertools import product
from tqdm import tqdm
import torch
import numpy as np
from time import time

def init_table(reward_vect, pen, max_dice = 8, nb_dice = 8, min_score = 21, max_score = 36):
    '''
    The table has shape (accept_r, took_1, took_2, took_3, took_4, took_5, took_6, max_sum - min_sum)
    '''

    # To reduce complexity, we keep track of the minimal and maximal possible sum of dice.
    min_sum = max_dice-nb_dice
    max_sum = 6*(max_dice-nb_dice) + 1
    table = torch.zeros((2, 2, 2, 2, 2, 2, 2, max_sum-min_sum))

    # init the table, in case you pick the reward and you have picked a 6 before
    # otherwise you get the pen
    for r in range(max_sum - min_sum):
        if r < min_score-min_sum:
            table[1, :, :, :, :, :, 1, r] = pen
        elif r > max_score-min_sum:
            table[1, :, :, :, :, :, 1, r] = reward_vect[max_score-min_score]
        else:
            table[1, :, :, :, :, :, 1, r] = reward_vect[r-min_score]
        table[1, :, :, :, :, :, 0, r] = pen


    # We compute the table backward.
    if nb_dice == 0:
        # Table at position 0 is directly returned.
        table[0] += -np.inf
        return [table]
    else:
        # Otherwise we get the previous tables.
        tables = init_table(reward_vect, pen, max_dice = max_dice, nb_dice = nb_dice - 1, min_score = min_score, max_score = max_score) 
    
    # Get the symetric throws and all possible conditions.
    sym_throws, probas = get_throws(nb_dice, sym = True)
    conditions = get_throws(6, [0, 1])

    print("Compute table {}".format(nb_dice))
    for sym_throw, proba in tqdm(zip(sym_throws, probas)):
        for cond in conditions:
            for sum in range(max_sum-min_sum):

                # Computes the reward of all actions. 
                Qs = []
                for i, d in enumerate(sym_throw):

                    # Impossible actions get very negative reward.
                    if cond[i] == 1 or d == 0:
                        Qs.append(pen)

                    # Otherwise, add the condition and get the previous reward.
                    else:
                        new_cond = np.copy(cond)
                        new_cond[i] = 1
                        Qs.append(max(tables[nb_dice-d][:, *new_cond, sum + i*d]).item())

                table[0, *cond, sum] += max(Qs)*proba
    
    tables.append(table)
    return tables


def get_throws(nb_dice, dice = [1, 2, 3, 4, 5, 6], sym = False):
    '''
    Either construct the powerset of a set, or the ordering independent powerset.
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
        if sym:
            sym_throw = [throws[-1].count(roll) for roll in range(1, 1+6)]
            if sym_throw in syms:
                probas[syms.index(sym_throw)] += proba 
            else:
                syms.append(sym_throw)
                probas.append(0)
                probas[-1] += proba

    if sym:
        return syms, probas
    else:
        return throws


def throw_dice(nb_dice, seed = 42):
    rng = np.random.default_rng(seed=seed)
    throw = [rng.integers(1, 6+1) for _ in range(nb_dice)]
    sym_throw = [throw.count(roll) for roll in range(1, 1+6)]
    return sym_throw


def get_opt_action(tables, throw, pen, cond = [0,0,0,0,0,0], sum = 0, nb_dice = 8):
    Qs = []
    sym_throw = [throw.count(roll) for roll in range(1, 1+6)]
    for i, d in enumerate(sym_throw):
        if cond[i] == 1 or d == 0:
            Qs.append(pen)
        else:
            cond[i] = 1
            Qs.append(max(tables[nb_dice-d][:, *cond, sum + i*d]).item())
    print(Qs)
    return torch.argmax(torch.Tensor(Qs), dim = -1).item() + 1


t = time()
pen = -3
nb_dice = 8
tables = init_table([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=21, max_score=36)
print("Took {} second to compute.".format(time() - t))

print(get_opt_action(tables, [1, 3, 3, 3, 3, 4, 4, 5], pen, nb_dice = nb_dice))