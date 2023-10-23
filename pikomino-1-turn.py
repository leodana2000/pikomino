'''
Implements optimal strategy for pikomino in one turn.
We use dynamic programming to find the optimal solution.
'''

from itertools import product
from tqdm import tqdm
import torch
from time import time

def init_table(reward_vect, pen, max_dice = 8, nb_dice = 8, min_score = 21, max_score = 36):
    '''
    The table has shape (accept_r, took_1, took_2, took_3, took_4, took_5, took_6, max_sum - min_sum)
    '''

    min_sum = max_dice-nb_dice
    max_sum = 6*(max_dice-nb_dice) + 1
    table = torch.zeros((2, 2, 2, 2, 2, 2, 2, max_sum-min_sum))

    # init the table, in case you pick the reward and you have picked a 6 before
    # otherwise you get the pen
    for r in range(min_sum, max_sum):
        if r < min_score:
            table[1, :, :, :, :, :, 1, r-min_sum] = pen
        else:
            table[1, :, :, :, :, :, 1, r-min_sum] = min(r, max_score)
        table[1, :, :, :, :, :, 0, r-min_sum] = pen


    # We compute the table backward.
    if nb_dice == 0:
        # Table at position 0 is directly returned.
        table[0] += -100
        return [table]
    else:
        # Otherwise we get the previous tables.
        tables = init_table(reward_vect, pen, nb_dice = nb_dice - 1) 
    
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

                    # Otherwise, add the condition aand get the previous reward.
                    else:
                        cond[i] = 1
                        Qs.append(max(tables[-d][:, cond[0], cond[1], cond[2], cond[3], cond[4], cond[5], sum + (i+1)*d - d]))
                
                # Adds the weigthed reward to the.
                table[0, cond[0], cond[1], cond[2], cond[3], cond[4], cond[5], sum] += max(Qs)*proba
    
    tables.append(table)
    return tables


def get_throws(nb_dice, dice = [1, 2, 3, 4, 5, 6], proba = 6**-8, sym = False):
    '''
    Either construct the powerset of a set, or the ordering independent powerset.
    '''

    throws = []
    syms = []
    probas = []
    dices = [dice for _ in range(nb_dice)]
    
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
    
t = time()
init_table([1 for i in range(21, 36+1)], -1)
print("Took {} second to compute.".format(time() - t))