'''
Implements optimal strategy for pikomino in one turn.
We use dynamic programming to find the optimal solution.
'''

# Action space
class params:
    SUM : int
    PREVIOUS_DICE : list[int]
    IS_WORMS = 6 in PREVIOUS_DICE

