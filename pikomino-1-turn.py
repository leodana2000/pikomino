'''
Implements optimal strategy for pikomino in one turn.
We use dynamic programming to find the optimal solution.
'''

# Action space
class Action:
    SUM : int
    IS_WORMS : bool
    PREVIOUS_DICE : list[int]

