import pikomino_1_turn as p1t
from time import time

# Test 1: time to compute a table.
t = time()
nb_dice = 8
max_dice = 8
min_score = 21
max_score = 36
pen = 0
sum = 0
cond = 0
nb_cond = 0
cond_list = [p1t.init_cond(i) for i in range(0, 6+1)]

tables_0 = p1t.init_table([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=min_score, max_score=max_score)
print("Took {} second to compute.".format(time() - t)) # Answer: 64s

throw = [1, 3, 3, 3, 3, 4, 4, 5]
print("The optimal action for pen={}, and throw {}, is {}.".format(pen, throw, p1t.get_opt_action(throw, tables_0, pen, cond, nb_cond, cond_list, sum, nb_dice, max_dice))) # Answer: 3
del tables_0

pen = -3
tables_m3 = p1t.init_table([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=min_score, max_score=max_score)
print("The optimal action for pen={}, and throw {}, is {}.".format(pen, throw, p1t.get_opt_action(throw, tables_m3, pen, cond, nb_cond, cond_list, sum, nb_dice, max_dice))) # Answer: 3
del tables_m3

pen = 0
table_1 = p1t.init_table([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=min_score, max_score=max_score)
proba_p_1 = table_1[nb_dice][nb_cond][0, cond, 0]
print("The probability to obtain a pikomino with a value 1 or higher is {}.".format(proba_p_1)) # Answer: 0.92
del table_1

table_2 = p1t.init_table([0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=min_score, max_score=max_score)
proba_p_2 = table_2[nb_dice][nb_cond][0, cond, 0]
print("The probability to obtain a pikomino with a value 2 or higher is {}.".format(proba_p_2)) # Answer: 0.89
del table_2

table_3 = p1t.init_table([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=min_score, max_score=max_score)
proba_p_3 = table_3[nb_dice][nb_cond][0, cond, 0]
print("The probability to obtain a pikomino with a value 3 or higher is {}.".format(proba_p_3)) # Answer: 0.88
del table_3

table_4 = p1t.init_table([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=min_score, max_score=max_score)
proba_p_4 = table_4[nb_dice][nb_cond][0, cond, 0]
print("The probability to obtain a pikomino with a value 4 is {}.".format(proba_p_4))           # Answer: 0.81
del table_4

tile_24 = p1t.init_table([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=min_score, max_score=max_score, limit=True)
p_24 = tile_24[nb_dice][nb_cond][0, cond, 0]
print("The probability to obtain exactly tile 24 is {}.".format(p_24))                          # Answer: 0.19
del tile_24

tile_27_plus = p1t.init_table([0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1], pen, max_dice = nb_dice, nb_dice = nb_dice, min_score=min_score, max_score=max_score)
p_27p = tile_27_plus[nb_dice][nb_cond][0, cond, 0]
print("The probability to obtain exactly tile 24 is {}.".format(p_27p))                         # Answer: 0.88
del tile_27_plus

print("Took {} second to compute this file.".format(time() - t))