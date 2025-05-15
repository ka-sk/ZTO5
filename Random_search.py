import pandas as pd
import numpy as np
import random

n = 10
m = 3

p = pd.read_csv(f'n_{n}_m_{m}.csv').to_numpy()

def total_time(sol:list):
    time = [0]*n
    for mach in range(m):
        for idx, task in enumerate(sol):

            time[idx] += p[task, mach]

            if mach == 1:
                time[idx] += time[idx-1]

    return time[-1]

def random_neighbour(sol):

    # losowanie skąd
    move_from = random.randint(0, len(sol))

    move_to = random.randint(0, len(sol-1)) 
    move_to = move_to + 1 if move_to>=move_from else move_to

    sol[move_from], sol[move_to] = sol[move_to], sol[move_from]

    return total_time(sol)




        