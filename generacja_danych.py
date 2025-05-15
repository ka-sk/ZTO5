import numpy as np
import RandomNumberGenerator as rng
import pandas as pd

# ziarno losowości
Z = 42
# liczba zadań n
n = 10
# liczba maszyn m
m = 3

RNG = rng.RandomNumberGenerator(seedVaule=Z)

# czas wykonywania dla n wierszy m kolumn
p = np.empty([n, m])

for zad in range(n):
    for mach in range(m):
        p[zad, mach] = RNG.nextInt(1, 99)

p = pd.DataFrame(p).to_csv(f'n_{n}_m_{m}.csv')
