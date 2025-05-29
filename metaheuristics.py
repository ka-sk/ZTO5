# main_algorithm.py

import pandas as pd
import numpy as np
import random
import math
from typing import List, Tuple

def read_processing_times(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, header=None)
    return df.to_numpy()

def total_time(sequence: list, p: np.ndarray) -> float:
    n, m = p.shape
    C = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        job = sequence[i - 1]
        for k in range(1, m + 1):
            C[i, k] = max(C[i - 1, k], C[i, k - 1]) + p[job, k - 1]
    return C[n, m]

def swap_neighbour(sol: list) -> list:
    i, j = random.sample(range(len(sol)), 2)
    neighbour = sol.copy()
    neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
    return neighbour

def insert_neighbour(sol: list) -> list:
    i, j = random.sample(range(len(sol)), 2)
    neighbour = sol.copy()
    job = neighbour.pop(i)
    neighbour.insert(j, job)
    return neighbour

def initial_solution(n: int, method: str = "random", p: np.ndarray = None, k: int = 10) -> List[int]:
    if method == "random":
        sol = list(range(n))
        random.shuffle(sol)
        return sol
    elif method == "sorted_sum" and p is not None:
        return sorted(range(n), key=lambda i: sum(p[i]))
    elif method == "best_of_k" and p is not None:
        best = None
        best_cost = float("inf")
        for _ in range(k):
            sol = list(range(n))
            random.shuffle(sol)
            cost = total_time(sol, p)
            if cost < best_cost:
                best = sol
                best_cost = cost
        return best
    else:
        raise ValueError("Unknown initial solution method or missing data.")

def random_search(p: np.ndarray, iterations: int = 1000,
                  neighbour_func=swap_neighbour,
                  init_method: str = "random") -> Tuple[List[int], float]:
    n, _ = p.shape
    current = initial_solution(n, init_method, p)
    best = current.copy()
    best_val = total_time(best, p)

    for _ in range(iterations):
        neighbour = neighbour_func(current)
        n_val = total_time(neighbour, p)
        if n_val < best_val:
            best, best_val = neighbour.copy(), n_val
            current = neighbour
    return best, best_val

def simulated_annealing(p: np.ndarray,
                        iterations: int = 10000,
                        initial_temp: float = None,
                        alpha: float = 0.995,
                        neighbour_func=swap_neighbour,
                        init_method: str = "random") -> Tuple[List[int], float]:
    n, _ = p.shape
    current = initial_solution(n, init_method, p)
    cur_val = total_time(current, p)
    best = current.copy()
    best_val = cur_val

    if initial_temp is None:
        samples = [total_time(insert_neighbour(current), p) for _ in range(100)]
        initial_temp = max(samples) - min(samples) or 1.0

    t = initial_temp
    for _ in range(iterations):
        neighbour = neighbour_func(current)
        n_val = total_time(neighbour, p)
        delta = n_val - cur_val
        if delta < 0 or random.random() < math.exp(-delta / t):
            current, cur_val = neighbour, n_val
            if cur_val < best_val:
                best, best_val = current.copy(), cur_val
        t *= alpha
        if t < 1e-8:
            break

    return best, best_val
