import pandas as pd
import numpy as np
import random
import os
import sys
import copy
import matplotlib.pyplot as plt

from typing import List, Tuple


def read_processing_times(filename: str) -> np.ndarray:
    """
    Reads processing times matrix from a CSV file.
    The CSV should have m columns and n rows.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    try:
        df = pd.read_csv(filename, header=None)
        return df.to_numpy()
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def total_time(sequence: list, p: np.ndarray) -> float:
    """
    Computes makespan for a flowshop given a job sequence.
    """
    n, m = p.shape
    if len(sequence) != n:
        raise ValueError(f"Sequence length {len(sequence)} does not match number of jobs {n}")
    C = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        job = sequence[i-1]
        for k in range(1, m + 1):
            C[i, k] = max(C[i-1, k], C[i, k-1]) + p[job, k-1]
    return C[n, m]


def random_neighbour(sol: list) -> list:
    """
    Generates a neighbour by swapping two random positions in the sequence.
    """
    if len(sol) < 2:
        return sol
    i, j = random.sample(range(len(sol)), 2)
    neighbour = sol.copy()
    neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
    return neighbour


def random_search(p: np.ndarray,
                  max_iter: int = 1000,
                  plot_history: bool = False) -> Tuple[List[int], float]:
    """
    Random Search (RS) algorithm for flowshop makespan minimization.
    Records best makespan at each iteration and optionally plots the history.
    Returns best sequence and its makespan.

    :param p: processing time matrix
    :param max_iter: number of iterations
    :param plot_history: if True, will display a plot of best makespan vs iteration
    """
    n, _ = p.shape
    current = list(range(n))
    random.shuffle(current)
    best = current.copy()
    best_val = total_time(best, p)
    history = [best_val]
    
    for iteration in range(1, max_iter + 1):
        neighbour = random_neighbour(current)
        cur_val = total_time(current, p)
        n_val = total_time(neighbour, p)
        if n_val < cur_val:
            current = neighbour
            if n_val < best_val:
                best, best_val = neighbour.copy(), n_val
        history.append(best_val)
    
    if plot_history:
        plt.figure()
        plt.plot(range(len(history)), history)
        plt.xlabel('Iteration')
        plt.ylabel('Best makespan')
        plt.title('Random Search: Najlepsze rozwiązanie od liczby iteracji')
        plt.savefig(f"RS_{p.shape}.png")
        plt.close()
    
    return best, best_val


def simulated_annealing(p: np.ndarray,
                        max_iter: int = 10000,
                        initial_temp: float = None,
                        alpha: float = 0.995,
                        epoch_length: int = 100,
                        stagnation_threshold: int = 1000,
                        plot_history: bool = False) -> Tuple[List[int], float]:
    """
    Simulated Annealing (SA) for flowshop makespan minimization.
    Records best makespan at each epoch and optionally plots the history.
    Returns best sequence and its makespan.
    """
    n, _ = p.shape
    current = list(range(n))
    random.shuffle(current)
    cur_val = total_time(current, p)
    best, best_val = current.copy(), cur_val
    
    # initial temperature estimation
    if initial_temp is None:
        samples = []
        for _ in range(1000):
            s = random_neighbour(current)
            samples.append(total_time(s, p))
        temp_range = max(samples) - min(samples)
        initial_temp = temp_range if temp_range > 0 else 1.0
    t = initial_temp
    no_improve = 0
    history = [best_val]
    
    for iteration in range(1, max_iter + 1):
        for _ in range(epoch_length):
            neighbour = random_neighbour(current)
            n_val = total_time(neighbour, p)
            delta = n_val - cur_val
            if delta < 0 or random.random() < np.exp(-delta / t):
                current, cur_val = neighbour, n_val
                if cur_val < best_val:
                    best, best_val = current.copy(), cur_val
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            if no_improve >= stagnation_threshold:
                t = initial_temp
                no_improve = 0
        t *= alpha
        history.append(best_val)
        if t <= 1e-8:
            break
    
    if plot_history:
        plt.figure()
        plt.plot(range(len(history)), history)
        plt.xlabel('Epoch')
        plt.ylabel('Best makespan')
        plt.title('Simulated Annealing: Najlepsze rozwiązanie od liczby iteracji')
        plt.savefig(f"SA_{p.shape}.png")
        plt.close()
    
    return best, best_val


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file> [max_iter] [plot]")
        sys.exit(1)
    filename = sys.argv[1]
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    plot_flag = sys.argv[3].lower() == 'plot' if len(sys.argv) > 3 else False
    
    try:
        p = read_processing_times(filename)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    print("Running Random Search...")
    rs_seq, rs_val = random_search(p, max_iter, plot_history=plot_flag)
    print(f"RS best makespan: {rs_val}, sequence: {rs_seq}")
    
    print("Running Simulated Annealing...")
    sa_seq, sa_val = simulated_annealing(p, max_iter, plot_history=plot_flag)
    print(f"SA best makespan: {sa_val}, sequence: {sa_seq}")

if __name__ == "__main__":
    main()
