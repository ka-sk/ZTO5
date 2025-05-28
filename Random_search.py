import pandas as pd
import numpy as np
import random
import os
import sys
import copy

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
    Computes makespan for a flowshop with 3 machines given a job sequence.
    Uses dynamic programming: C[j,k] = completion time of job j on machine k.
    """
    n, m = p.shape
    if len(sequence) != n:
        raise ValueError(f"Sequence length {len(sequence)} does not match number of jobs {n}")
    # Initialize completion matrix
    C = np.zeros((n + 1, m + 1))  # extra zero row/col for easier indexing
    # Compute
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


def random_search(p: np.ndarray, max_iter: int = 1000) -> tuple:
    """
    Random Search (RS) algorithm for flowshop makespan minimization.
    Returns best sequence and its makespan.
    """
    n, m = p.shape
    # Initial random solution
    current = list(range(n))
    random.shuffle(current)
    best = current.copy()
    best_val = total_time(best, p)
    for it in range(max_iter):
        neighbour = random_neighbour(current)
        val = total_time(neighbour, p)
        if val < total_time(current, p):
            current = neighbour
            if val < best_val:
                best = neighbour
                best_val = val
    return best, best_val


def descending_search(p: np.ndarray, max_iter: int = 1000) -> tuple:
    """
    Descending Search (DS) algorithm (steepest descent) for flowshop makespan minimization.
    Explores full neighbourhood and moves to best.
    """
    n, m = p.shape
    # Initial solution (random)
    current = list(range(n))
    random.shuffle(current)
    current_val = total_time(current, p)
    best, best_val = current, current_val
    for it in range(max_iter):
        improved = False
        best_neighbour = None
        best_neighbour_val = current_val
        # explore all swaps
        for i in range(n-1):
            for j in range(i+1, n):
                neighbour = current.copy()
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                val = total_time(neighbour, p)
                if val < best_neighbour_val:
                    best_neighbour_val = val
                    best_neighbour = neighbour
        if best_neighbour is not None and best_neighbour_val < current_val:
            current = best_neighbour
            current_val = best_neighbour_val
            if current_val < best_val:
                best, best_val = current.copy(), current_val
            improved = True
        if not improved:
            break  # local optimum
    return best, best_val


def main():
    # Example usage: python script.py n_10_m_3.csv 1000
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file> [max_iter]")
        sys.exit(1)
    filename = sys.argv[1]
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    try:
        p = read_processing_times(filename)
    except Exception as e:
        print(e)
        sys.exit(1)
    print("Running Random Search...")
    rs_seq, rs_val = random_search(p, max_iter)
    print(f"RS best makespan: {rs_val}, sequence: {rs_seq}")
    print("Running Descending Search...")
    ds_seq, ds_val = descending_search(p, max_iter)
    print(f"DS best makespan: {ds_val}, sequence: {ds_seq}")

if __name__ == "__main__":
    main()
