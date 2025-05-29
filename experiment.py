# badania.py

import pandas as pd
import numpy as np
import time
import itertools
from typing import Callable, Dict, List, Tuple
import main_algorithm as alg

def random_instance(n: int, m: int, low=1, high=99, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=(n, m))

def compute_optimal(p: np.ndarray) -> Tuple[List[int], float]:
    n = len(p)
    best_seq = None
    best_val = float("inf")
    for perm in itertools.permutations(range(n)):
        val = alg.total_time(list(perm), p)
        if val < best_val:
            best_val = val
            best_seq = list(perm)
    return best_seq, best_val

def run_single(method: Callable, p: np.ndarray, **kwargs) -> Dict[str, float]:
    start = time.perf_counter()
    _, val = method(p, **kwargs)
    elapsed = time.perf_counter() - start
    return {'makespan': val, 'time': elapsed}

def run_experiment(p: np.ndarray, methods: Dict[str, Callable], runs=10, method_kwargs=None) -> pd.DataFrame:
    records = []
    for name, method in methods.items():
        kwargs = method_kwargs.get(name, {}) if method_kwargs else {}
        for run in range(1, runs + 1):
            res = run_single(method, p, **kwargs)
            records.append({
                'method': name,
                'run': run,
                'makespan': res['makespan'],
                'time': res['time']
            })
    return pd.DataFrame(records)

def experiment_vary_n(ns: List[int], m: int, instance_generator, methods, runs=10, method_kwargs=None) -> pd.DataFrame:
    all_records = []
    for n in ns:
        p = instance_generator(n, m)
        optimal_val = None
        if n <= 10:
            _, optimal_val = compute_optimal(p)
        df = run_experiment(p, methods, runs, method_kwargs)
        df['n'] = n
        df['gap'] = (df['makespan'] - optimal_val) / optimal_val if optimal_val else np.nan
        all_records.append(df)
        print(f"Zakończono eksperyment dla n = {n}")
    return pd.concat(all_records, ignore_index=True)

def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['n', 'method']).agg(
        mean_makespan=('makespan', 'mean'),
        std_makespan=('makespan', 'std'),
        mean_time=('time', 'mean'),
        std_time=('time', 'std'),
        mean_gap=('gap', 'mean'),
        std_gap=('gap', 'std'),
    ).reset_index()

if __name__ == "__main__":
    methods = {
        'RS': alg.random_search,
        'SA': alg.simulated_annealing
    }

    kwargs = {
        'RS': {'neighbour_func': alg.swap_neighbour, 'init_method': 'random'},
        'SA': {'neighbour_func': alg.insert_neighbour, 'init_method': 'best_of_k', 'alpha': 0.99}
    }

    df_raw = experiment_vary_n(list(range(3, 11)), 3, random_instance, methods, runs=20, method_kwargs=kwargs)
    df_summary = summarize_results(df_raw)
    df_summary.to_csv('experiment_summary.csv', index=False)
    print("\nEksperyment zakończony. Wyniki zapisane w 'experiment_summary.csv'.")
