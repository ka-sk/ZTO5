import itertools
import time
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple
import metaheuristics as met
import RandomNumberGenerator as rng

# Assuming random_search and simulated_annealing are imported from main module
def compute_optimal(p: np.ndarray) -> Tuple[List[int], float]:
    """
    Brute force szukanie najbardziej optymalnego rozwiązania.
    Tylko jeżeli n <= ~10.
    Zwraca (best_sequence, best_makespan).
    """
    n, _ = p.shape
    best_seq = None
    best_val = float('inf')
    for perm in itertools.permutations(range(n)):
        val = met.total_time(list(perm), p)
        if val < best_val:
            best_val = val
            best_seq = list(perm)
    return best_seq, best_val


def run_single(method: Callable, p: np.ndarray, **kwargs) -> Dict[str, float]:
    """
    Wykonuje pojedynczy algorymt (method(p, **kwargs)) i oblicza jakość rozwiązania i czas.
    Returns dict with 'makespan' and 'time'.
    """
    start = time.perf_counter()
    seq, val = method(p, **kwargs)
    elapsed = time.perf_counter() - start
    return {'makespan': val, 'time': elapsed}


def run_experiment(
        p: np.ndarray,
        methods: Dict[str, Callable],
        runs: int = 10,
        method_kwargs: Dict[str, Dict] = None
    ) -> pd.DataFrame:
    """
    Wykonuje wielokrotnie eksperyment na tej samej instancji p.
    methods: {'RS': random_search, 'SA': simulated_annealing}
    method_kwargs: opcjonalne zmienne dla danej metody.
    Zwraca DataFrame z kolumnami: ['method', 'run', 'makespan', 'time']
    """
    records = []
    for name, method in methods.items():
        kwargs = method_kwargs.get(name, {}) if method_kwargs else {}
        for run in range(1, runs+1):
            res = run_single(method, p, **kwargs)
            records.append({'method': name, 'run': run,
                            'makespan': res['makespan'], 'time': res['time']})
    return pd.DataFrame(records)


def experiment_vary_n(
        ns: List[int],
        m: int,
        instance_generator: Callable[[int, int], np.ndarray],
        methods: Dict[str, Callable],
        runs: int = 10,
        method_kwargs: Dict[str, Dict] = None
    ) -> pd.DataFrame:
    """
    Zmieniaj liczbę zadań n według listy ns, przy stałej liczbie maszyn m.
    Dla każdej wartości n, generuj p=instance_generator(n,m), uruchom run_experiment(),
    i opcjonalnie oblicz optimum za pomocą compute_optimal dla małych n.
    Zwraca zagregowany DataFrame z kolumnami: ['n', 'method', 'run', 'makespan', 'time', 'gap'].
    gap = (makespan - optimal) / optimal
    """
    all_records = []
    try:
        for n in ns:
            p = instance_generator(n, m)
            # compute optimal if n small
            optimal_seq, optimal_val = (None, None)
            if n <= 10:
                optimal_seq, optimal_val = compute_optimal(p)
            df = run_experiment(p, methods, runs, method_kwargs)
            df['n'] = n
            if optimal_val:
                df['gap'] = (df['makespan'] - optimal_val) / optimal_val
            else:
                df['gap'] = np.nan
            all_records.append(df)
            print(f"Eksperyment zakończony dla n={n}.")
    except KeyboardInterrupt:
        print("\nPrzerwano eksperyment przez użytkownika. Zwracam wyniki częściowe...")
    except Exception as e:
        print(f"\nWystąpił błąd: {e}. Zwracam wyniki częściowe...")
    return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame(
        columns=['n', 'method', 'run', 'makespan', 'time', 'gap'])


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mając dane z eksperymentu DataFrame (with cols ['n','method','run','makespan','time','gap']),
    oblicz statystyki podusmowujące: mean, std for makespan, time, gap.
    Zwraca te statystyki w DataFrame ['n','method'].
    """
    return df.groupby(['n','method']).agg(
        mean_makespan=('makespan','mean'),
        std_makespan=('makespan','std'),
        mean_time=('time','mean'),
        std_time=('time','std'),
        mean_gap=('gap','mean'),
        std_gap=('gap','std')
    ).reset_index()

# Example instance generator: random p_ij from uniform [1,99]
def random_instance(n: int, m: int, low: int = 1, high: int = 99, Z: int = 42) -> np.ndarray:
    RNG = rng.RandomNumberGenerator(seedVaule=Z)

    # czas wykonywania dla n wierszy m kolumn
    p = np.empty([n, m])

    for zad in range(n):
        for mach in range(m):
            p[zad, mach] = RNG.nextInt(low, high)
    
    return p

# Usage example:
if __name__ == "__main__":
    methods = {'RS': met.random_search, 'SA': met.simulated_annealing}
    df_raw = experiment_vary_n([3, 4, 5, 6, 7, 8, 9, 10], 3, random_instance, methods, runs=20)
    df_summary = summarize_results(df_raw)
    df_summary.to_csv('experiment_summary.csv', index=False)
