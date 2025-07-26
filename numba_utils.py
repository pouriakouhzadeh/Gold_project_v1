from numba import njit
import numpy as np

@njit
def numba_skew(x):
    n = len(x)
    if n == 0:
        return np.nan
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    diff = x - m
    return np.mean(diff**3) / (s**3)

@njit
def numba_kurtosis(x):
    n = len(x)
    if n == 0:
        return np.nan
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    diff = x - m
    return np.mean(diff**4) / (s**4) - 3.0

@njit
def numba_median(x):
    arr = np.sort(x.copy())
    n = len(arr)
    if n % 2 == 1:
        return arr[n // 2]
    else:
        return (arr[n // 2 - 1] + arr[n // 2]) / 2.0

@njit
def numba_up_count(x):
    count = 0
    for i in range(1, len(x)):
        if x[i] - x[i - 1] > 0:
            count += 1
    return count

@njit
def numba_last_local_max_idx(x):
    return len(x) - np.argmax(x) - 1

@njit
def numba_last_local_min_idx(x):
    return len(x) - np.argmin(x) - 1
