"""Utilities for testing code performance."""

from typing import Tuple

import numpy as np
import scipy.stats
from numpy.typing import NDArray


def unit_normal_battery(signal: NDArray[np.floating], *, mult: float = 1.0, sig_thresh: float = 5.0, A2_cut: float = 2.28, do_assert: bool = True, verbose: bool = False) -> Tuple[bool, float, float, float]:
    """Check if signal is consistent with unit normal white noise.

    Uses several tests, including an Anderston-Darling test
    and tests for zero mean and unit variance.
    default anderson darling cutoff of 2.28 is hand selected to
    give ~1 in 1e5 empirical probablity of false positive for n=64.
    The calibration is about the same for other n tested, such as n=32.
    """
    n_sig = signal.size
    if n_sig == 0:
        return True, 0.0, 0.0, 0.0

    sig_adjust = signal / mult
    mean_wave = np.mean(sig_adjust)
    std_wave = np.std(sig_adjust)
    std_std_wave: float = float(np.std(sig_adjust) * np.sqrt(2 / n_sig))

    # anderson darling test statistic assuming true mean and variance are unknown
    sig_sort = np.sort((sig_adjust - mean_wave) / std_wave)
    phis = scipy.stats.norm.cdf(sig_sort)
    xs = np.arange(1, n_sig + 1)
    A2: float = -n_sig - 1 / n_sig * np.sum((2 * xs - 1) * np.log(phis) + (2 * (n_sig - xs) + 1) * np.log(1 - phis))
    A2Star: float = A2 * (1 + 4 / n_sig - 25 / n_sig**2)
    if verbose:
        print(A2Star, A2_cut)

    mean_stat: float = float(np.abs(mean_wave) / std_wave * np.sqrt(n_sig))
    std_stat: float = float(np.abs(std_wave - 1.0) / std_std_wave)
    test1: bool = mean_stat < sig_thresh
    test2: bool = std_stat < sig_thresh
    test3: bool = bool(A2Star < A2_cut)  # should be less than cutoff value

    test_combo: bool = test1 and test2 and test3

    # check mean and variance
    if do_assert:
        assert test1
        assert test2
        assert test3

    return test_combo, A2Star, mean_stat, std_stat
