"""Utilities for testing code performance."""

import numpy as np
import scipy.stats
from numpy.typing import NDArray


def unit_normal_battery(
    signal: NDArray[np.floating],
    *,
    mult: float = 1.0,
    sig_thresh: float = 5.0,
    a2_cut: float = 2.28,
    do_assert: bool = True,
    verbose: bool = False,
) -> tuple[bool, float, float, float]:
    """
    Test if a signal is consistent with unit normal white noise.

    This function applies several statistical tests to determine if the input signal
    behaves like unit normal (mean 0, variance 1) white noise. It uses the Anderson-Darling
    test for normality, and checks for zero mean and unit variance. The default Anderson-Darling
    cutoff of 2.26 is hand selected give ~1 in 1e5 empirical probablity of false positive for n=64.
    The calibration is about the same for other n tested, such as n=32.

    Parameters
    ----------
    signal : NDArray[np.floating]
        The input signal array to be tested.
    mult : float
        Scaling factor applied to the signal before testing (default is 1.0).
    sig_thresh : float
        Threshold for the mean and standard deviation tests (default is 5.0).
    a2_cut : float
        Anderson-Darling test cutoff value (default is 2.28).
    do_assert : bool
        If True, assertions are raised if any test fails (default is True).
    verbose : bool
        If True, prints the Anderson-Darling statistic and cutoff (default is False).

    Returns
    -------
    test_combo : bool
        True if all tests are passed, False otherwise.
    a2_star : float
        Anderson-Darling test statistic (bias-corrected).
    mean_stat : float
        Normalized mean test statistic.
    std_stat : float
        Normalized standard deviation test statistic.
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
    a2: float = -n_sig - 1 / n_sig * np.sum((2 * xs - 1) * np.log(phis) + (2 * (n_sig - xs) + 1) * np.log(1 - phis))
    a2_star: float = a2 * (1 + 4 / n_sig - 25 / n_sig**2)
    if verbose:
        print(a2_star, a2_cut)

    mean_stat: float = float(np.abs(mean_wave) / std_wave * np.sqrt(n_sig))
    std_stat: float = float(np.abs(std_wave - 1.0) / std_std_wave)
    test1: bool = mean_stat < sig_thresh
    test2: bool = std_stat < sig_thresh
    test3: bool = bool(a2_star < a2_cut)  # should be less than cutoff value

    test_combo: bool = test1 and test2 and test3

    # check mean and variance
    if do_assert:
        assert test1
        assert test2
        assert test3

    return test_combo, a2_star, mean_stat, std_stat
