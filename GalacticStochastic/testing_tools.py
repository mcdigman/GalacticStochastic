"""utilities for testing the code"""
import numpy as np
import scipy.stats


def unit_normal_battery(signal, mult=1., sig_thresh=5., A2_cut=2.28, do_assert=True, verbose=False):
    """Battery of tests for checking if signal is unit normal white noise
    default anderson darling cutoff of 2.28 is hand selected to
    give ~1 in 1e5 empirical probablity of false positive for n=64
    calibration looks about same for n=32 could probably choose better way
    with current defaults that should make it the most sensitive test
    """
    n_sig = signal.size
    if n_sig == 0:
        return True, 0., 0., 0.

    sig_adjust = signal / mult
    mean_wave = np.mean(sig_adjust)
    std_wave = np.std(sig_adjust)
    std_std_wave = np.std(sig_adjust) * np.sqrt(2 / n_sig)

    # anderson darling test statistic assuming true mean and variance are unknown
    sig_sort = np.sort((sig_adjust - mean_wave) / std_wave)
    phis = scipy.stats.norm.cdf(sig_sort)
    xs = np.arange(1, n_sig + 1)
    A2 = -n_sig - 1 / n_sig * np.sum((2 * xs - 1) * np.log(phis) + (2 * (n_sig - xs) + 1) * np.log(1 - phis))
    A2Star = A2 * (1 + 4 / n_sig - 25 / n_sig**2)
    if verbose:
        print(A2Star, A2_cut)

    mean_stat = np.abs(mean_wave) / std_wave * np.sqrt(n_sig)
    std_stat = np.abs(std_wave - 1.) / std_std_wave
    test1 = mean_stat < sig_thresh
    test2 = std_stat < sig_thresh
    test3 = A2Star < A2_cut  # should be less than cutoff value

    # check mean and variance
    if do_assert:
        assert test1
        assert test2
        assert test3

    return test1 and test2 and test3, A2Star, mean_stat, std_stat
