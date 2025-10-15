"""Make various output plots for the iterative fitting process."""

import matplotlib.pyplot as plt
import numpy as np

from GalacticStochastic.iterative_fit_manager import IterativeFitManager


def plot_noise_spectrum_ambiguity(ifm: IterativeFitManager) -> None:
    """Make a plot of the difference between the upper and lower estimates of the spectrum.

    Parameters
    ----------
    ifm: IterativeFitManager
        Iterative fit manager with results to plot.
    """
    wc = ifm.noise_manager.wc
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.13, top=0.99, right=0.99, bottom=0.12)

    fs = np.arange(1, wc.Nf) * wc.DF
    S_stat_m = ifm.noise_manager.S_inst_m
    S_upper_m = np.mean(ifm.noise_manager.noise_upper.get_S()[:, 1:, 0:2], axis=0).mean(axis=1).T
    S_lower_m = np.mean(ifm.noise_manager.noise_lower.get_S()[:, 1:, 0:2], axis=0).mean(axis=1).T

    _ = ax.loglog(fs, S_upper_m)
    _ = ax.loglog(fs, S_lower_m)
    _ = ax.loglog(fs, S_stat_m[1:, 0], 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    _ = plt.legend(labels=['upper estimate', 'lower estimate', 'base'])
    _ = plt.ylim([2.0e-44, 4.0e-43])
    _ = plt.xlim([3.0e-4, 6.0e-3])
    _ = plt.xlabel('f (Hz)')
    _ = plt.ylabel(r'$\langle S^{AE}_{m} \rangle$')
    plt.show()


def plot_noise_spectrum_evolve(ifm: IterativeFitManager) -> None:
    """Plot the evolution of the noise power spectrum with iteration.

    Parameters
    ----------
    ifm: IterativeFitManager
        Iterative fit manager with results to plot.
    """
    S_stat_m = ifm.noise_manager.S_inst_m
    wc = ifm.noise_manager.wc
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.13, top=0.99, right=0.99, bottom=0.12)

    f = np.arange(1, wc.Nf) * wc.DF

    S_AE_total = ifm.noise_manager.bgd.get_galactic_total(shape_mode=1)[:, 1:, 0:2] ** 2
    S_AE_total_mean = np.mean(S_AE_total, axis=(0, 2))
    del S_AE_total

    S_AE_slices = np.mean(ifm.noise_manager.S_record_upper[[1, 2, 3, 4], :, 1:, 0:2], axis=(1, 3))

    S_AE_final = np.mean(ifm.noise_manager.noise_upper.get_S()[:, 1:, 0:2], axis=(0, 2))

    S_AE_inst = S_stat_m[1:, 0:2].mean(axis=1)

    _ = ax.loglog(f, 2 * wc.dt * (S_AE_total_mean + S_AE_inst), 'k', alpha=0.3, zorder=-90)
    _ = ax.loglog(f, 2 * wc.dt * S_AE_slices.T, '--', alpha=0.7)
    _ = ax.loglog(f, 2 * wc.dt * S_AE_final.T)
    _ = ax.loglog(f, 2 * wc.dt * S_AE_inst, 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    _ = plt.legend(labels=['initial', '1', '2', '3', '4', 'base'])
    # _ = plt.ylim([2.0e-44, 4.0e-43])
    _ = plt.ylim([1.e-43, 2.0e-39])
    _ = plt.xlim([1.0e-4, 6.0e-3])
    _ = plt.xlabel('f (Hz)')
    _ = plt.ylabel(r'$\langle S^{AE}_{m} \rangle$')
    plt.show()
