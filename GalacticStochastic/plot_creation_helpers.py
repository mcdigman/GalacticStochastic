"""helpers to make various output plots"""

import matplotlib.pyplot as plt
import numpy as np

from GalacticStochastic.iterative_fit_manager import IterativeFitManager


def plot_noise_spectrum_ambiguity(ifm: IterativeFitManager) -> None:
    """Make a plot of the difference between the upper and lower estimates of the spectrum"""
    wc = ifm.noise_manager.wc
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.13, top=0.99, right=0.99, bottom=0.12)

    fs = np.arange(1, wc.Nf) * wc.DF
    S_stat_m = ifm.noise_manager.S_inst_m
    S_upper_m = np.mean(ifm.noise_manager.noise_upper.S[:, 1:, 0:2], axis=0).mean(axis=1).T
    S_lower_m = np.mean(ifm.noise_manager.noise_lower.S[:, 1:, 0:2], axis=0).mean(axis=1).T

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
    """Plot the evolution of the noise power spectrum with iteration"""
    S_stat_m = ifm.noise_manager.S_inst_m
    wc = ifm.noise_manager.wc
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.13, top=0.99, right=0.99, bottom=0.12)
    _ = ax.loglog(
        np.arange(1, wc.Nf) * wc.DF,
        (
            ifm.noise_manager.bgd.get_galactic_total().reshape((wc.Nt, wc.Nf, ifm.noise_manager.bgd.nc_galaxy))[
                :, 1:, 0:2,
            ]
            ** 2
        )
        .mean(axis=0)
        .mean(axis=1)
        + S_stat_m[1:, 0],
        'k',
        alpha=0.3,
        zorder=-90,
    )
    _ = ax.loglog(
        np.arange(1, wc.Nf) * wc.DF,
        np.mean(ifm.noise_manager.S_record_upper[[1, 2, 3, 4], :, 1:, 0], axis=1).T,
        '--',
        alpha=0.7,
    )
    _ = ax.loglog(np.arange(1, wc.Nf) * wc.DF, np.mean(ifm.noise_manager.noise_upper.S[:, 1:, 0:2], axis=0).mean(axis=1).T)
    _ = ax.loglog(np.arange(1, wc.Nf) * wc.DF, S_stat_m[1:, 0], 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    _ = plt.legend(labels=['initial', '1', '2', '3', '4', 'base'])
    _ = plt.ylim([2.0e-44, 4.0e-43])
    _ = plt.xlim([3.0e-4, 6.0e-3])
    _ = plt.xlabel('f (Hz)')
    _ = plt.ylabel(r'$\langle S^{AE}_{m} \rangle$')
    plt.show()
