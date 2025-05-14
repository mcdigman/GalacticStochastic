"""helpers to make various output plots"""

import matplotlib.pyplot as plt
import numpy as np


def plot_noise_spectrum_ambiguity(ifm):
    """make a plot of the difference between the upper and lower estimates of the spectrum"""
    SAET_m = ifm.SAET_m
    wc = ifm.wc
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0., hspace=0., left=0.13, top=0.99, right=0.99, bottom=0.12)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_upper.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_lower.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, SAET_m[1:, 0], 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    plt.legend(['upper estimate', 'lower estimate', 'base'])
    plt.ylim([2.e-44, 4.e-43])
    plt.xlim([3.e-4, 6.e-3])
    plt.xlabel('f (Hz)')
    plt.ylabel(r"$\langle S^{AE}_{m} \rangle$")
    plt.show()


def plot_noise_spectrum_evolve(ifm):
    """plot the evolution of the noise power spectrum with iteration"""
    SAET_m = ifm.SAET_m
    wc = ifm.wc
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0., hspace=0., left=0.13, top=0.99, right=0.99, bottom=0.12)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, (ifm.bgd.get_galactic_total().reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2).mean(axis=0).mean(axis=1)+SAET_m[1:, 0], 'k', alpha=0.3, zorder=-90)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.SAET_tots_upper[[1, 2, 3, 4], :, 1:, 0], axis=1).T, '--', alpha=0.7)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_upper.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, SAET_m[1:, 0], 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    # TODO handle if not all iterations complete
    plt.legend(['initial', '1', '2', '3', '4', 'base'])
    plt.ylim([2.e-44, 4.e-43])
    plt.xlim([3.e-4, 6.e-3])
    plt.xlabel('f (Hz)')
    plt.ylabel(r"$\langle S^{AE}_{m} \rangle$")
    plt.show()
