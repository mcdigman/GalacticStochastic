"""run iterative processing of galactic background"""

import configparser

import matplotlib.pyplot as plt
import numpy as np

import GalacticStochastic.global_file_index as gfi
from GalacticStochastic.iteration_config import get_iteration_config
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from LisaWaveformTools.lisa_config import get_lisa_constants
from WaveletWaveforms.wdm_config import get_wavelet_model

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('default_parameters.ini')

    galaxy_file = config['files']['galaxy_file']
    galaxy_dir = config['files']['galaxy_dir']

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    ic = get_iteration_config(config)

    galactic_below_in, _, snr_tots_in, SAET_m, _, _, snr_min_in = gfi.load_init_galactic_file(galaxy_dir, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)

    for itrm in range(0, 1):
        stat_only = False
        nt_min = 256*(7-itrm)
        nt_max = nt_min+512*(itrm+1)
        #nt_min = 0
        #nt_max = wc.Nt
        print(nt_min, nt_max, wc.Nt, wc.Nf, stat_only)

        params_gb, _, _, _, _ = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

        ifm = IterativeFitManager(lc, wc, ic, SAET_m, galactic_below_in, snr_tots_in, snr_min_in, params_gb, nt_min, nt_max, stat_only)

        params_gb = None

        ifm.do_loop()

        do_hf_out = True
        if do_hf_out:
            gfi.store_processed_gb_file(galaxy_dir, galaxy_file, ifm.wc, ifm.lc, ifm.ic, ifm.nt_min, ifm.nt_max, ifm.bgd, ic.period_list, ifm.n_bin_use, ifm.SAET_m, ifm.SAET_fin, ifm.stat_only, ifm.bis.snrs_tot_upper, ifm.n_full_converged, ifm.argbinmap, ifm.bis.faints_old, ifm.bis.faints_cur, ifm.bis.brights, snr_min_in)

plot_noise_spectrum_ambiguity = False
if plot_noise_spectrum_ambiguity:
    SAET_m_shift = SAET_m
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0., hspace=0., left=0.13, top=0.99, right=0.99, bottom=0.12)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_upper.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_lower.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, SAET_m_shift[1:, 0], 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    plt.legend(['upper estimate', 'lower estimate', 'base'])
    plt.ylim([2.e-44, 4.e-43])
    plt.xlim([3.e-4, 6.e-3])
    plt.xlabel('f (Hz)')
    plt.ylabel(r"$\langle S^{AE}_{m} \rangle$")
    plt.show()


plot_noise_spectrum_evolve = False
if plot_noise_spectrum_evolve:
    SAET_m_shift = SAET_m
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0., hspace=0., left=0.13, top=0.99, right=0.99, bottom=0.12)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, (ifm.bgd.get_galactic_total().reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2).mean(axis=0).mean(axis=1)+SAET_m[1:, 0], 'k', alpha=0.3, zorder=-90)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.SAET_tots_upper[[1, 2, 3, 4], :, 1:, 0], axis=1).T, '--', alpha=0.7)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_upper.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, SAET_m_shift[1:, 0], 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    # TODO handle if not all iterations complete
    plt.legend(['initial', '1', '2', '3', '4', 'base'])
    plt.ylim([2.e-44, 4.e-43])
    plt.xlim([3.e-4, 6.e-3])
    plt.xlabel('f (Hz)')
    plt.ylabel(r"$\langle S^{AE}_{m} \rangle$")
    plt.show()
