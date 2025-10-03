"""make plot comparing galactic background noise spectra with and without cyclostationary model"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.special
import scipy.stats
from numpy.typing import NDArray

import GalacticStochastic.global_const as gc
from GalacticStochastic import config_helper
from GalacticStochastic.galactic_fit_helpers import get_S_cyclo
from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5


def result_normality_battery(nf_min_in: int, nf_max_in: int, signal_in: NDArray[np.floating], S_in: NDArray[np.floating]) -> NDArray[np.floating]:
    ns = signal_in[:, nf_min_in:nf_max_in, 0:2].size
    signal_white_out = signal_in[:, nf_min_in:nf_max_in, 0:2] / np.sqrt(S_in[:, nf_min_in:nf_max_in, 0:2])
    normal_res = scipy.stats.normaltest(signal_white_out.flatten())
    normal_resa = scipy.stats.normaltest(signal_white_out[:, :, 0].flatten())
    normal_rese = scipy.stats.normaltest(signal_white_out[:, :, 1].flatten())
    normal_resf = scipy.stats.normaltest(signal_white_out.mean(axis=0).mean(axis=1).flatten())
    normal_resfa = scipy.stats.normaltest(signal_white_out[:, :, 0].mean(axis=0).flatten())
    normal_resfe = scipy.stats.normaltest(signal_white_out[:, :, 1].mean(axis=0).flatten())
    normal_rest = scipy.stats.normaltest(signal_white_out.mean(axis=1).mean(axis=1).flatten())
    signal_white_outf = (signal_in[:, nf_min_in:nf_max_in, 0] / np.std(signal_in[:, nf_min_in:nf_max_in, 0], axis=0)).mean(axis=0)
    normal_resf_alt = scipy.stats.normaltest(signal_white_outf)

    print('normal fit res', normal_res)
    print('normal fit res A', normal_resa)
    print('normal fit res E', normal_rese)
    print('normal fit res freq', normal_resf)
    print('normal fit res time', normal_rest)
    print('normal fit res freq alt', normal_resf_alt)
    print('normal fit res freq A', normal_resfa)
    print('normal fit res freq E', normal_resfe)
    std_std = np.std(signal_white_out) * np.sqrt(2 / ns)
    stdf0 = np.std(signal_white_out[:, :].mean(axis=0).mean(axis=1))
    stdt0 = np.std(signal_white_out[:, :].mean(axis=1).mean(axis=1))
    nf0 = signal_white_out.shape[0] * signal_white_out.shape[2]
    nt0 = signal_white_out.shape[1] * signal_white_out.shape[2]
    std_stdf0 = stdf0 * np.sqrt(nf0) * np.sqrt(2 / signal_white_out.shape[1])
    std_stdt0 = stdt0 * np.sqrt(nt0) * np.sqrt(2 / signal_white_out.shape[0])
    print('mean white resids', np.mean(signal_white_out), 1.0 / np.sqrt(signal_white_out.size))
    print('std white resids', np.std(signal_white_out), '+/-', std_std, 1.0, (np.std(signal_white_out) - 1) / std_std)
    print('std white time', stdt0 * np.sqrt(nt0), '+/-', std_stdt0, 1.0, (stdt0 * np.sqrt(nt0) - 1) / std_stdt0)
    print('std white freq', stdf0 * np.sqrt(nf0), '+/-', std_stdf0, 1.0, (stdf0 * np.sqrt(nf0) - 1) / std_stdf0)
    return signal_white_out


def white_plot_ax(ax_in, title, data, extent_in):
    im_out = ax_in.imshow(np.rot90(data), extent=extent_in, cmap='YlOrRd', vmin=0, vmax=4, aspect='auto')

    ax_in.set_title(title, fontsize=14)
    ax_in.set_xlabel('t (yr)', fontsize=14)
    ax_in.set_xticks([0, 0.5, 1, 1.5, 2], minor=False)
    ax_in.set_xticklabels(['', '0.5', '1.0', '1.5', ''])
    ax_in.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    ax_in.set_yticks([1.0e-3, 2.0e-3, 3.0e-3])
    ax_in.set_yticklabels([])

    ax_in.tick_params(axis='both', which='major', labelsize=14)
    return im_out


if __name__ == '__main__':
    config, wc, lc, ic, instrument_random_seed = config_helper.get_config_objects('default_parameters.toml')

    nt_min = 256 * 6
    nt_max = nt_min + 512 * 2
    nt_lim = PixelGenericRange(nt_min, nt_max, wc.DT, 0.)
    nt_min_report = 0
    nt_max_report = nt_max - nt_min
    nt_lim_report = PixelGenericRange(nt_min_report, nt_max_report, wc.DT, 0.)
    nt_range = (nt_min, nt_max)

    ifm_cyclo = fetch_or_run_iterative_loop(config, cyclo_mode=0, nt_range=nt_range, fetch_mode=1)
    ifm_stat = fetch_or_run_iterative_loop(config, cyclo_mode=1, nt_range=nt_range, fetch_mode=1)

    noise_realization = ifm_stat.noise_manager.get_instrument_realization()

    bgd_cyclo = ifm_cyclo.noise_manager.bgd
    bgd_stat = ifm_stat.noise_manager.bgd
    # _, bgd_cyclo = gfi.load_processed_gb_file(config, snr_thresh, wc, nt_lim, cyclo_mode=0)
    # _, bgd_stat = gfi.load_processed_gb_file(config, snr_thresh, wc, nt_lim, cyclo_mode=1)

    galactic_cyclo = bgd_cyclo.get_galactic_below_high().reshape((wc.Nt, wc.Nf, bgd_cyclo.nc_galaxy))
    galactic_stat = bgd_stat.get_galactic_below_high().reshape((wc.Nt, wc.Nf, bgd_stat.nc_galaxy))

    signal_full_cyclo = galactic_cyclo + noise_realization
    signal_full_stat = galactic_stat + noise_realization

    S_inst_m = instrument_noise_AET_wdm_m(lc, wc)

    filter_periods = 1
    S_cyclo_model, _, _, _, _ = get_S_cyclo(galactic_cyclo, S_inst_m, wc.DT, 0, filter_periods,
                                            period_list=(1, 2, 3, 4, 5))

    fs = np.arange(1, wc.Nf) * wc.DF

    f_min = 8.0e-5
    f_max = 4.0e-3
    nf_min = int(np.argmax(fs > f_min))
    nf_max = int(np.argmax(fs > f_max))

    signal_white_resid_cyclo = result_normality_battery(nf_min, nf_max, signal_full_cyclo, S_cyclo_model)
    signal_white_resid_stat = result_normality_battery(nf_min, nf_max, signal_full_stat, S_cyclo_model)

    extent = (nt_min_report * wc.DT / gc.SECSYEAR, nt_lim_report.nx_max * wc.DT / gc.SECSYEAR, nf_min * wc.DF, nf_max * wc.DF)

    do_2plot = True
    if do_2plot:
        fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.85))
        fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.085, top=0.91, right=1.075, bottom=0.18)

        im = white_plot_ax(ax[0], r'Constant', (signal_white_resid_stat[nt_lim.nx_min:nt_lim.nx_max, :, 0:2] ** 2).sum(axis=2), extent)
        im = white_plot_ax(ax[1], r'Cyclostationary', (signal_white_resid_cyclo[nt_lim.nx_min:nt_lim.nx_max, :, 0:2] ** 2).sum(axis=2), extent)

        cbar = fig.colorbar(im, ax=ax[0:2], shrink=1.0, pad=0.01, orientation='vertical')
        cbar.set_ticks([0, 2, 4])
        cbar.ax.tick_params(axis='y', which='major', labelsize=14)

        plt.show()

    do_3plot = True
    if do_3plot:
        signal_white_2 = (
            galactic_cyclo[:, nf_min:nf_max, 0:2] ** 2 / (S_cyclo_model - S_inst_m)[:, nf_min:nf_max, 0:2]
        ).sum(axis=2)

        fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.85))
        fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.055, top=0.91, right=1.110, bottom=0.18)

        im = white_plot_ax(ax[0], r'Constant', (signal_white_resid_stat[nt_lim.nx_min:nt_lim.nx_max, :, 0:2] ** 2).sum(axis=2), extent)
        im = white_plot_ax(ax[1], r'Cyclostationary', (signal_white_resid_cyclo[nt_lim.nx_min:nt_lim.nx_max, :, 0:2] ** 2).sum(axis=2), extent)
        im = white_plot_ax(ax[2], r'Galactic Residual', signal_white_2[nt_lim.nx_min:nt_lim.nx_max], extent)

        cbar = fig.colorbar(im, ax=ax[0:3], shrink=1.0, pad=0.01, orientation='vertical')
        cbar.set_ticks([0, 2, 4])
        cbar.ax.tick_params(axis='y', which='major', labelsize=14)

        plt.show()
