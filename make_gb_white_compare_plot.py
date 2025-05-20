"""make plot comparing galactic background noise spectra with and without cyclostationary model"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.special
import scipy.stats

import GalacticStochastic.global_const as gc
import GalacticStochastic.global_file_index as gfi
from GalacticStochastic import config_helper
from GalacticStochastic.galactic_fit_helpers import get_S_cyclo
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m

mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5


def result_normality_battery(signal_in):
    ns = signal_in[:, nf_min:nf_max, 0:2].size
    signal_white_out = signal_in[:, nf_min:nf_max, 0:2] / np.sqrt(S_cyclo_model[:, nf_min:nf_max, 0:2])
    normal_res = scipy.stats.normaltest(signal_white_out.flatten())
    normal_resa = scipy.stats.normaltest(signal_white_out[:, :, 0].flatten())
    normal_rese = scipy.stats.normaltest(signal_white_out[:, :, 1].flatten())
    normal_resf = scipy.stats.normaltest(signal_white_out.mean(axis=0).mean(axis=1).flatten())
    normal_resfa = scipy.stats.normaltest(signal_white_out[:, :, 0].mean(axis=0).flatten())
    normal_resfe = scipy.stats.normaltest(signal_white_out[:, :, 1].mean(axis=0).flatten())
    normal_rest = scipy.stats.normaltest(signal_white_out.mean(axis=1).mean(axis=1).flatten())
    signal_white_outf = (signal_in[:, nf_min:nf_max, 0] / np.std(signal_in[:, nf_min:nf_max, 0], axis=0)).mean(axis=0)
    normal_resf_alt = scipy.stats.normaltest(signal_white_outf)

    print("normal fit res", normal_res)
    print("normal fit res A", normal_resa)
    print("normal fit res E", normal_rese)
    print("normal fit res freq", normal_resf)
    print("normal fit res time", normal_rest)
    print("normal fit res freq alt", normal_resf_alt)
    print("normal fit res freq A", normal_resfa)
    print("normal fit res freq E", normal_resfe)
    std_std = np.std(signal_white_out) * np.sqrt(2 / ns)
    stdf0 = np.std(signal_white_out[:, :].mean(axis=0).mean(axis=1))
    stdt0 = np.std(signal_white_out[:, :].mean(axis=1).mean(axis=1))
    nf0 = signal_white_out.shape[0] * signal_white_out.shape[2]
    nt0 = signal_white_out.shape[1] * signal_white_out.shape[2]
    std_stdf0 = stdf0 * np.sqrt(nf0) * np.sqrt(2 / signal_white_out.shape[1])
    std_stdt0 = stdt0 * np.sqrt(nt0) * np.sqrt(2 / signal_white_out.shape[0])
    print('mean white resids', np.mean(signal_white_out), 1. / np.sqrt(signal_white_out.size))
    print('std white resids', np.std(signal_white_out), "+/-", std_std, 1., (np.std(signal_white_out) - 1) / std_std)
    print('std white time', stdt0 * np.sqrt(nt0), "+/-", std_stdt0, 1., (stdt0 * np.sqrt(nt0) - 1) / std_stdt0)
    print('std white freq', stdf0 * np.sqrt(nf0), "+/-", std_stdf0, 1., (stdf0 * np.sqrt(nf0) - 1) / std_stdf0)
    return signal_white_out


if __name__ == '__main__':

    config, wc, lc = config_helper.get_config_objects('default_parameters.toml')
    galaxy_dir = config['files']['galaxy_dir']

    nt_min = 256 * 6
    nt_max = nt_min + 512 * 2
    nt_min_report = 0
    nt_max_report = nt_max - nt_min

    snr_thresh = 7.
    smooth_lengthf = 6

    noise_realization = gfi.get_noise_common(galaxy_dir, snr_thresh, wc)

    _, galactic_cyclo = gfi.load_processed_gb_file(galaxy_dir, snr_thresh, wc, lc, nt_min, nt_max, False)
    _, galactic_stat = gfi.load_processed_gb_file(galaxy_dir, snr_thresh, wc, lc, nt_min, nt_max, True)

    signal_full_cyclo = galactic_cyclo + noise_realization
    signal_full_stat = galactic_stat + noise_realization

    S_inst_m = instrument_noise_AET_wdm_m(lc, wc)

    S_cyclo_model, _, _, _, _ = get_S_cyclo(galactic_cyclo, S_inst_m, wc, 0, True, period_list=(1, 2, 3, 4, 5))

    fs = np.arange(1, wc.Nf) * wc.DF

    nf_min = np.argmax(fs > 8.e-5)
    nf_max = np.argmax(fs > 4.e-3)

    signal_white_resid_cyclo = result_normality_battery(signal_full_cyclo)
    signal_white_resid_stat = result_normality_battery(signal_full_stat)


extent = (nt_min_report * wc.DT / gc.SECSYEAR, nt_max_report * wc.DT / gc.SECSYEAR, nf_min * wc.DF, nf_max * wc.DF)


def white_plot_ax(ax_in, title, data):
    im_out = ax_in.imshow(np.rot90(data), extent=extent, cmap='YlOrRd', vmin=0, vmax=4, aspect=aspect)

    ax_in.set_title(title, fontsize=14)
    ax_in.set_xlabel('t (yr)', fontsize=14)
    ax_in.set_xticks([0, 0.5, 1, 1.5, 2], minor=False)
    ax_in.set_xticklabels(['', '0.5', '1.0', '1.5', ''])
    ax_in.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    ax_in.set_yticks([1.e-3, 2.e-3, 3.e-3])
    ax_in.set_yticklabels([])
    for label_loc in ax[0].get_yticklabels(minor=False):
        label_loc.get_font_properties().set_size(14)

    for label_loc in ax_in.get_xticklabels(minor=False):
        label_loc.get_font_properties().set_size(14)
    return im_out


do_2plot = True
if do_2plot:
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.85))
    fig.subplots_adjust(wspace=0., hspace=0., left=0.085, top=0.91, right=1.075, bottom=0.18)

    aspect = 'auto'
    im = white_plot_ax(ax[0], r"Constant", (signal_white_resid_stat[nt_min:nt_max, :, 0:2]**2).sum(axis=2))
    im = white_plot_ax(ax[1], r"Cyclostationary", (signal_white_resid_cyclo[nt_min:nt_max, :, 0:2]**2).sum(axis=2))

    cbar = fig.colorbar(im, ax=ax[0:2], shrink=1., pad=0.01, orientation='vertical')
    cbar.set_ticks([0, 2, 4])
    for label in cbar.ax.get_yticklabels(minor=False):
        label.get_font_properties().set_size(14)

    plt.show()


do_3plot = True
if do_3plot:
    signal_white_2 = (galactic_cyclo[:, nf_min:nf_max, 0:2] ** 2 / (S_cyclo_model - S_inst_m)[:, nf_min:nf_max, 0:2]).sum(axis=2)

    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.85))
    fig.subplots_adjust(wspace=0., hspace=0., left=0.055, top=0.91, right=1.110, bottom=0.18)

    aspect = 'auto'
    im = white_plot_ax(ax[0], r"Constant", (signal_white_resid_stat[nt_min:nt_max, :, 0:2]**2).sum(axis=2))
    im = white_plot_ax(ax[1], r"Cyclostationary", (signal_white_resid_cyclo[nt_min:nt_max, :, 0:2]**2).sum(axis=2))
    im = white_plot_ax(ax[2], r"Galactic Residual", signal_white_2[nt_min:nt_max])

    cbar = fig.colorbar(im, ax=ax[0:3], shrink=1., pad=0.01, orientation='vertical')
    cbar.set_ticks([0, 2, 4])
    for label in cbar.ax.get_yticklabels(minor=False):
        label.get_font_properties().set_size(14)

    plt.show()
