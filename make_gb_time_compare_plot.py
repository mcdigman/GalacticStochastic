"""scratch to test processing of galactic background"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import GalacticStochastic.global_file_index as gfi
from GalacticStochastic import config_helper
from GalacticStochastic.galactic_fit_helpers import fit_gb_spectrum_evolve, get_S_cyclo
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


if __name__ == '__main__':
    snr_thresh = 7.0
    smooth_targ_length = 0.25

    config, wc, lc = config_helper.get_config_objects('default_parameters.toml')
    galaxy_dir = config['files']['galaxy_dir']

    fs = np.arange(0, wc.Nf) * wc.DF

    stat_only = False
    idx_use = [0, 1, 3, 7]
    nt_mins = np.array([256 * 7, 256 * 6, 256 * 5, 256 * 4, 256 * 3, 256 * 2, 256 * 1, 256 * 0])[idx_use]
    nt_maxs = np.array([512 * 1, 512 * 2, 512 * 3, 512 * 4, 512 * 5, 512 * 6, 512 * 7, 512 * 8])[idx_use] + nt_mins

    nt_ranges = nt_maxs - nt_mins
    nk = nt_maxs.size
    assert nt_mins.size == nt_maxs.size

    itrl_fit = 0

    if not stat_only:
        filter_periods = 1
        period_list = (1, 2, 3, 4, 5)
    else:
        filter_periods = 0
        period_list = ()

    S_stat_m = np.zeros((nk, wc.Nf, 3))
    S_stat_smooth_m = np.zeros((nk, wc.Nf, 3))

    S_inst_m = instrument_noise_AET_wdm_m(lc, wc)
    S_stat_offset = 1.0 * S_inst_m[:, 0]

    r_tots = np.zeros((nk, wc.Nt, S_inst_m.shape[-1]))

    for itrk in range(nk):
        _, galactic_below_high = gfi.load_processed_gb_file(
            galaxy_dir, snr_thresh, wc, lc, nt_mins[itrk], nt_maxs[itrk], stat_only,
        )
        S_stat_m[itrk] = np.mean(galactic_below_high, axis=0)

        S_stat_smooth_m[itrk, 0, :] = S_stat_m[itrk, 0, :]

        (_, r_tots[itrk], S_stat_smooth_m[itrk], _, _) = get_S_cyclo(
            galactic_below_high, S_inst_m, wc, smooth_targ_length, filter_periods, period_list=period_list, Nt_loc=wc.Nt,
        )

        for itrc in range(2):
            S_stat_smooth_m[itrk, :, itrc] += S_stat_offset

    arg_cut = wc.Nf - 1
    fit_mask = (fs > 1.0e-5) & (fs < fs[arg_cut])

    S_fit_evolve_m, _ = fit_gb_spectrum_evolve(
        S_stat_smooth_m[itrl_fit:, fit_mask, :], fs[fit_mask], fs[1:], nt_ranges[itrl_fit:], S_stat_offset[fit_mask], wc,
    )

    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.13, top=0.99, right=0.99, bottom=0.12)
    ax.loglog(
        fs[1:],
        wc.dt * (S_stat_smooth_m[:, 1:, :2].mean(axis=2) - S_stat_offset[1:] + S_inst_m[1:, 0]).T,
        alpha=0.5,
        label='_nolegend_',
    )
    ax.set_prop_cycle(None)
    ax.loglog(fs[1:], wc.dt * (S_fit_evolve_m[:] + S_inst_m[1:, 0]).T, linewidth=3)
    ax.set_prop_cycle(None)
    ax.loglog(fs[1:], wc.dt * (S_inst_m[1:, 0]), 'k--')
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    plt.ylim([wc.dt * 2.0e-44, wc.dt * 2.0e-43])
    plt.xlim([3.0e-4, 6.0e-3])
    plt.xlabel('f [Hz]')
    plt.ylabel(r'$S^{AE}(f)$ [Hz$^{-1}$]')
    plt.legend(['1 year', '2 years', '4 years', '8 years'])
    plt.show()
