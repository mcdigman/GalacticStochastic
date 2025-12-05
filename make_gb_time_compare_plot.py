"""scratch to test processing of galactic background"""

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from GalacticStochastic import config_helper
from GalacticStochastic.galactic_fit_helpers import fit_gb_spectrum_evolve, get_S_cyclo
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


if __name__ == '__main__':
    smooth_targ_length = 0.25
    _: Any
    # filename_config = 'default_parameters.toml'
    # filename_config = 'Galaxies/GalaxyFullLDC/run_old_parameters_format4.toml'
    # filename_config = 'parameters_default_12year.toml'
    # target_directory = 'Galaxies/GalaxyFullLDC/'
    filename_config = 'Galaxies/COSMIC_alpha25/parameters_default_12year.toml'
    target_directory = 'Galaxies/COSMIC_alpha25/'
    # filename_config = 'Galaxies/FZq3Z/parameters_default_12year.toml'
    # target_directory = 'Galaxies/FZq3Z/'
    # filename_config = 'Galaxies/FZAlpha5/parameters_default_12year.toml'
    # target_directory = 'Galaxies/FZAlpha5/'
    # filename_config = 'Galaxies/FZFiducial/parameters_default_12year.toml'
    # target_directory = 'Galaxies/FZFiducial/'

    config, wc, lc, ic, instrument_random_seed = config_helper.get_config_objects(filename_config)
    config['files']['galaxy_dir'] = target_directory

    fs = np.arange(0, wc.Nf) * wc.DF

    cyclo_mode = 0
    idx_use = [0, 1, 3, 7, 11]
    nt_incr = int(wc.Nt // 24)
    # nt_mins = np.array([nt_incr * 7, nt_incr * 6, nt_incr * 5, nt_incr * 4, nt_incr * 3, nt_incr * 2, nt_incr * 1, nt_incr * 0])[idx_use]
    # nt_maxs = np.array([2 * nt_incr * 1, 2 * nt_incr * 2, 2 * nt_incr * 3, 2 * nt_incr * 4, 2 * nt_incr * 5, 2 * nt_incr * 6, 2 * nt_incr * 7, 2 * nt_incr * 8])[idx_use] + nt_mins
    nt_mins = np.array([nt_incr * 11, nt_incr * 10, nt_incr * 9, nt_incr * 8, nt_incr * 7, nt_incr * 6, nt_incr * 5, nt_incr * 4, nt_incr * 3, nt_incr * 2, nt_incr * 1, nt_incr * 0])[idx_use]
    nt_maxs = np.array([2 * nt_incr * 1, 2 * nt_incr * 2, 2 * nt_incr * 3, 2 * nt_incr * 4, 2 * nt_incr * 5, 2 * nt_incr * 6, 2 * nt_incr * 7, 2 * nt_incr * 8, 2 * nt_incr * 9, 2 * nt_incr * 10, 2 * nt_incr * 11, 2 * nt_incr * 12])[idx_use] + nt_mins

    nt_ranges = nt_maxs - nt_mins
    nk = nt_maxs.size
    assert nt_mins.size == nt_maxs.size

    itrl_fit = 0

    if not cyclo_mode:
        filter_periods = 1
        period_list: tuple[int, ...] = (1, 2, 3, 4, 5)
    else:
        filter_periods = 0
        period_list = ()

    S_stat_m = np.zeros((nk, wc.Nf, 3))
    S_stat_smooth_m = np.zeros((nk, wc.Nf, 3))

    S_inst_m = instrument_noise_AET_wdm_m(lc, wc)
    S_stat_offset = 1.0 * S_inst_m[:, 0]

    r_tots = np.zeros((nk, wc.Nt, S_inst_m.shape[-1]))

    for itrk in range(nk):
        nt_lim = PixelGenericRange(int(nt_mins[itrk]), int(nt_maxs[itrk]), wc.DT, 0)
        ifm = fetch_or_run_iterative_loop(config, cyclo_mode, nt_range_snr=(nt_lim.nx_min, nt_lim.nx_max), fetch_mode=1)
        galactic_below_high = ifm.noise_manager.bgd.get_galactic_below_high(shape_mode=1)

        S_stat_m[itrk] = np.mean(galactic_below_high, axis=0)

        S_stat_smooth_m[itrk, 0, :] = S_stat_m[itrk, 0, :]

        (_, r_tots[itrk], S_stat_smooth_m[itrk], _, _) = get_S_cyclo(galactic_below_high, S_inst_m, wc.DT,
                                                                     smooth_targ_length, filter_periods,
                                                                     period_list=period_list)

        for itrc in range(2):
            S_stat_smooth_m[itrk, :, itrc] += S_stat_offset

    arg_cut = wc.Nf - 1
    fit_mask = (fs > 1.0e-5) & (fs < fs[arg_cut])

    S_fit_evolve_m, _ = fit_gb_spectrum_evolve(
        S_stat_smooth_m[itrl_fit:, fit_mask, :], fs[fit_mask], fs[1:], nt_ranges[itrl_fit:], S_stat_offset[fit_mask], wc.DT,
    )

fig = plt.figure(figsize=(5.4, 3.5))
ax = fig.subplots(1)
fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.13, top=0.99, right=0.99, bottom=0.12)
_ = ax.loglog(
    fs[1:],
    wc.dt * (S_stat_smooth_m[:, 1:, :2].mean(axis=2) - S_stat_offset[1:] + S_inst_m[1:, 0]).T,
    alpha=0.5,
    label='_nolegend_',
)
ax.set_prop_cycle(None)
_ = ax.loglog(fs[1:], wc.dt * (S_fit_evolve_m[:] + S_inst_m[1:, 0]).T, linewidth=3)
ax.set_prop_cycle(None)
_ = ax.loglog(fs[1:], wc.dt * (S_inst_m[1:, 0]), 'k--')
ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
# _ = plt.ylim([wc.dt * 2.0e-44, wc.dt * 2.0e-43])
_ = plt.ylim([1.e-43, 2.0e-39])
_ = plt.xlim([3.0e-4, 6.0e-3])
_ = plt.xlabel('f [Hz]')
_ = plt.ylabel(r'$S^{AE}(f)$ [Hz$^{-1}$]')
_ = plt.legend(labels=['1 year', '2 years', '4 years', '8 years'])
plt.show()

# fig = plt.figure(figsize=(5.4, 3.5))
# ax = fig.subplots(1)
# fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.13, top=0.99, right=0.99, bottom=0.12)
# _ = ax.loglog(
#    fs[1:],
#    np.sqrt(wc.dt * (S_stat_smooth_m[:, 1:, :2].mean(axis=2) - S_stat_offset[1:] + S_inst_m[1:, 0]).T),
#    alpha=0.5,
#    label='_nolegend_',
# )
# ax.set_prop_cycle(None)
# _ = ax.loglog(fs[1:], np.sqrt(wc.dt * (S_fit_evolve_m[:] + S_inst_m[1:, 0]).T), linewidth=3)
# ax.set_prop_cycle(None)
# _ = ax.loglog(fs[1:], np.sqrt(wc.dt * (S_inst_m[1:, 0])), 'k--')
# ax.set_prop_cycle(None)
# _ = ax.scatter(ifm.bis.params_gb[ifm.bis.get_bright()[ifm.bis._argbinmap]][:, 3], ifm.bis.params_gb[ifm.bis.get_bright()[ifm.bis._argbinmap]][:, 0])
# _ = ax.scatter(ifm.bis.params_gb[~ifm.bis.get_bright()[ifm.bis._argbinmap]][:, 3], ifm.bis.params_gb[~ifm.bis.get_bright()[ifm.bis._argbinmap]][:, 0])
# ax.loglog(fs[1:], np.sqrt(np.mean(np.sum(ifm.noise_manager.bgd.get_galactic_total(shape_mode=1)[:, 1:, 0:2]**2, axis=2), axis=0)))
# ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
# _ = plt.ylim([wc.dt * 2.0e-44, wc.dt * 2.0e-43])
# _ = plt.ylim([1.e-43, 2.0e-39])
# _ = plt.xlim([1.0e-4, 6.0e-3])
# _ = plt.xlabel('f [Hz]')
# _ = plt.ylabel(r'$S^{AE}(f)$ [Hz$^{-1}$]')
# _ = plt.legend(labels=['1 year', '2 years', '4 years', '8 years'])
# plt.show()
