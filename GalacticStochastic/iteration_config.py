"""read wavelet transform constants in from config file and compute derived parameters"""

from typing import NamedTuple

import numpy as np


class IterationConfig(NamedTuple):
    max_iterations: int
    snr_thresh: float
    snr_min: tuple[float, ...]
    snr_cut_bright: tuple[float, ...]
    smooth_lengthf: tuple[float, ...]
    period_list: tuple[np.floating, ...]
    n_cyclo_switch: int
    n_min_faint_adapt: int
    faint_converge_change_thresh: int
    smooth_lengthf_fix: float
    fmin_binary: float
    fmax_binary: float
    nc_galaxy: int
    snr_min_preprocess: float
    snr_min_reprocess: float
    noise_model_storage_mode: int
    background_storage_mode: int
    fit_state_storage_mode: int
    inclusion_state_storage_mode: int
    manager_storage_mode: int


def get_iteration_config(config: dict) -> IterationConfig:
    """Get lisa constant object from config file"""
    # maximum number of iterations to allow
    max_iterations = int(config['iterative_fit_constants']['max_iterations'])
    assert max_iterations >= 0

    # iteration to permit use of cyclostationary noise model
    n_cyclo_switch = int(config['iterative_fit_constants']['n_cyclo_switch'])
    assert n_cyclo_switch >= 0

    # minimum iterations to adapt the faint background
    n_min_faint_adapt = int(config['iterative_fit_constants']['n_min_faint_adapt'])
    assert n_min_faint_adapt >= 0

    # minimum iterations to adapt the faint background
    faint_converge_change_thresh = int(config['iterative_fit_constants']['faint_converge_change_thresh'])
    assert faint_converge_change_thresh >= 0

    snr_thresh = float(config['iterative_fit_constants']['snr_thresh'])
    assert snr_thresh >= 0.0

    # starting faint cutoff snr
    snr_low_initial = float(config['iterative_fit_constants']['snr_low_initial'])
    assert snr_low_initial >= 0.0

    # multiplier for faint cutoff snr in iterations after the zeroth
    snr_low_mult = float(config['iterative_fit_constants']['snr_low_mult'])
    assert snr_low_mult >= 0.0

    #  the faint cutoff snr
    snr_min = np.zeros(max_iterations)
    snr_min[0] = snr_low_initial
    snr_min[1:] = snr_low_mult * snr_low_initial

    # the list of periods to allow in the cyclostationary model
    period_list_in = np.array(config['iterative_fit_constants'].get('period_list'), dtype=float)
    assert np.all(period_list_in >= 0.0)
    period_list = tuple(period_list_in)

    # final frequency smoothing length for galactic spectrum in log frequency bins
    smooth_lengthf_fix = float(config['iterative_fit_constants']['smooth_lengthf_fix'])
    assert smooth_lengthf_fix >= 0.0

    fsmooth_settle_mult = float(config['iterative_fit_constants']['fsmooth_settle_mult'])

    fsmooth_settle_scale = float(config['iterative_fit_constants']['fsmooth_settle_scale'])

    fsmooth_settle_offset = float(config['iterative_fit_constants']['fsmooth_settle_offset'])

    fsmooth_fix_itr = int(config['iterative_fit_constants']['fsmooth_fix_itr'])
    assert fsmooth_fix_itr >= 0
    fsmooth_fix_itr = min(max_iterations, fsmooth_fix_itr)

    # phase in the frequency smoothing length gradually
    # give absorbing constants a relative advantage on early iterations
    # because for the first iteration we included no galactic background

    smooth_lengthf = np.zeros(max_iterations) + smooth_lengthf_fix
    for itrn in range(fsmooth_fix_itr):
        smooth_lengthf[itrn] += fsmooth_settle_mult * np.exp(-fsmooth_settle_scale * itrn - fsmooth_settle_offset)
        assert smooth_lengthf[itrn] >= 0.0

    # final bright snr cutoff
    snr_high_fix = float(config['iterative_fit_constants']['snr_high_fix'])
    assert snr_high_fix >= 0.0

    # phase in bright snr cutoff gradually
    snr_high_settle_mult = float(config['iterative_fit_constants']['snr_high_settle_mult'])

    snr_high_settle_scale = float(config['iterative_fit_constants']['snr_high_settle_scale'])

    snr_high_settle_offset = float(config['iterative_fit_constants']['snr_high_settle_offset'])

    snr_high_fix_itr = int(config['iterative_fit_constants']['snr_high_fix_itr'])
    assert snr_high_fix_itr >= 0
    snr_high_fix_itr = min(max_iterations, snr_high_fix_itr)

    snr_cut_bright = np.zeros(max_iterations) + snr_high_fix
    for itrn in range(snr_high_fix_itr):
        snr_cut_bright[itrn] += snr_high_settle_mult * np.exp(-snr_high_settle_scale * itrn - snr_high_settle_offset)
        assert snr_cut_bright[itrn] >= 0.0

    # minimum binary frequency to allow
    fmin_binary = float(config['iterative_fit_constants']['fmin_binary'])

    # maximum binary frequency to allow
    fmax_binary = float(config['iterative_fit_constants']['fmax_binary'])

    # number of galactic binary tdi channels
    nc_galaxy = int(config['iterative_fit_constants']['nc_galaxy'])
    assert nc_galaxy > 0
    assert nc_galaxy <= 3

    # minimum snr cut for preprocessing run
    snr_min_preprocess = float(config['iterative_fit_constants']['snr_min_preprocess'])

    # minimum snr cut for preprocessing run
    snr_min_reprocess = float(config['iterative_fit_constants']['snr_min_reprocess'])
    assert snr_min_reprocess >= snr_min_preprocess

    # select mode for the storage of the various objects
    noise_model_storage_mode = int(config['iterative_fit_constants'].get('noise_model_storage_mode', 0))
    background_storage_mode = int(config['iterative_fit_constants'].get('background_storage_mode', 0))
    fit_state_storage_mode = int(config['iterative_fit_constants'].get('fit_state_storage_mode', 0))
    inclusion_state_storage_mode = int(config['iterative_fit_constants'].get('inclusion_state_storage_mode', 0))
    manager_storage_mode = int(config['iterative_fit_constants'].get('manager_storage_mode', 0))

    # make arrays into tuples to ensure the configuration is immutable
    return IterationConfig(
        max_iterations,
        snr_thresh,
        tuple(snr_min),
        tuple(snr_cut_bright),
        tuple(smooth_lengthf),
        period_list,
        n_cyclo_switch,
        n_min_faint_adapt,
        faint_converge_change_thresh,
        smooth_lengthf_fix,
        fmin_binary,
        fmax_binary,
        nc_galaxy,
        snr_min_preprocess,
        snr_min_reprocess,
        noise_model_storage_mode,
        background_storage_mode,
        fit_state_storage_mode,
        inclusion_state_storage_mode,
        manager_storage_mode,
    )
