"""read wavelet transform constants in from config file and compute derived parameters"""

import ast
import json
from collections import namedtuple

import numpy as np

IterationConfig = namedtuple('IterationConfig', ['max_iterations', 'snr_thresh', 'snr_min', 'snr_cut_bright', 'smooth_lengthf', 'period_list', 'n_cyclo_switch', 'n_min_faint_adapt', 'faint_converge_change_thresh', 'smooth_lengthf_fix', 'fmin_binary', 'fmax_binary', 'nc_galaxy', 'snr_min_preprocess', 'snr_min_reprocess'])


def get_iteration_config(config):
    """Get lisa constant object from config file"""
    # maximum number of iterations to allow
    max_iterations = int(ast.literal_eval(config['iterative fit constants']['max_iterations']))
    assert max_iterations >= 0

    # iteration to permit use of cyclostationary noise model
    n_cyclo_switch = int(ast.literal_eval(config['iterative fit constants']['n_cyclo_switch']))
    assert n_cyclo_switch >= 0

    # minimum iterations to adapt the faint background
    n_min_faint_adapt = int(ast.literal_eval(config['iterative fit constants']['n_min_faint_adapt']))
    assert n_min_faint_adapt >= 0

    # minimum iterations to adapt the faint background
    faint_converge_change_thresh = int(ast.literal_eval(config['iterative fit constants']['faint_converge_change_thresh']))
    assert faint_converge_change_thresh >= 0

    snr_thresh = float(ast.literal_eval(config['iterative fit constants']['snr_thresh']))
    assert snr_thresh >= 0.

    # starting faint cutoff snr
    snr_low_initial = float(ast.literal_eval(config['iterative fit constants']['snr_low_initial']))
    assert snr_low_initial >= 0.

    # multiplier for faint cutoff snr in iterations after the zeroth
    snr_low_mult = float(ast.literal_eval(config['iterative fit constants']['snr_low_mult']))
    assert snr_low_mult >= 0.

    #  the faint cutoff snr
    snr_min = np.zeros(max_iterations)
    snr_min[0] = snr_low_initial
    snr_min[1:] = snr_low_mult * snr_low_initial

    # the list of periods to allow in the cyclostationary model
    period_list = np.array(json.loads(config.get('iterative fit constants', 'period_list')), dtype=float)
    assert np.all(period_list >= 0.)
    period_list = tuple(period_list)

    # final frequency smoothing length for galactic spectrum in log frequency bins
    smooth_lengthf_fix = float(ast.literal_eval(config['iterative fit constants']['smooth_lengthf_fix']))
    assert smooth_lengthf_fix >= 0.

    fsmooth_settle_mult = float(ast.literal_eval(config['iterative fit constants']['fsmooth_settle_mult']))

    fsmooth_settle_scale = float(ast.literal_eval(config['iterative fit constants']['fsmooth_settle_scale']))

    fsmooth_settle_offset = float(ast.literal_eval(config['iterative fit constants']['fsmooth_settle_offset']))

    fsmooth_fix_itr = int(ast.literal_eval(config['iterative fit constants']['fsmooth_fix_itr']))
    assert fsmooth_fix_itr >= 0
    fsmooth_fix_itr = min(max_iterations, fsmooth_fix_itr)

    # phase in the frequency smoothing length gradually
    # give absorbing constants a relative advantage on early iterations
    # because for the first iteration we included no galactic background

    smooth_lengthf = np.zeros(max_iterations) + smooth_lengthf_fix
    for itrn in range(fsmooth_fix_itr):
        smooth_lengthf[itrn] += fsmooth_settle_mult * np.exp(-fsmooth_settle_scale * itrn - fsmooth_settle_offset)
        assert smooth_lengthf[itrn] >= 0.

    # final bright snr cutoff
    snr_high_fix = float(ast.literal_eval(config['iterative fit constants']['snr_high_fix']))
    assert snr_high_fix >= 0.

    # phase in bright snr cutoff gradually
    snr_high_settle_mult = float(ast.literal_eval(config['iterative fit constants']['snr_high_settle_mult']))

    snr_high_settle_scale = float(ast.literal_eval(config['iterative fit constants']['snr_high_settle_scale']))

    snr_high_settle_offset = float(ast.literal_eval(config['iterative fit constants']['snr_high_settle_offset']))

    snr_high_fix_itr = int(ast.literal_eval(config['iterative fit constants']['snr_high_fix_itr']))
    assert snr_high_fix_itr >= 0
    snr_high_fix_itr = min(max_iterations, snr_high_fix_itr)

    snr_cut_bright = np.zeros(max_iterations) + snr_high_fix
    for itrn in range(snr_high_fix_itr):
        snr_cut_bright[itrn] += snr_high_settle_mult * np.exp(-snr_high_settle_scale * itrn - snr_high_settle_offset)
        assert snr_cut_bright[itrn] >= 0.

    # minimum binary frequency to allow
    fmin_binary = float(ast.literal_eval(config['iterative fit constants']['fmin_binary']))

    # maximum binary frequency to allow
    fmax_binary = float(ast.literal_eval(config['iterative fit constants']['fmax_binary']))

    # number of galactic binary tdi channels
    nc_galaxy = int(ast.literal_eval(config['iterative fit constants']['nc_galaxy']))
    assert nc_galaxy > 0
    assert nc_galaxy <= 3

    # minimum snr cut for preprocessing run
    snr_min_preprocess = float(ast.literal_eval(config['iterative fit constants']['snr_min_preprocess']))

    # minimum snr cut for preprocessing run
    snr_min_reprocess = float(ast.literal_eval(config['iterative fit constants']['snr_min_reprocess']))
    assert snr_min_reprocess >= snr_min_preprocess

    # make arrays into tuples to ensure the configuration is immutable
    return IterationConfig(max_iterations, snr_thresh, tuple(snr_min), tuple(snr_cut_bright), tuple(smooth_lengthf), period_list, n_cyclo_switch, n_min_faint_adapt, faint_converge_change_thresh, smooth_lengthf_fix, fmin_binary, fmax_binary, nc_galaxy, snr_min_preprocess, snr_min_reprocess)
