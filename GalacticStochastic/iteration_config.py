"""read wavelet transform constants in from config file and compute derived parameters"""

import ast
import json
from collections import namedtuple

import numpy as np

IterationConfig = namedtuple('IterationConfig', ['n_iterations', 'snr_thresh', 'snr_min', 'snr_cut_bright', 'smooth_lengthf'])

def get_iteration_config(config):
    """Get lisa constant object from config file"""

    # Mean arm length of the LISA detector (meters)
    n_iterations = int(ast.literal_eval(config['iterative fit constants']['n_iterations']))
    assert n_iterations >= 0

    snr_thresh = float(ast.literal_eval(config['iterative fit constants']['snr_thresh']))
    assert snr_thresh >= 0.

    snr_min = float(ast.literal_eval(config['iterative fit constants']['snr_min']))
    assert snr_min >= 0.

    #period_list = np.array(json.loads(config.get('iterative fit constants','period_list'))),dtype=np.float64)
    assert np.all(period_list >= 0.)

    #TODO also need ways to read in snr_cut_bright, smooth_lengthf

    return IterationConfig(n_iterations, snr_thresh, snr_min, snr_cut_bright, smooth_lengthf)
