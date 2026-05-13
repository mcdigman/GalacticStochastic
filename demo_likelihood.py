"""Test internal agreement between TaylorF2 waveform variants"""
from pathlib import Path
from time import perf_counter

import numpy as np
import tomllib
from numpy.typing import NDArray

from LisaWaveformTools.binary_params_manager import M_SUN_SEC, PC_M, BinaryIntrinsicParamsManager
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalStationaryDenseNoiseModel
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from LisaWaveformTools.taylorf2_freq_source import TaylorF2WaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, wavelet_sparse_to_dense
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletTaylorFreq
from WaveletWaveforms.wdm_config import get_wavelet_model

if __name__ == '__main__':
    model1 = 'imrphenomd'
    toml_filename = 'likelihood_demo_parameters.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    # Define parameters for a low-mass, low-fdot binary system
    intrinsic_params_packed: NDArray[np.floating] = np.array([
        np.log(1.e1 * PC_M),  # Log luminosity distance in meters
        600.0 * M_SUN_SEC,  # Total mass in seconds
        260.0 * M_SUN_SEC,  # Chirp mass in seconds
        4.50e-3,              # Initial frequency in Hz
        1.0e8,             # Coalescence time in seconds
        0.3,               # Phase at coalescence
        0.0,               # Normalize postnewtonian spin parameter
        0.0,               # Antisymmetric component of aligned spin
        0.0,               # Precessing spin
        0.0,               # Initial eccentricity
    ])
    intrinsic_params_manager = BinaryIntrinsicParamsManager(intrinsic_params_packed)
    intrinsic = intrinsic_params_manager.params
    extrinsic = ExtrinsicParams(
        costh=0.1, phi=0.1, cosi=0.2, psi=0.3
    )

    source_params = SourceParams(intrinsic=intrinsic, extrinsic=extrinsic)

    amplitude_pn_mode = 0
    include_pn_ss3 = 0
    t_obs = wc.Tobs

    # Initialize TaylorF2 waveform source
    taylorf2_waveform1 = TaylorF2WaveformFreq(
        params=source_params,
        lc=lc,
        nf_lim_absolute=PixelGenericRange(0, wc.Nf, wc.DF, 0.0),
        freeze_limits=1,
        t_obs=t_obs,
        model_select=model1,
        amplitude_pn_mode=amplitude_pn_mode,
        include_pn_ss3=include_pn_ss3,
        tc_mode=0,
    )

    # get wavelet waveform object (should be able to call update_params on this object later)
    wavelet_waveform = BinaryWaveletTaylorFreq(
        params=source_params,
        wc=wc,
        lc=lc,
        nt_lim_waveform=PixelGenericRange(0, wc.Nt, wc.DT, 0.0),
        nf_lim_waveform=taylorf2_waveform1.nf_lim,
        source_waveform=taylorf2_waveform1,
        wavelet_mode=1,
    )

    # get a noise object
    SAET_m = instrument_noise_AET_wdm_m(lc, wc)
    noise_dense = DiagonalStationaryDenseNoiseModel(SAET_m, wc, prune=0, nc_snr=lc.nc_snr)

    # get a signal to inject
    wavelet_coeffs_base = wavelet_waveform.get_unsorted_coeffs()
    wavelet_coeffs_base_white = noise_dense.whiten_sparse(wavelet_coeffs_base)
    waveform_dense_white = wavelet_sparse_to_dense(wavelet_coeffs_base_white, wc)

    # add simulated noise to injected signal
    instrument_noise_white = noise_dense.generate_dense_noise(white_mode=1)
    data_white = instrument_noise_white + waveform_dense_white

    wavelet_coeffs = wavelet_waveform.get_unsorted_coeffs()
    logL_got = noise_dense.get_log_likelihood_sparse(wavelet_coeffs, data_white, PixelGenericRange(0, wc.Nt, wc.DT, 0.0))

    n_run = 100
    t0 = perf_counter()
    for _itrm in range(n_run):
        wavelet_waveform.update_params(source_params)
        wavelet_coeffs = wavelet_waveform.get_unsorted_coeffs()
        logL_got = noise_dense.get_log_likelihood_sparse(wavelet_coeffs, data_white, PixelGenericRange(0, wc.Nt, wc.DT, 0.0))
    tf = perf_counter()
    print('Waveforms and likelihoods calculated in %8.5f ms' % (1000.0 * (tf - t0) / n_run))
