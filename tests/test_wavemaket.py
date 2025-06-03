"""Unit tests for get_time_tdi_amp_phase.
There is also some incidental/partial coverage of wavemaket.
"""

from pathlib import Path

import numpy as np
import pytest
import tomllib
import WDMWaveletTransforms.fft_funcs as fft
from numpy.testing import assert_allclose
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.signal import butter, filtfilt, hilbert
from WDMWaveletTransforms.transform_freq_funcs import tukey
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_freq, inverse_wavelet_freq_time, transform_wavelet_freq, transform_wavelet_freq_time

from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.ra_waveform_freq import AntennaResponseChannels
from LisaWaveformTools.ra_waveform_time import get_time_tdi_amp_phase
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from tests.test_time_tdi_ampphase import get_RR_t_mult, get_waveform_helper
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange, sparse_addition_helper
from WaveletWaveforms.taylor_time_coefficients import (
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.taylor_time_wavelet_funcs import wavemaket, wavemaket_direct
from WaveletWaveforms.wdm_config import get_wavelet_model


def tukey_waveform_amp(AET_waveform: StationaryWaveformTime, tukey_alpha, arg_cut):
    """Apply a tukey filter to the input intrinsic_waveform's tdi amplitude.
    Cut off everything beyond arg_cut to 0.0
    """
    AT = AET_waveform.AT
    nc = AT.shape[0]
    for itrc in range(nc):
        tukey(AT[itrc], tukey_alpha, arg_cut)
        AT[itrc, arg_cut:] = 0.0


def get_aet_waveform_helper(lc, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha, nt_loc, dt_loc):
    # ensure the RRs and IIs are scaled diferently in different channels
    RR_scale_mult = np.array([1.0])
    II_scale_mult = np.array([1.0])

    # generate the intrinsic_waveform in the wavelet domain and transform it
    waveform, AET_waveform, arg_cut = get_waveform_helper(
        p_input, f_input, fp_input, fpp_input, amp_input, nt_loc, dt_loc, nc_waveform, max_f=f_high_cut,
    )
    T = waveform.T

    RR_t_mult, II_t_mult = get_RR_t_mult(rr_model, nt_loc, 1, dt_loc)
    RR = np.outer(RR_scale_mult, RR_t_mult)
    II = np.outer(II_scale_mult, II_t_mult)
    dRR = np.zeros((nc_waveform, nt_loc))
    dII = np.zeros((nc_waveform, nt_loc))
    spacecraft_channels = AntennaResponseChannels(T, RR, II, dRR, dII)
    get_time_tdi_amp_phase(spacecraft_channels, AET_waveform, waveform, lc, dt_loc)

    # tukey window the amplitudes to cut down on edge artifacts
    tukey_waveform_amp(AET_waveform, tukey_alpha, arg_cut)

    return AET_waveform, arg_cut


def get_wavelet_alternative_representation_helper(wavelet_waveform, wc, tukey_alpha, f_lowpass, min_mag_mult=1.e-2):
    b, a = butter(4, f_lowpass, fs=1. / wc.dt, btype='low', analog=False)
    nd_loc = wc.Nt * wc.Nf
    nc_waveform = wavelet_waveform.n_set.size
    wavelet_dense = np.zeros((nd_loc, nc_waveform))
    sparse_addition_helper(wavelet_waveform, wavelet_dense)
    wavelet_dense = wavelet_dense.reshape((wc.Nt, wc.Nf, nc_waveform))

    # cut off the signal completely if it drops to zero
    t_sum = np.sum(wavelet_dense**2, axis=(1, 2))
    nt_max_active = wc.Nt - np.argmax(t_sum[::-1] > 0.)
    nt_max_cut = wc.Nt * wc.Nf
    if nt_max_active < wc.Nt:
        nt_max_cut = min(nt_max_cut, int(np.ceil(nt_max_active * wc.Nf + wc.Tw / wc.dt + 1)))

    # get the time domain signal from the wavelets
    signal_time = np.zeros((nd_loc, nc_waveform))
    for itrc in range(nc_waveform):
        signal_time[:, itrc] = inverse_wavelet_freq_time(wavelet_dense[:, :, itrc], wc.Nf, wc.Nt)

    # get the frequency domain signal from the wavelets
    signal_freq = np.zeros((nd_loc // 2 + 1, nc_waveform), dtype=np.complex128)

    for itrc in range(nc_waveform):
        signal_freq[:, itrc] = inverse_wavelet_freq(wavelet_dense[:, :, itrc], wc.Nf, wc.Nt)

    analytic = np.asarray(hilbert(signal_time, axis=0), dtype=np.complex128)

    analytic[nt_max_cut:] = 0.

    if nt_max_cut < wc.Nt * wc.Nf:
        for itrc in range(nc_waveform):
            tukey(analytic[:, itrc], tukey_alpha, nt_max_cut)

    envelope = np.abs(analytic)

    p_envelope = np.asarray(np.unwrap(np.angle(analytic), axis=0), dtype=np.float64)
    f_envelope = np.gradient(p_envelope, wc.dt, axis=0) / (2 * np.pi)
    fd_envelope = np.gradient(f_envelope, wc.dt, axis=0)

    envelope = np.asarray(filtfilt(b, a, envelope, axis=0), dtype=np.float64)

    for itrc in range(nc_waveform):
        tukey(envelope[:, itrc], tukey_alpha, nt_max_cut)

    envelope[nt_max_cut:] = 0.

    for itrc in range(nc_waveform):
        tukey(f_envelope[:, itrc], tukey_alpha, nt_max_cut)

    f_envelope = np.asarray(filtfilt(b, a, f_envelope, axis=0), dtype=np.float64)

    for itrc in range(nc_waveform):
        tukey(f_envelope[:, itrc], tukey_alpha, nt_max_cut)

    f_envelope[nt_max_cut:] = 0.

    for itrc in range(nc_waveform):
        tukey(fd_envelope[:, itrc], tukey_alpha, nt_max_cut)

    fd_envelope = np.asarray(filtfilt(b, a, fd_envelope, axis=0), dtype=np.float64)

    for itrc in range(nc_waveform):
        tukey(fd_envelope[:, itrc], tukey_alpha, nt_max_cut)

    for itrc in range(nc_waveform):
        tukey(p_envelope[:, itrc], tukey_alpha, nt_max_cut)

    p_envelope = np.asarray(filtfilt(b, a, f_envelope, axis=0), dtype=np.float64)

    p_envelope[nt_max_cut:] = 0.

    fd_envelope[nt_max_cut:] = 0.

    mag_got = np.abs(signal_freq)

    # index of predicted brightest frequency
    arg_peak = np.argmax(mag_got)
    min_mag = min_mag_mult * np.max(mag_got)

    # fft phases
    angle_got = np.asarray(np.angle(signal_freq), dtype=np.float64)

    # find the frequency range where both signals have non-trivial amplitude
    itr_low_cut = int(np.argmax(mag_got[:, 0] >= min_mag))
    itr_high_cut = int(max(itr_low_cut, int(mag_got.shape[0] - np.argmax(mag_got[::-1, 0] >= min_mag))))

    # trim out irrelevant/numerically unstable fft angles at faint amplitudes
    angle_got[:itr_low_cut] = 0.
    angle_got[itr_high_cut:] = 0.

    # unwrap the fft phases by 2 pi
    angle_got = np.unwrap(angle_got, axis=0)

    # standardize the factors of 2 pi to the predicted peak frequency
    angle_got -= angle_got[arg_peak] - angle_got[arg_peak] % (2 * np.pi)

    return wavelet_dense, signal_time, signal_freq, mag_got, angle_got, envelope, p_envelope, f_envelope, fd_envelope, itr_low_cut, itr_high_cut


def multishape_method_match_helper(p_offset, f0_mult, f0p_mult, f0pp_mult, rr_model, gridsize2_mult):
    # get the config for the first (Nf, Nt) pair
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)
    Nf = int(config_in1['wavelet_constants']['Nf'])
    Nt = int(config_in1['wavelet_constants']['Nt'])

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    # replace the Nf and Nt from the file
    config_in2['wavelet_constants']['Nf'] = int(Nf // gridsize2_mult)
    config_in2['wavelet_constants']['Nt'] = int(Nt * gridsize2_mult)

    wc = get_wavelet_model(config_in2)
    lc = get_lisa_constants(config_in2)
    taylor_time_table = get_taylor_table_time(wc, cache_mode='check', output_mode='hf')

    f_high_cut = 1 / (2 * wc.dt) - 1 / wc.Tobs
    nc_waveform = 1
    tukey_alpha_in = 0.
    tukey_alpha = 0.1

    # Create fake input data for a pure sine wave
    p_input = p_offset
    f_input = wc.DF * wc.Nf * f0_mult
    # fp_input = f0p_mult * f_input / wc.Tobs
    fp_input = f0p_mult * wc.dfd
    fpp_input = f0pp_mult * f_input / wc.Tobs**2
    amp_input = 1.0

    f_lowpass = 10 * wc.df_bw  # f_input / 2000.0 #wc1.DF*(wc1.Nf/4)

    AET_waveform, _ = get_aet_waveform_helper(lc, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha_in, wc.Nt, wc.DT)

    # get the sparse wavelet intrinsic_waveform
    wavelet_waveform1 = get_empty_sparse_taylor_time_waveform(nc_waveform, wc)

    wavelet_waveform2 = get_empty_sparse_taylor_time_waveform(nc_waveform, wc)

    nt_lim = PixelTimeRange(0, wc.Nt)

    # call wavemaket
    wavemaket(wavelet_waveform1, AET_waveform, nt_lim, wc, taylor_time_table, force_nulls=False)

    wavemaket_direct(wavelet_waveform2, AET_waveform, nt_lim, wc, taylor_time_table)

    wavelet_dense1, _, signal_freq1, mag_got1, angle_got1, envelope1, p_envelope1, f_envelope1, fd_envelope1, itr_low_cut1, itr_high_cut1 = get_wavelet_alternative_representation_helper(wavelet_waveform1, wc, tukey_alpha, f_lowpass, 8.e-2)
    wavelet_dense2, _, signal_freq2, mag_got2, angle_got2, envelope2, p_envelope2, f_envelope2, fd_envelope2, itr_low_cut2, itr_high_cut2 = get_wavelet_alternative_representation_helper(wavelet_waveform2, wc, tukey_alpha, f_lowpass, 8.e-2)
    itr_low_cut = max(itr_low_cut1, itr_low_cut2)
    itr_high_cut = min(itr_high_cut1, itr_high_cut2)

    # check the average time domain phase also agrees
    dp_envelope = (np.mean(p_envelope1 - p_envelope2, axis=0) + np.pi) % (2 * np.pi) - np.pi

    mag_peak = mag_got1.max()

    angle_got_diff = (angle_got1[itr_low_cut:itr_high_cut] - angle_got2[itr_low_cut:itr_high_cut] + np.pi) % (2 * np.pi) - np.pi

    # check the full signals are similar in the part of the intrinsic_waveform that is bright
    diff_bright = signal_freq2[itr_low_cut:itr_high_cut, 0] - signal_freq1[itr_low_cut:itr_high_cut, 0]
    abs_diff_bright = np.abs(diff_bright)**2

    power_got_bright_freq = np.sum(mag_got1[itr_low_cut:itr_high_cut]**2)
    bright_power_diff = np.sum(abs_diff_bright) / power_got_bright_freq

    # check the total signal power is similar
    power_got1 = np.sum(wavelet_dense1**2)
    power_got2 = np.sum(wavelet_dense2**2)
    power_diff = np.abs((power_got1 - power_got2) / power_got1)

    angle_got_diff_mean = np.mean(angle_got_diff, axis=0)
    assert_allclose(envelope1, envelope2, atol=1.e-2, rtol=1.e-2)

    assert_allclose(f_envelope1, f_envelope2, atol=1.e-2 * f_input, rtol=1.e-4)
    assert_allclose(np.mean(f_envelope1 - f_envelope2), 0., atol=1.e-3 * f_input)

    assert_allclose(dp_envelope, 0., atol=1.e-1)

    # check no systematic difference in the phase accumulate
    assert_allclose(np.abs(angle_got_diff_mean), 0., atol=5.e-3)

    # check magnitudes match in the mutually bright range
    assert_allclose(mag_got1[itr_low_cut:itr_high_cut] / mag_peak, mag_got2[itr_low_cut:itr_high_cut] / mag_peak, atol=3.e-2, rtol=1.e-3)

    assert bright_power_diff < 1.e-3
    assert power_diff < 5.e-2

    assert_allclose(fd_envelope1, fd_envelope2, atol=1.e2 * f_input / wc.Tobs, rtol=1.e-4)
    assert_allclose(np.mean(fd_envelope1 - fd_envelope2), 0., atol=2.e-2 * max(np.max(np.abs(fd_envelope1)), np.max(np.abs(fd_envelope2))))

    mask = (wavelet_dense1 != 0.) & (wavelet_dense2 != 0.)
    assert_allclose(wavelet_dense1[mask], wavelet_dense2[mask], atol=6.e-3, rtol=1.e-10)

    # check phases match in the mutually bright range
    assert_allclose(angle_got_diff, 0., atol=1.e-1, rtol=1.e-2)


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.4, 0.5,
    ],
)
@pytest.mark.parametrize('f0p_mult', [-31., 31., -20., 20., -15., 15., -10., 10., -9.999999, 9.999999, -3.25, 3.25, -3.75, 3.75, 1. / 3., -1. / 3., 2. / 3., -2. / 3., -2.000001, 2.000001, -2., 2., -1.99999, 1.99999, -1.5, 1.5, -1.00001, 1.00001, -1., 1., -0.99999, 0.99999, -0.5, 0.5, -0.001, 0., 0.001])
@pytest.mark.parametrize('f0pp_mult', [0.0])
@pytest.mark.parametrize('rr_model', ['const'])
@pytest.mark.parametrize('gridsize2_mult', [1])
@pytest.mark.parametrize('p_offset', [0., np.pi / 2.])
# @pytest.mark.skip
def test_wavemaket_method_match_slopes(p_offset, f0_mult, f0p_mult, f0pp_mult, rr_model, gridsize2_mult):
    """Test whether the signal computed in the time domain matches computing
    it in the wavelet domain with several different pixel grid sizes for a galactic binary
    with a moderately large second derivative:
    """
    multishape_method_match_helper(p_offset, f0_mult, f0p_mult, f0pp_mult, rr_model, gridsize2_mult)


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.4, 0.5,
    ],
)
@pytest.mark.parametrize('f0p_mult', [0.])
@pytest.mark.parametrize('f0pp_mult', [-0.1, 0.1, 0.0])
@pytest.mark.parametrize('rr_model', ['const'])
@pytest.mark.parametrize('gridsize2_mult', [32, 16, 8, 4, 2, 1, 0.5])
@pytest.mark.parametrize('p_offset', [np.pi / 2., 0.])
# @pytest.mark.skip
def test_wavemaket_multishape_method_match(p_offset, f0_mult, f0p_mult, f0pp_mult, rr_model, gridsize2_mult):
    """Test whether the signal computed in the time domain matches computing
    it in the wavelet domain with several different pixel grid sizes for a galactic binary
    with a moderately large second derivative:
    """
    multishape_method_match_helper(p_offset, f0_mult, f0p_mult, f0pp_mult, rr_model, gridsize2_mult)


@pytest.mark.parametrize('direct', [True, False])
@pytest.mark.parametrize(
    'f0_mult',
    [
        0.4, 0.5,
    ],
)
@pytest.mark.parametrize('p_offset', [0., np.pi / 2.])
# @pytest.mark.skip
def test_wavemaket_extreme_size(p_offset, f0_mult, direct):
    """Test with maximally rapid oscillations in FTd that integrate out to constant frequency"""
    # get the config for the first (Nf, Nt) pair
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    taylor_time_table = get_taylor_table_time(wc, cache_mode='check', output_mode='hf')

    f_lowpass = 10 * wc.df_bw  # f_input / 2000.0 #wc1.DF*(wc1.Nf/4)
    b, a = butter(4, f_lowpass, fs=1. / wc.dt, btype='low', analog=False)

    nc_waveform = 1
    tukey_alpha = 0.1

    nd_loc = wc.Nf * wc.Nt

    dfmax = wc.dfd * (wc.Nfd - wc.Nfd_negative - 1.0001)  # maximum possible df

    f_input = wc.DF * wc.Nf * f0_mult

    T = np.arange(wc.Nt) * wc.DT

    T_fine = wc.dt * np.arange(0, nd_loc)
    fd_design = 1.0 * dfmax * (-1)**(np.arange(0, nd_loc) // (64 * wc.Nf))
    # fd_design = dfmax*np.cos(w_design*T_fine)
    f_design = cumtrapz(fd_design, T_fine, initial=0)
    p_design = 2 * np.pi * cumtrapz(f_design, T_fine, initial=0) + p_offset
    f_design += f_input
    p_design += 2 * np.pi * (f_input * T_fine)
    assert np.allclose(f_design, cumtrapz(fd_design, T_fine, initial=0) + f_input)
    a_design = np.full(nd_loc, 1.)
    time_design = a_design * np.cos(p_design)

    envelope_design = np.abs(hilbert(time_design, axis=0))

    wavelet_design = transform_wavelet_freq_time(time_design, wc.Nf, wc.Nt)

    AET_PT = np.zeros((nc_waveform, wc.Nt))
    AET_FT = np.zeros((nc_waveform, wc.Nt))
    AET_FTd = np.zeros((nc_waveform, wc.Nt))
    AET_AT = np.zeros((nc_waveform, wc.Nt))

    # amplitude constants
    AET_AT[0, :] = a_design[::wc.Nf]
    AET_FTd[0, :] = fd_design[::wc.Nf]
    AET_FT[0, :] = f_design[::wc.Nf]
    AET_PT[0, :] = p_design[::wc.Nf]

    AET_waveform = StationaryWaveformTime(T, AET_PT, AET_FT, AET_FTd, AET_AT)

    wavelet_waveform = get_empty_sparse_taylor_time_waveform(nc_waveform, wc)

    nt_lim = PixelTimeRange(0, wc.Nt)

    if direct:
        wavemaket_direct(wavelet_waveform, AET_waveform, nt_lim, wc, taylor_time_table)
    else:
        wavemaket(wavelet_waveform, AET_waveform, nt_lim, wc, taylor_time_table, force_nulls=False)
    print(wavelet_waveform.n_set)

    wavelet_dense, _, _, _, _, envelope, _, f_envelope, fd_envelope, _, _ = get_wavelet_alternative_representation_helper(wavelet_waveform, wc, tukey_alpha, f_lowpass)

    # mimic filtering for plotting

    envelope_design = np.asarray(filtfilt(b, a, envelope_design, axis=0), dtype=np.float64)

    tukey(envelope_design, tukey_alpha, envelope_design.shape[0])

    tukey(f_design, tukey_alpha, f_design.shape[0])

    f_design = np.asarray(filtfilt(b, a, f_design, axis=0), dtype=np.float64)

    tukey(f_design, tukey_alpha, f_design.shape[0])

    tukey(fd_design, tukey_alpha, fd_design.shape[0])

    fd_design = np.asarray(filtfilt(b, a, fd_design, axis=0), dtype=np.float64)

    tukey(fd_design, tukey_alpha, fd_design.shape[0])

    tukey(p_design, tukey_alpha, p_design.shape[0])

    p_design = np.asarray(filtfilt(b, a, p_design, axis=0))

    assert_allclose(f_envelope[:, 0], f_design, atol=1.e-2 * np.max(f_design), rtol=1.e-2)
    assert_allclose(envelope[:, 0], envelope_design, atol=1.e-1 * np.max(envelope_design), rtol=1.e-1)
    assert_allclose(fd_envelope[:, 0], fd_design, atol=2.e-2 * float(np.max(fd_design)), rtol=1.e-2)
    mask = wavelet_dense[:, :, 0] != 0.

    # can't guarantee all pixels are good because of the corners, but most should be
    n_close = np.isclose(wavelet_dense[mask, 0], wavelet_design[mask], atol=1.e-1 * np.max(np.abs(wavelet_design)), rtol=1.e-1).sum()
    assert n_close > 0.99 * mask.sum()

    match = np.sum(wavelet_design * wavelet_dense[:, :, 0]) / np.sqrt(np.sum(wavelet_design**2)) / np.sqrt(np.sum(wavelet_design**2))
    resid = np.sum((wavelet_design - wavelet_dense[:, :, 0])**2) / np.sqrt(np.sum(wavelet_design**2)) / np.sqrt(np.sum(wavelet_design**2))
    assert_allclose(resid, 0., atol=1.e-1)
    assert match > 0.9


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.4, 0.57,
    ],
)
@pytest.mark.parametrize('f0p_mult', [0.])
@pytest.mark.parametrize('f0pp_mult', [1.0, -0.1])
@pytest.mark.parametrize('rr_model', ['const'])
@pytest.mark.parametrize('gridsize2_mult', [32, 16, 8, 4, 2, 0.5])
@pytest.mark.parametrize('p_offset', [0., np.pi / 2.])
@pytest.mark.parametrize('direct', [True, False])
# @pytest.mark.skip
def test_wavemaket_dimension_comparison_midevolve2(p_offset, f0_mult, f0p_mult, f0pp_mult, rr_model, gridsize2_mult, direct):
    """Test whether the signal computed in the time domain matches computing
    it in the wavelet domain with several different pixel grid sizes for a galactic binary
    with a moderately large second derivative:
    """
    # get the config for the first (Nf, Nt) pair
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

    wc1 = get_wavelet_model(config_in1)
    lc1 = get_lisa_constants(config_in1)
    taylor_time_table1 = get_taylor_table_time(wc1, cache_mode='check', output_mode='hf')

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    # replace the Nf and Nt from the file
    config_in2['wavelet_constants']['Nf'] = int(wc1.Nf // gridsize2_mult)
    config_in2['wavelet_constants']['Nt'] = int(wc1.Nt * gridsize2_mult)

    wc2 = get_wavelet_model(config_in2)
    lc2 = get_lisa_constants(config_in2)
    taylor_time_table2 = get_taylor_table_time(wc2, cache_mode='check', output_mode='hf')

    nd_loc1 = wc1.Nt * wc1.Nf
    nd_loc2 = wc2.Nt * wc2.Nf
    assert nd_loc1 == nd_loc2
    assert wc1.dt == wc2.dt
    assert lc1 == lc2

    f_high_cut = 1 / (2 * wc1.dt) - 1 / wc1.Tobs
    nc_waveform = 1
    tukey_alpha_in = 0.2
    tukey_alpha = 0.1

    # Create fake input data for a pure sine wave
    p_input = p_offset
    f_input = wc1.DF * wc1.Nf * f0_mult
    fp_input = f0p_mult * wc1.DF * wc1.Nf * 0.4 / wc1.Tobs
    fpp_input = f0pp_mult * wc1.DF * wc1.Nf * 0.4 / wc1.Tobs**2
    amp_input = 1.0

    fp_max = fp_input + fpp_input * wc1.Tobs
    fp_max_allow1 = 8 * wc1.DF / wc1.Tw
    fp_max_allow2 = 8 * wc2.DF / wc2.Tw
    fp_max_avail1 = wc1.dfd * (wc1.Nfd - wc1.Nfd_negative)
    fp_max_avail2 = wc2.dfd * (wc2.Nfd - wc2.Nfd_negative)
    print(fp_max, fp_max_allow1, fp_max_allow2, fp_max_avail1, fp_max_avail2)
    assert fp_max < fp_max_allow1
    assert fp_max < fp_max_allow2
    assert fp_max < fp_max_avail1
    # skip cases where there isn't a big enough grid to evaluate
    if not fp_max < fp_max_avail2:
        return

    f_lowpass = 10 * wc1.df_bw  # f_input / 2000.0 #wc1.DF*(wc1.Nf/4)

    AET_waveform1, _ = get_aet_waveform_helper(lc1, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha_in, wc1.Nt, wc1.DT)
    AET_waveform2, _ = get_aet_waveform_helper(lc2, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha_in, wc2.Nt, wc2.DT)

    # get the sparse wavelet intrinsic_waveform
    wavelet_waveform1 = get_empty_sparse_taylor_time_waveform(nc_waveform, wc1)
    wavelet_waveform2 = get_empty_sparse_taylor_time_waveform(nc_waveform, wc2)

    nt_lim1 = PixelTimeRange(0, wc1.Nt)
    nt_lim2 = PixelTimeRange(0, wc2.Nt)
    # call wavemaket
    if direct:
        wavemaket_direct(wavelet_waveform1, AET_waveform1, nt_lim1, wc1, taylor_time_table1)
        wavemaket_direct(wavelet_waveform2, AET_waveform2, nt_lim2, wc2, taylor_time_table2)
    else:
        wavemaket(wavelet_waveform1, AET_waveform1, nt_lim1, wc1, taylor_time_table1, force_nulls=False)
        wavemaket(wavelet_waveform2, AET_waveform2, nt_lim2, wc2, taylor_time_table2, force_nulls=False)

    wavelet_dense1, _, signal_freq1, mag_got1, angle_got1, envelope1, p_envelope1, f_envelope1, fd_envelope1, itr_low_cut1, itr_high_cut1 = get_wavelet_alternative_representation_helper(wavelet_waveform1, wc1, tukey_alpha, f_lowpass, 8.e-2)
    wavelet_dense2, _, signal_freq2, mag_got2, angle_got2, envelope2, p_envelope2, f_envelope2, fd_envelope2, itr_low_cut2, itr_high_cut2 = get_wavelet_alternative_representation_helper(wavelet_waveform2, wc2, tukey_alpha, f_lowpass, 8.e-2)
    itr_low_cut = max(itr_low_cut1, itr_low_cut2)
    itr_high_cut = min(itr_high_cut1, itr_high_cut2)

    # check the average time domain phase also agrees
    dp_envelope = (np.mean(p_envelope1 - p_envelope2, axis=0) + np.pi) % (2 * np.pi) - np.pi

    mag_peak = mag_got1.max()

    angle_got_diff = (angle_got1[itr_low_cut:itr_high_cut] - angle_got2[itr_low_cut:itr_high_cut] + np.pi) % (2 * np.pi) - np.pi

    # check the full signals are similar in the part of the intrinsic_waveform that is bright
    diff_bright = signal_freq2[itr_low_cut:itr_high_cut, 0] - signal_freq1[itr_low_cut:itr_high_cut, 0]
    abs_diff_bright = np.abs(diff_bright)**2

    power_got_bright_freq = np.sum(mag_got1[itr_low_cut:itr_high_cut]**2)
    bright_power_diff = np.sum(abs_diff_bright) / power_got_bright_freq

    # check the total signal power is similar
    power_got1 = np.sum(wavelet_dense1**2)
    power_got2 = np.sum(wavelet_dense2**2)
    power_diff = np.abs((power_got1 - power_got2) / power_got1)

    angle_got_diff_mean = np.mean(angle_got_diff, axis=0)

    assert_allclose(envelope1, envelope2, atol=1.e-2, rtol=1.e-2)
    assert_allclose(f_envelope1, f_envelope2, atol=2.e-2 * f_input, rtol=1.e-4)
    assert_allclose(np.mean(f_envelope1 - f_envelope2), 0., atol=2.e-5 * f_input)

    max_fd_envelope = max(np.max(np.abs(fd_envelope1)), np.max(np.abs(fd_envelope2)))
    assert_allclose(fd_envelope1, fd_envelope2, atol=5.e-1 * max_fd_envelope, rtol=1.e-4)
    assert_allclose(np.mean(fd_envelope1 - fd_envelope2), 0., atol=1.e-3 * max_fd_envelope)

    assert_allclose(dp_envelope, 0., atol=1.e-1)

    # check phases match in the mutually bright range
    assert_allclose(angle_got_diff, 0., atol=4.e-1, rtol=1.e-2)

    # check no systematic difference in the phase accumulate
    assert_allclose(np.abs(angle_got_diff_mean), 0., atol=5.e-3)

    # check magnitudes match in the mutually bright range
    assert_allclose(mag_got1[itr_low_cut:itr_high_cut] / mag_peak, mag_got2[itr_low_cut:itr_high_cut] / mag_peak, atol=6.e-2, rtol=6.e-2)
    assert bright_power_diff < 1.e-3
    assert power_diff < 5.e-2


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.1, 1 / 16, 9 / 128, 5 / 64, 3 / 32, 1 / 8, 3 / 16, 1 / 4, 1 / 2,
    ],
)
@pytest.mark.parametrize('f0p_mult', [0., 0.001])
@pytest.mark.parametrize('f0pp_mult', [0., 0.001])
@pytest.mark.parametrize('rr_model', ['const'])
@pytest.mark.parametrize('gridsize2_mult', [16, 8, 4, 2, 0.5])
@pytest.mark.parametrize('p_offset', [0., np.pi / 2.])
@pytest.mark.parametrize('direct', [True, False])
# @pytest.mark.skip
def test_wavemaket_dimension_comparison_slowevolve(p_offset, f0_mult, f0p_mult, f0pp_mult, rr_model, gridsize2_mult, direct):
    """Test whether the signal computed in the time domain matches computing
    it in the wavelet domain with several different pixel grid sizes for a galactic binary
    with a moderately large second derivative:
    """
    # get the config for the first (Nf, Nt) pair
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

    wc1 = get_wavelet_model(config_in1)
    lc1 = get_lisa_constants(config_in1)
    taylor_time_table1 = get_taylor_table_time(wc1, cache_mode='check', output_mode='hf')

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    # replace the Nf and Nt from the file
    config_in2['wavelet_constants']['Nf'] = int(wc1.Nf // gridsize2_mult)
    config_in2['wavelet_constants']['Nt'] = int(wc1.Nt * gridsize2_mult)

    wc2 = get_wavelet_model(config_in2)
    lc2 = get_lisa_constants(config_in2)
    taylor_time_table2 = get_taylor_table_time(wc2, cache_mode='check', output_mode='hf')

    nd_loc1 = wc1.Nt * wc1.Nf
    nd_loc2 = wc2.Nt * wc2.Nf
    assert nd_loc1 == nd_loc2
    assert wc1.dt == wc2.dt
    assert lc1 == lc2

    f_high_cut = 1 / (2 * wc1.dt) - 1 / wc1.Tobs
    nc_waveform = 1
    tukey_alpha_in = 0.01
    tukey_alpha = 0.1

    # Create fake input data for a pure sine wave
    p_input = p_offset
    f_input = wc1.DF * wc1.Nf * f0_mult
    fp_input = f0p_mult * f_input / wc1.Tobs
    fpp_input = f0pp_mult * f_input / wc1.Tobs**2
    amp_input = 1.0

    f_lowpass = 10 * wc1.df_bw  # f_input / 2000.0 #wc1.DF*(wc1.Nf/4)

    AET_waveform1, _ = get_aet_waveform_helper(lc1, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha_in, wc1.Nt, wc1.DT)
    AET_waveform2, _ = get_aet_waveform_helper(lc2, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha_in, wc2.Nt, wc2.DT)

    # get the sparse wavelet intrinsic_waveform
    wavelet_waveform1 = get_empty_sparse_taylor_time_waveform(nc_waveform, wc1)
    wavelet_waveform2 = get_empty_sparse_taylor_time_waveform(nc_waveform, wc2)

    nt_lim1 = PixelTimeRange(0, wc1.Nt)
    nt_lim2 = PixelTimeRange(0, wc2.Nt)

    # call wavemaket
    if direct:
        wavemaket_direct(wavelet_waveform1, AET_waveform1, nt_lim1, wc1, taylor_time_table1)
        wavemaket_direct(wavelet_waveform2, AET_waveform2, nt_lim2, wc2, taylor_time_table2)
    else:
        wavemaket(wavelet_waveform1, AET_waveform1, nt_lim1, wc1, taylor_time_table1, force_nulls=False)
        wavemaket(wavelet_waveform2, AET_waveform2, nt_lim2, wc2, taylor_time_table2, force_nulls=False)

    wavelet_dense1, _, signal_freq1, mag_got1, angle_got1, envelope1, p_envelope1, f_envelope1, fd_envelope1, itr_low_cut1, itr_high_cut1 = get_wavelet_alternative_representation_helper(wavelet_waveform1, wc1, tukey_alpha, f_lowpass, 8.e-2)
    wavelet_dense2, _, signal_freq2, mag_got2, angle_got2, envelope2, p_envelope2, f_envelope2, fd_envelope2, itr_low_cut2, itr_high_cut2 = get_wavelet_alternative_representation_helper(wavelet_waveform2, wc2, tukey_alpha, f_lowpass, 8.e-2)
    itr_low_cut = max(itr_low_cut1, itr_low_cut2)
    itr_high_cut = min(itr_high_cut1, itr_high_cut2)

    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(0, wc1.Nf*wc1.Nt)*wc1.dt/(3600*24*365), signal_time1[:,0])
    # plt.xlabel('Time [yr]')
    # plt.ylabel('Amplitude')
    # plt.title('Sine Wave, %5.3f s sampling: ~%9d parameters' % (wc1.dt, wc1.Nf*wc1.Nt))
    # plt.show()

    # plt.plot(np.fft.rfftfreq(wc1.Nf*wc1.Nt, d=wc1.dt)*1000, mag_got1[:,0])
    # plt.xlabel('f [mHz]')
    # plt.ylabel('Amplitude')
    # plt.title('FFT Sine Wave: ~2 parameters')
    # plt.show()

    # print(wc1.Nf*wc1.DF, f_input)
    # plt.plot(np.arange(0, wc1.Nf*wc1.Nt//2+1)/(2*wc1.dt), mag_got1[:,0])
    # extent = (0 * wc1.DT / gc.SECSYEAR, wc1.Nt * wc1.DT / gc.SECSYEAR, 0 * wc1.DF*1000, wc1.Nf * wc1.DF*100)
    # plt.imshow(np.rot90(wavelet_dense1[:,:,0]**2), aspect='auto', extent=extent, cmap='YlOrRd')
    # plt.ylabel('f [mHz]')
    # plt.xlabel('t [yr]')
    # plt.title('Wavelet Sine Wave: ~%5d parameters' % wavelet_waveform1.n_set[0])
    # plt.show()
    # assert False

    assert_allclose(envelope1, envelope2, atol=1.e-2, rtol=1.e-2)
    assert_allclose(f_envelope1, f_envelope2, atol=1.e-2 * f_input, rtol=1.e-4)
    assert_allclose(np.mean(f_envelope1 - f_envelope2), 0., atol=6.e-5 * f_input)

    assert_allclose(fd_envelope1, fd_envelope2, atol=1.e2 * f_input / wc1.Tobs, rtol=1.e-4)
    assert_allclose(np.mean(fd_envelope1 - fd_envelope2), 0., atol=1.e-4 * f_input / wc1.Tobs)

    # check the average time domain phase also agrees
    dp_envelope = (np.mean(p_envelope1 - p_envelope2, axis=0) + np.pi) % (2 * np.pi) - np.pi
    assert_allclose(dp_envelope, 0., atol=1.e-1)

    mag_peak = mag_got1.max()

    angle_got_diff = (angle_got1[itr_low_cut:itr_high_cut] - angle_got2[itr_low_cut:itr_high_cut] + np.pi) % (2 * np.pi) - np.pi

    # check the full signals are similar in the part of the intrinsic_waveform that is bright
    diff_bright = signal_freq2[itr_low_cut:itr_high_cut, 0] - signal_freq1[itr_low_cut:itr_high_cut, 0]
    abs_diff_bright = np.abs(diff_bright)**2

    power_got_bright_freq = np.sum(mag_got1[itr_low_cut:itr_high_cut]**2)
    bright_power_diff = np.sum(abs_diff_bright) / power_got_bright_freq

    # check the total signal power is similar
    power_got1 = np.sum(wavelet_dense1**2)
    power_got2 = np.sum(wavelet_dense2**2)
    power_diff = np.abs((power_got1 - power_got2) / power_got1)

    angle_got_diff_mean = np.mean(angle_got_diff, axis=0)

    # check phases match in the mutually bright range
    assert_allclose(angle_got_diff, 0., atol=3.e-1, rtol=1.e-2)

    # check no systematic difference in the phase accumulate
    assert_allclose(np.abs(angle_got_diff_mean), 0., atol=1.e-1)

    # check magnitudes match in the mutually bright range
    assert_allclose(mag_got1[itr_low_cut:itr_high_cut] / mag_peak, mag_got2[itr_low_cut:itr_high_cut] / mag_peak, atol=3.e-2, rtol=1.e-3)
    assert bright_power_diff < 1.e-3
    assert power_diff < 5.e-2


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.5,
    ],
)
@pytest.mark.parametrize('f0p_mult', [0.])
@pytest.mark.parametrize('f0pp_mult', [0.1])
@pytest.mark.parametrize('rr_model', ['const'])
@pytest.mark.parametrize('gridsize2_mult', [32, 16, 8, 4, 2, 0.5])
@pytest.mark.parametrize('p_offset', [0., np.pi / 2.])
@pytest.mark.parametrize('direct', [True, False])
# @pytest.mark.skip
def test_wavemaket_dimension_comparison_midevolve(p_offset, f0_mult, f0p_mult, f0pp_mult, rr_model, gridsize2_mult, direct):
    """Test whether the signal computed in the time domain matches computing
    it in the wavelet domain with several different pixel grid sizes for a galactic binary
    with a moderately large second derivative:
    """
    # get the config for the first (Nf, Nt) pair
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

    wc1 = get_wavelet_model(config_in1)
    lc1 = get_lisa_constants(config_in1)
    taylor_time_table1 = get_taylor_table_time(wc1, cache_mode='check', output_mode='hf')

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    # replace the Nf and Nt from the file
    config_in2['wavelet_constants']['Nf'] = int(wc1.Nf // gridsize2_mult)
    config_in2['wavelet_constants']['Nt'] = int(wc1.Nt * gridsize2_mult)

    wc2 = get_wavelet_model(config_in2)
    lc2 = get_lisa_constants(config_in2)
    taylor_time_table2 = get_taylor_table_time(wc2, cache_mode='check', output_mode='hf')

    nd_loc1 = wc1.Nt * wc1.Nf
    nd_loc2 = wc2.Nt * wc2.Nf
    assert nd_loc1 == nd_loc2
    assert wc1.dt == wc2.dt
    assert lc1 == lc2

    f_high_cut = 1 / (2 * wc1.dt) - 1 / wc1.Tobs
    nc_waveform = 1
    tukey_alpha_in = 0.1
    tukey_alpha = 0.1

    # Create fake input data for a pure sine wave
    p_input = p_offset
    f_input = wc1.DF * wc1.Nf * f0_mult
    fp_input = f0p_mult * f_input / wc1.Tobs
    fpp_input = f0pp_mult * f_input / wc1.Tobs**2
    amp_input = 1.0

    f_lowpass = 10 * wc1.df_bw  # f_input / 2000.0 #wc1.DF*(wc1.Nf/4)

    AET_waveform1, _ = get_aet_waveform_helper(lc1, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha_in, wc1.Nt, wc1.DT)
    AET_waveform2, _ = get_aet_waveform_helper(lc2, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha_in, wc2.Nt, wc2.DT)

    # get the sparse wavelet intrinsic_waveform
    wavelet_waveform1 = get_empty_sparse_taylor_time_waveform(nc_waveform, wc1)
    wavelet_waveform2 = get_empty_sparse_taylor_time_waveform(nc_waveform, wc2)

    nt_lim1 = PixelTimeRange(0, wc1.Nt)
    nt_lim2 = PixelTimeRange(0, wc2.Nt)

    # call wavemaket
    if direct:
        wavemaket_direct(wavelet_waveform1, AET_waveform1, nt_lim1, wc1, taylor_time_table1)
        wavemaket_direct(wavelet_waveform2, AET_waveform2, nt_lim2, wc2, taylor_time_table2)
    else:
        wavemaket(wavelet_waveform1, AET_waveform1, nt_lim1, wc1, taylor_time_table1, force_nulls=False)
        wavemaket(wavelet_waveform2, AET_waveform2, nt_lim2, wc2, taylor_time_table2, force_nulls=False)

    wavelet_dense1, _, signal_freq1, mag_got1, angle_got1, envelope1, p_envelope1, f_envelope1, fd_envelope1, itr_low_cut1, itr_high_cut1 = get_wavelet_alternative_representation_helper(wavelet_waveform1, wc1, tukey_alpha, f_lowpass, 8.e-2)
    wavelet_dense2, _, signal_freq2, mag_got2, angle_got2, envelope2, p_envelope2, f_envelope2, fd_envelope2, itr_low_cut2, itr_high_cut2 = get_wavelet_alternative_representation_helper(wavelet_waveform2, wc2, tukey_alpha, f_lowpass, 8.e-2)
    itr_low_cut = max(itr_low_cut1, itr_low_cut2)
    itr_high_cut = min(itr_high_cut1, itr_high_cut2)

    assert_allclose(envelope1, envelope2, atol=1.e-2, rtol=1.e-2)

    assert_allclose(f_envelope1, f_envelope2, atol=1.e-2 * f_input, rtol=1.e-4)
    assert_allclose(np.mean(f_envelope1 - f_envelope2), 0., atol=5.e-6 * f_input)

    assert_allclose(fd_envelope1, fd_envelope2, atol=1.e1 * f_input / wc1.Tobs, rtol=1.e-4)
    assert_allclose(np.mean(fd_envelope1 - fd_envelope2), 0., atol=1.e-4 * f_input / wc1.Tobs)

    # check the average time domain phase also agrees
    dp_envelope = (np.mean(p_envelope1 - p_envelope2, axis=0) + np.pi) % (2 * np.pi) - np.pi
    assert_allclose(dp_envelope, 0., atol=1.e-1)

    mag_peak = mag_got1.max()

    angle_got_diff = (angle_got1[itr_low_cut:itr_high_cut] - angle_got2[itr_low_cut:itr_high_cut] + np.pi) % (2 * np.pi) - np.pi

    # check the full signals are similar in the part of the intrinsic_waveform that is bright
    diff_bright = signal_freq2[itr_low_cut:itr_high_cut, 0] - signal_freq1[itr_low_cut:itr_high_cut, 0]
    abs_diff_bright = np.abs(diff_bright)**2

    power_got_bright_freq = np.sum(mag_got1[itr_low_cut:itr_high_cut]**2)
    bright_power_diff = np.sum(abs_diff_bright) / power_got_bright_freq

    # check the total signal power is similar
    power_got1 = np.sum(wavelet_dense1**2)
    power_got2 = np.sum(wavelet_dense2**2)
    power_diff = np.abs((power_got1 - power_got2) / power_got1)

    angle_got_diff_mean = np.mean(angle_got_diff, axis=0)

    # check phases match in the mutually bright range
    assert_allclose(angle_got_diff, 0., atol=1.e-1, rtol=1.e-2)

    # check no systematic difference in the phase accumulate
    assert_allclose(np.abs(angle_got_diff_mean), 0., atol=1.e-3)

    # check magnitudes match in the mutually bright range
    assert_allclose(mag_got1[itr_low_cut:itr_high_cut] / mag_peak, mag_got2[itr_low_cut:itr_high_cut] / mag_peak, atol=3.e-2, rtol=1.e-3)
    assert bright_power_diff < 1.e-3
    assert power_diff < 1.e-2


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.4,
    ],
)
@pytest.mark.parametrize('f0p_mult', [0.])
@pytest.mark.parametrize('f0pp_mult', [-0.8])
@pytest.mark.parametrize('rr_model', ['const'])
# @pytest.mark.skip
def test_wavemaket_1d(f0_mult, f0p_mult, f0pp_mult, rr_model):
    """Test whether the signal computed in the time domain matches computing
    it in the wavelet domain and transforming to time.
    """
    # get the config for the first (Nf, Nt) pair
    toml_filename_in = 'tests/wavemaket_test_config2.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

    wc1 = get_wavelet_model(config_in1)
    lc1 = get_lisa_constants(config_in1)
    taylor_time_table1 = get_taylor_table_time(wc1, cache_mode='check', output_mode='hf')

    nt_loc = wc1.Nt
    nd_loc = wc1.Nt * wc1.Nf
    f_lowpass = wc1.DF * (wc1.Nf - 1)
    dt_loc = wc1.dt
    f_high_cut = 1 / (2 * wc1.dt) - 1 / wc1.Tobs
    nc_waveform = 1
    tukey_alpha = 0.2

    # Create fake input data for a pure sine wave
    p_input = 0.
    f_input = wc1.DF * wc1.Nf * f0_mult
    fp_input = f0p_mult * f_input / wc1.Tobs
    fpp_input = f0pp_mult * f_input / wc1.Tobs**2
    amp_input = 1.0

    AET_waveform_fine, _ = get_aet_waveform_helper(lc1, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha, nd_loc, dt_loc)

    # get the time domain signal from the fine sampling
    # signal should be pure cos
    signal_time_pred_cos = (AET_waveform_fine.AT * np.cos(AET_waveform_fine.PT)).T

    # get predicted signals in the frequency domain
    signal_freq_pred_cos = np.zeros((nd_loc // 2 + 1, nc_waveform), dtype=np.complex128)
    for itrc in range(nc_waveform):
        signal_freq_pred_cos[:, itrc] = fft.rfft(signal_time_pred_cos[:, itrc])

    # get predicted signal in wavelet domain
    signal_wave_pred_cos = np.zeros((nt_loc, wc1.Nf, nc_waveform))
    for itrc in range(nc_waveform):
        signal_wave_pred_cos[:, :, itrc] = transform_wavelet_freq(signal_freq_pred_cos[:, itrc], wc1.Nf, nt_loc)

    AET_waveform, arg_cut = get_aet_waveform_helper(lc1, rr_model, p_input, f_input, fp_input, fpp_input, amp_input, nc_waveform, f_high_cut, tukey_alpha, nt_loc, wc1.DT)

    # get the sparse wavelet intrinsic_waveform
    wavelet_waveform = get_empty_sparse_taylor_time_waveform(nc_waveform, wc1)

    # call wavemaket
    nt_lim1 = PixelTimeRange(0, wc1.Nt)
    wavemaket(wavelet_waveform, AET_waveform, nt_lim1, wc1, taylor_time_table1, force_nulls=False)
    print(wavelet_waveform.n_set)
    wavelet_dense, signal_time, signal_freq, mag_got, angle_got, _, _, _, _, itr_low_cut, itr_high_cut = get_wavelet_alternative_representation_helper(wavelet_waveform, wc1, tukey_alpha, f_lowpass, 1.e-3)

    # isolate just the predicted wavelet values that we are in the sparse representation
    wavelet_waveform_sparse_cos = get_empty_sparse_taylor_time_waveform(nc_waveform, wc1)
    for itrc in range(nc_waveform):
        wave_pred_sparse_cos = signal_wave_pred_cos.reshape((nt_loc * wc1.Nf, nc_waveform))[wavelet_waveform.pixel_index[itrc, :wavelet_waveform.n_set[itrc]], itrc]

        wavelet_waveform_sparse_cos.wave_value[itrc, :wavelet_waveform.n_set[itrc]] = wave_pred_sparse_cos

        wavelet_waveform_sparse_cos.pixel_index[itrc, :wavelet_waveform.n_set[itrc]] = wavelet_waveform.pixel_index[itrc, :wavelet_waveform.n_set[itrc]]

        wavelet_waveform_sparse_cos.n_set[itrc] = wavelet_waveform.n_set[itrc]

        wave_got_sparse = wavelet_waveform.wave_value[itrc, :wavelet_waveform.n_set[itrc]]

    # get dense representation of just the part of the intrinsic_waveform that matches
    signal_wave_pred_cos_matched = np.zeros((nt_loc * wc1.Nf, nc_waveform))

    sparse_addition_helper(wavelet_waveform_sparse_cos, signal_wave_pred_cos_matched)

    signal_wave_pred_cos_matched = signal_wave_pred_cos_matched.reshape((nt_loc, wc1.Nf, nc_waveform))

    # get dense representation of just the part of the intrinsic_waveform that is not in the sparse representation
    signal_wave_pred_cos_unmatched = signal_wave_pred_cos - signal_wave_pred_cos_matched

    # get the frequency domain signal from the wavelets for just the pixels mutually on
    signal_freq_cos_matched = np.zeros((nd_loc // 2 + 1, nc_waveform), dtype=np.complex128)
    for itrc in range(nc_waveform):
        signal_freq_cos_matched[:, itrc] = inverse_wavelet_freq(signal_wave_pred_cos_matched[:, :, itrc], wc1.Nf, nt_loc)

    # get the time domain signal from the wavelets for just the pixels mutually on
    signal_time_cos_matched = np.zeros((nd_loc, nc_waveform))
    for itrc in range(nc_waveform):
        signal_time_cos_matched[:, itrc] = inverse_wavelet_freq_time(signal_wave_pred_cos_matched[:, :, itrc], wc1.Nf, nt_loc)

    # compare the frequency domain representations for the pixels that match
    mag_pred = np.abs(signal_freq_cos_matched[:, 0])

    mag_peak = mag_got.max()

    # index of predicted brightest frequency
    arg_peak = np.argmax(mag_pred)

    # fft phases
    angle_pred = np.angle(signal_freq_pred_cos[:, 0])

    # unwrap the fft phases by 2 pi
    angle_pred = np.unwrap(angle_pred)

    # standardize the factors of 2 pi to the predicted peak frequency
    angle_pred -= angle_pred[arg_peak] - angle_pred[arg_peak] % (2 * np.pi)

    # trim out irrelevant/numerically unstable fft angles at faint amplitudes
    angle_pred[:itr_low_cut] = 0.
    angle_got[:itr_low_cut] = 0.
    angle_pred[itr_high_cut:] = 0.
    angle_got[itr_high_cut:] = 0.

    # check the full signals are similar in the part of the intrinsic_waveform that is bright
    abs_diff_bright = np.abs(signal_freq_pred_cos[:, 0] - signal_freq[:, 0])[itr_low_cut:itr_high_cut]**2

    power_got_bright_freq = np.sum(mag_got[itr_low_cut:itr_high_cut]**2)
    bright_power_diff = np.sum(abs_diff_bright) / power_got_bright_freq

    # check the total signal power is similar
    power_got = np.sum(wavelet_dense**2)
    power_pred = np.sum(signal_wave_pred_cos**2)
    power_diff = np.abs((power_got - power_pred) / power_got)

    # maximum predicted wavelet value to scale closeness check
    wave_pred_sparse_cos = wavelet_waveform_sparse_cos.wave_value[0, :wavelet_waveform.n_set[0]]
    wave_got_sparse = wavelet_waveform.wave_value[0, :wavelet_waveform.n_set[0]]
    max_wave = max(np.max(np.abs(wave_pred_sparse_cos)), 1.e-5)

    nrm_sparse_got = np.linalg.norm(wave_got_sparse)
    nrm_sparse_cos = np.linalg.norm(wave_pred_sparse_cos)
    match_sparse_cos = np.sum(wave_got_sparse * wave_pred_sparse_cos) / (nrm_sparse_got * nrm_sparse_cos)
    resid_sparse = np.sum((wave_got_sparse - wave_pred_sparse_cos)**2) / (nrm_sparse_got * nrm_sparse_cos)

    # scaled maximum value at an unpredicted point
    residual_maximum = np.sqrt(np.max(signal_wave_pred_cos_unmatched[:, :, 0]**2)) / max_wave
    # scaled rms value for unpredicted points
    residual_rms = np.sqrt(np.mean(signal_wave_pred_cos_unmatched[:, :, 0]**2)) / max_wave

    # isolate stricter comparison to just the part largely unaffected by windowing
    nt_low_center = int((tukey_alpha + 0.05) * arg_cut)
    nt_high_center = max(int(arg_cut - nt_low_center), nt_low_center)
    residual_max_center = np.sqrt(np.max(signal_wave_pred_cos_unmatched[nt_low_center:nt_high_center, :, 0]**2)) / max_wave
    residual_rms_center = np.sqrt(np.mean(signal_wave_pred_cos_unmatched[nt_low_center:nt_high_center, :, 0]**2)) / max_wave

    match_cos = np.sum(signal_time * signal_time_pred_cos) / (
        np.linalg.norm(signal_time) * np.linalg.norm(signal_time_pred_cos)
    )
    resid = np.sum((signal_time - signal_time_pred_cos) ** 2) / (
        np.linalg.norm(signal_time) * np.linalg.norm(signal_time_pred_cos)
    )

    resid_matched = np.sum((signal_time - signal_time_cos_matched) ** 2) / (
        np.linalg.norm(signal_time) * np.linalg.norm(signal_time_cos_matched)
    )
    print(f_input)
    print(match_cos)
    print(resid, resid_matched)
    print('residual maximum', residual_maximum, residual_rms, residual_max_center, residual_rms_center)
    print(nt_low_center, nt_high_center)
    print(match_sparse_cos)
    print(resid_sparse)
    print(power_got, power_pred, power_diff)
    print(power_got_bright_freq, bright_power_diff)
    # check angles match in the mutually bright range
    assert_allclose(angle_pred[itr_low_cut:itr_high_cut], angle_got[itr_low_cut:itr_high_cut, 0], atol=1.e-2, rtol=3.e-2)
    # check magnitudes match in the mutually bright range
    assert_allclose(mag_pred[itr_low_cut:itr_high_cut] / mag_peak, mag_got[itr_low_cut:itr_high_cut, 0] / mag_peak, atol=3.e-2, rtol=1.e-3)
    assert bright_power_diff < 1.e-3
    assert power_diff < 1.e-2

    assert_allclose(wave_got_sparse, wave_pred_sparse_cos, atol=3.e-2 * max_wave, rtol=1.e-3)

    assert resid_sparse < 4.0e-3
    assert match_sparse_cos > 0.998

    assert residual_maximum < 2.e-3
    assert residual_rms < 1.e-5

    # assert residual_max_center < 1.e-12
    # assert residual_rms_center < 1.e-13

    assert residual_max_center < 1.e-3
    assert residual_rms_center < 1.e-5
    assert resid < 4.0e-3
    assert match_cos > 0.998
    assert resid_matched < 4.e-3


if __name__ == '__main__':
    pytest.cmdline.main(['tests/test_wavemaket.py'])
