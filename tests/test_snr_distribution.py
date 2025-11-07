"""Test that the computed SNR scales as expected with changes in (Nf, Nt, dt, mult)."""

from pathlib import Path

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose, assert_array_equal
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_freq, inverse_wavelet_time

from GalacticStochastic.testing_tools import unit_normal_battery
from LisaWaveformTools.instrument_noise import instrument_noise_AET, instrument_noise_AET_wdm_m
from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams, LinearFrequencyWaveletWaveformTime
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalStationaryDenseNoiseModel, get_sparse_likelihood_helper_prewhitened
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from WaveletWaveforms.sparse_waveform_functions import (
    PixelGenericRange,
    wavelet_dense_select_sparse,
    wavelet_sparse_to_dense,
    whiten_sparse_data,
)
from WaveletWaveforms.wdm_config import get_wavelet_model


@pytest.mark.parametrize(
    'channel_mult',
    [
        (1.0, 1.0, 1.0, 1.0),
    ],
)
@pytest.mark.parametrize('amp_mult', [1.0, 10.0])
@pytest.mark.parametrize('noise_curve_mode', [1, 0])
def test_noise_generation_scale_fix(channel_mult: tuple[float, float, float, float], amp_mult: float, noise_curve_mode: int) -> None:
    """Test recover same snr in wavelet, time, and frequency domains"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    if noise_curve_mode == 0:
        response_mode = 0
        amp0_use = 1.0e-20 * amp_mult
    else:
        response_mode = 2
        amp0_use = 1.0 * amp_mult

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    config_in['lisa_constants']['noise_curve_mode'] = noise_curve_mode
    config_in['wavelet_constants']['Nst'] = 512

    # replace the Nf and Nt from the file
    config_in['wavelet_constants']['Nf'] = int(config_in['wavelet_constants']['Nf'] * channel_mult[0])
    config_in['wavelet_constants']['Nt'] = int(config_in['wavelet_constants']['Nt'] * channel_mult[1])
    config_in['wavelet_constants']['dt'] = float(config_in['wavelet_constants']['dt'] * channel_mult[2])
    config_in['wavelet_constants']['mult'] = int(config_in['wavelet_constants']['mult'] * channel_mult[3])

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    noise = instrument_noise_AET_wdm_m(lc, wc)

    fs = np.arange(0, wc.Nf) * wc.DF

    noise_exp = np.zeros((fs.size, noise.shape[-1]))

    noise_exp[1:] = instrument_noise_AET(fs[1:], lc)  # /(2 * wc.dt)

    fs_full = np.fft.rfftfreq(wc.Nf * wc.Nt, d=wc.dt)

    noise_exp_full = np.zeros((fs_full.size, noise.shape[-1]))

    noise_exp_full[1:] = instrument_noise_AET(fs_full[1:], lc)  # /(2 * wc.dt)

    inv_chol_noise_exp_full = np.zeros((fs_full.size, noise.shape[-1]))
    inv_chol_noise_exp_full[1:] = np.sqrt(1. / noise_exp_full[1:])

    seed = 31415
    nc_snr = int(noise.shape[1])

    noise_manager = DiagonalStationaryDenseNoiseModel(noise, wc, prune=1, nc_snr=nc_snr, seed=seed)

    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=amp0_use,  # amplitude
        phi0=0.3,  # phase at t=0
        F0=1.0e-4,  # initial frequency (Hz)
        FTd0=0.,  # frequency derivative (Hz/s)
    )

    assert intrinsic.FTd0 < 8 * wc.DF / wc.Tw
    assert intrinsic.FTd0 < wc.dfd * (wc.Nfd - wc.Nfd_negative)
    assert intrinsic.FTd0 >= wc.dfd * (-wc.Nfd_negative)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)

    nt_lim_snr = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)

    waveform = LinearFrequencyWaveletWaveformTime(
        params,
        wc,
        lc,
        nt_lim_waveform,
        table_cache_mode='check',
        table_output_mode='skip',
        response_mode=response_mode,
    )

    wavelet_waveform = waveform.get_unsorted_coeffs()

    waveform_dense = wavelet_sparse_to_dense(wavelet_waveform, wc)

    waveform_dense_white = waveform_dense * noise_manager.get_inv_chol_S()

    signal_time_white = np.zeros((wc.Nf * wc.Nt, nc_snr))

    for itrc in range(nc_snr):
        signal_time_white[:, itrc] = inverse_wavelet_time(waveform_dense_white[:, :, itrc], wc.Nf, wc.Nt)

    signal_time = np.zeros((wc.Nf * wc.Nt, nc_snr))

    for itrc in range(nc_snr):
        signal_time[:, itrc] = inverse_wavelet_time(waveform_dense[:, :, itrc], wc.Nf, wc.Nt)

    signal_freq_alt = np.zeros(((wc.Nf * wc.Nt) // 2 + 1, nc_snr), dtype=np.complex128)

    for itrc in range(nc_snr):
        signal_freq_alt[:, itrc] = inverse_wavelet_freq(waveform_dense[:, :, itrc], wc.Nf, wc.Nt) / np.sqrt(wc.Nt * wc.Nf / 2)

    signal_freq_white = np.zeros(((wc.Nf * wc.Nt) // 2 + 1, nc_snr), dtype=np.complex128)

    for itrc in range(nc_snr):
        signal_freq_white[:, itrc] = inverse_wavelet_freq(waveform_dense_white[:, :, itrc], wc.Nf, wc.Nt) / np.sqrt(wc.Nt * wc.Nf / 2)

    signal_freq = np.fft.rfft(signal_time, axis=0) / np.sqrt(wc.Nt * wc.Nf / 2)
    signal_freq_white = np.fft.rfft(signal_time_white, axis=0) / np.sqrt(wc.Nt * wc.Nf / 2)

    scale_freq = float(np.max(np.abs(signal_freq)))
    assert_allclose(np.real(signal_freq[wc.Nt:]), np.real(signal_freq_alt[wc.Nt:]), atol=1.e-3 * scale_freq, rtol=1.e-4)
    assert_allclose(np.imag(signal_freq[wc.Nt:]), np.imag(signal_freq_alt[wc.Nt:]), atol=1.e-3 * scale_freq, rtol=1.e-4)

    snr_channel = noise_manager.get_sparse_snrs(wavelet_waveform, nt_lim_snr)

    template_amp_cos = np.trapezoid((signal_time.T * np.cos(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs
    template_amp_sin = np.trapezoid((signal_time.T * np.sin(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs
    template_amp_direct = 2 * np.trapezoid((signal_time.T * np.cos(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf) - intrinsic.phi0)), dx=wc.dt, axis=1) / wc.Tobs

    phase_got = np.arctan2(template_amp_sin, template_amp_cos)
    amp_got = 2 * np.sqrt(template_amp_cos**2 + template_amp_sin**2)

    arglim_min: int = int(max(wc.Nt, 2 * int(np.int64(np.pi * max(wc.Tobs / wc.Tw, wc.Tobs / wc.DT)))))
    snr_channel_got_freq = np.linalg.norm(np.abs(signal_freq) * np.sqrt(2 * wc.dt) * inv_chol_noise_exp_full, axis=0)
    assert_allclose(snr_channel, snr_channel_got_freq, atol=1.e-9, rtol=3.e-2)
    # required by parsevals theorem
    assert_allclose(np.sum(np.abs(signal_freq)**2, axis=0), np.sum(signal_time**2, axis=0), atol=1.e-10, rtol=1.e-7)
    assert_allclose(np.sum(signal_time**2, axis=0), np.sum(waveform_dense**2, axis=(0, 1)), atol=1.e-10, rtol=1.e-7)
    assert_allclose(np.sum(signal_time_white**2, axis=0), np.sum(waveform_dense_white**2, axis=(0, 1)), atol=1.e-7, rtol=1.e-7)
    assert_allclose(np.sum(np.abs(signal_freq)**2, axis=0), np.sum(waveform_dense**2, axis=(0, 1)), atol=1.e-10, rtol=1.e-7)
    assert_allclose(np.sum(np.abs(signal_freq_white)**2, axis=0), np.sum(signal_time_white**2, axis=0), atol=1.e-7, rtol=1.e-7)
    assert_allclose(np.sum(np.abs(signal_freq_white)[arglim_min:]**2, axis=0), 2 * wc.dt * np.sum(np.abs(signal_freq * inv_chol_noise_exp_full)[arglim_min:]**2, axis=0), atol=1.e-7, rtol=1.e-7)
    if noise_curve_mode == 1:
        assert_allclose(np.sum(signal_time_white**2, axis=0), 2 * wc.dt * np.sum(signal_time**2, axis=0), atol=1.e-10, rtol=1.e-10)
        assert_allclose(np.sum(waveform_dense_white**2, axis=(0, 1)), 2 * wc.dt * np.sum(signal_time**2, axis=0), atol=1.e-10, rtol=1.e-7)
        assert_allclose(np.sum(waveform_dense_white**2, axis=(0, 1)), 2 * wc.dt * np.sum(waveform_dense**2, axis=(0, 1)), atol=1.e-10, rtol=1.e-7)

    if noise_curve_mode == 1:
        assert_allclose(phase_got, intrinsic.phi0, atol=1.e-10, rtol=1.e-3)

    assert_allclose(amp_got, template_amp_direct, atol=1.e-10, rtol=1.e-3)
    assert_allclose(amp_got, intrinsic.amp0_t, atol=1.e-10, rtol=1.e-3)
    assert_allclose(template_amp_direct, intrinsic.amp0_t, atol=1.e-10, rtol=1.e-3)


# scaling on (Nf, Nt, dt, mult) in the second configuration
@pytest.mark.parametrize(
    'channel_mult',
    [
        (1.0, 1.0, 1.0, 1.0),
        (2.0, 0.5, 1.0, 1.0),
    ],
)
@pytest.mark.parametrize('amp_mult', [1.0, 10.0])
@pytest.mark.parametrize('noise_curve_mode', [0, 1])
def test_noise_generation_scaling_time(channel_mult: tuple[float, float, float, float], amp_mult: float, noise_curve_mode: int) -> None:
    """Test recover same snr distribution in wavelet, time, matched filter, and frequency domains."""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    if noise_curve_mode == 0:
        response_mode = 2
        amp0_use = 1.0e-20 * amp_mult
    else:
        response_mode = 2
        amp0_use = 1.0 * amp_mult

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    config_in['lisa_constants']['noise_curve_mode'] = noise_curve_mode
    config_in['wavelet_constants']['Nst'] = 512

    # replace the Nf and Nt from the file
    config_in['wavelet_constants']['Nf'] = int(config_in['wavelet_constants']['Nf'] * channel_mult[0])
    config_in['wavelet_constants']['Nt'] = int(config_in['wavelet_constants']['Nt'] * channel_mult[1])
    config_in['wavelet_constants']['dt'] = float(config_in['wavelet_constants']['dt'] * channel_mult[2])
    config_in['wavelet_constants']['mult'] = int(config_in['wavelet_constants']['mult'] * channel_mult[3])

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    noise = instrument_noise_AET_wdm_m(lc, wc)

    fs = np.arange(0, wc.Nf) * wc.DF

    noise_exp = np.zeros((fs.size, noise.shape[-1]))

    noise_exp[1:] = instrument_noise_AET(fs[1:], lc)  # /(2 * wc.dt)

    fs_full = np.fft.rfftfreq(wc.Nf * wc.Nt, d=wc.dt)

    noise_exp_full = np.zeros((fs_full.size, noise.shape[-1]))

    noise_exp_full[1:] = instrument_noise_AET(fs_full[1:], lc)  # /(2 * wc.dt)

    inv_chol_noise_exp_full = np.zeros((fs_full.size, noise.shape[-1]))
    inv_chol_noise_exp_full[1:] = np.sqrt(1. / noise_exp_full[1:])

    seed = 31415
    nc_snr = int(noise.shape[1])

    noise_manager = DiagonalStationaryDenseNoiseModel(noise, wc, prune=1, nc_snr=nc_snr, seed=seed)

    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=amp0_use,  # amplitude
        phi0=0.3,  # phase at t=0
        F0=1.0e-4,  # initial frequency (Hz)
        FTd0=0.,  # frequency derivative (Hz/s)
    )

    assert intrinsic.FTd0 < 8 * wc.DF / wc.Tw
    assert intrinsic.FTd0 < wc.dfd * (wc.Nfd - wc.Nfd_negative)
    assert intrinsic.FTd0 >= wc.dfd * (-wc.Nfd_negative)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)

    nt_lim_snr = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)

    waveform = LinearFrequencyWaveletWaveformTime(
        params,
        wc,
        lc,
        nt_lim_waveform,
        table_cache_mode='check',
        table_output_mode='skip',
        response_mode=response_mode,
    )

    wavelet_waveform = waveform.get_unsorted_coeffs()

    waveform_dense = wavelet_sparse_to_dense(wavelet_waveform, wc)

    waveform_dense_white = waveform_dense * noise_manager.get_inv_chol_S()

    waveform_sparse_white = wavelet_dense_select_sparse(waveform_dense_white, wavelet_waveform, wc, inplace_mode=0)

    signal_time_white = np.zeros((wc.Nf * wc.Nt, nc_snr))

    for itrc in range(nc_snr):
        signal_time_white[:, itrc] = inverse_wavelet_time(waveform_dense_white[:, :, itrc], wc.Nf, wc.Nt)

    signal_time = np.zeros((wc.Nf * wc.Nt, nc_snr))

    for itrc in range(nc_snr):
        signal_time[:, itrc] = inverse_wavelet_time(waveform_dense[:, :, itrc], wc.Nf, wc.Nt)

    signal_freq = np.zeros(((wc.Nf * wc.Nt) // 2 + 1, nc_snr), dtype=np.complex128)

    for itrc in range(nc_snr):
        signal_freq[:, itrc] = inverse_wavelet_freq(waveform_dense[:, :, itrc], wc.Nf, wc.Nt) / np.sqrt(wc.Nt * wc.Nf / 2)

    signal_freq_white = np.zeros(((wc.Nf * wc.Nt) // 2 + 1, nc_snr), dtype=np.complex128)

    for itrc in range(nc_snr):
        signal_freq_white[:, itrc] = inverse_wavelet_freq(waveform_dense_white[:, :, itrc], wc.Nf, wc.Nt) / np.sqrt(wc.Nt * wc.Nf / 2)

    # get realizations of the noise and mask the non-overlapping part so we can isolate the band-limited noise
    seed_gen = 31415926
    n_seed = 100
    rng = np.random.default_rng(seed_gen)
    seeds = rng.integers(0, np.iinfo(np.int_).max, n_seed)

    signal_amp_cos_unwhitened = np.zeros((n_seed, nc_snr))
    signal_amp_sin_unwhitened = np.zeros((n_seed, nc_snr))

    noise_amp_cos_unwhitened = np.zeros((n_seed, nc_snr))
    noise_amp_sin_unwhitened = np.zeros((n_seed, nc_snr))

    signal_amp_cos_unwhitened[:] = np.trapezoid((signal_time.T * np.cos(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs
    signal_amp_sin_unwhitened[:] = np.trapezoid((signal_time.T * np.sin(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs
    signal_amp_mag_unwhitened = (signal_amp_cos_unwhitened**2 + signal_amp_sin_unwhitened**2) * wc.Nt * wc.Nf * 2

    noise_exp_point = instrument_noise_AET(np.array([intrinsic.F0]), lc)[0]  # /(2 * wc.dt)
    signal_signal_matched_expect = np.sum(signal_amp_mag_unwhitened / noise_exp_point, axis=-1) * 2 * wc.dt

    signal_noise = np.zeros(n_seed)
    signal_noise_time = np.zeros(n_seed)
    signal_noise_freq = np.zeros(n_seed)

    noise_noise = np.zeros(n_seed)
    noise_noise_time = np.zeros(n_seed)
    noise_noise_freq = np.zeros(n_seed)

    noise_amp_cos = np.zeros((n_seed, nc_snr))
    noise_amp_sin = np.zeros((n_seed, nc_snr))

    signal_amp_cos = np.zeros((n_seed, nc_snr))
    signal_amp_sin = np.zeros((n_seed, nc_snr))

    signal_amp_cos[:] = np.trapezoid((signal_time_white.T * np.cos(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs
    signal_amp_sin[:] = np.trapezoid((signal_time_white.T * np.sin(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs
    signal_amp_mag = (signal_amp_cos**2 + signal_amp_sin**2) * wc.Nt * wc.Nf * 2

    signal_signal = np.full(n_seed, np.sum(waveform_sparse_white.wave_value * waveform_sparse_white.wave_value))
    signal_signal_time = np.full(n_seed, np.sum(signal_time_white * signal_time_white))
    signal_signal_freq = np.full(n_seed, np.sum(np.abs(signal_freq_white)**2))
    signal_signal_matched = np.sum(signal_amp_mag, axis=-1)

    assert_allclose(signal_signal, np.sum(waveform_dense_white * waveform_dense_white), atol=1.0e-100, rtol=1.0e-13)
    assert_allclose(signal_signal_matched, signal_signal_matched_expect, atol=1.e-100, rtol=6.e-2)

    # noise_real_white = noise_manager.generate_dense_noise(seed_override=seed_loc, white_mode=1)
    noise_real_white = noise_manager.generate_dense_noise(white_mode=1)

    noise_time_white = np.zeros((wc.Nf * wc.Nt, nc_snr))
    noise_time = np.zeros((wc.Nf * wc.Nt, nc_snr))
    noise_freq_white = np.zeros((wc.Nf * wc.Nt // 2 + 1, nc_snr), dtype=np.complex128)

    for itrs, seed_loc in enumerate(seeds):
        noise_real = noise_manager.generate_dense_noise(seed_override=seed_loc, white_mode=0)
        noise_real_white = noise_real * noise_manager.get_inv_chol_S()
        noise_sparse_white = wavelet_dense_select_sparse(noise_real_white, wavelet_waveform, wc, inplace_mode=0)

        for itrc in range(nc_snr):
            noise_time_white[:, itrc] = inverse_wavelet_time(noise_real_white[:, :, itrc], wc.Nf, wc.Nt)

        for itrc in range(nc_snr):
            noise_time[:, itrc] = inverse_wavelet_time(noise_real[:, :, itrc], wc.Nf, wc.Nt)

        for itrc in range(nc_snr):
            noise_freq_white[:, itrc] = inverse_wavelet_freq(noise_real_white[:, :, itrc], wc.Nf, wc.Nt) / np.sqrt(wc.Nt * wc.Nf / 2)

        noise_amp_cos[itrs] = np.trapezoid((noise_time_white.T * np.cos(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs
        noise_amp_sin[itrs] = np.trapezoid((noise_time_white.T * np.sin(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs

        noise_amp_cos_unwhitened[itrs] = np.trapezoid((noise_time.T * np.cos(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs
        noise_amp_sin_unwhitened[itrs] = np.trapezoid((noise_time.T * np.sin(2 * np.pi * intrinsic.F0 * wc.dt * np.arange(0, wc.Nt * wc.Nf))), dx=wc.dt, axis=1) / wc.Tobs

        noise_noise_time[itrs] = np.sum(noise_time_white * noise_time_white)
        signal_noise_time[itrs] = np.sum(signal_time_white * noise_time_white)

        noise_noise_freq[itrs] = np.sum(np.abs(noise_freq_white)**2)
        signal_noise_freq[itrs] = np.sum(np.real(signal_freq_white * np.conjugate(noise_freq_white)))

        noise_noise[itrs] = np.sum(noise_sparse_white.wave_value * noise_sparse_white.wave_value)
        signal_noise[itrs] = np.sum(noise_sparse_white.wave_value * waveform_sparse_white.wave_value)

    noise_amp_mag = (noise_amp_cos**2 + noise_amp_sin**2) * wc.Nt * wc.Nf * 2
    noise_noise_matched = np.sum(noise_amp_mag, axis=-1)

    signal_noise_amp_mag = (signal_amp_cos * noise_amp_cos + signal_amp_sin * noise_amp_sin) * wc.Nt * wc.Nf * 2
    signal_noise_matched = np.sum(signal_noise_amp_mag, axis=-1)

    noise_amp_mag_unwhitened = (noise_amp_cos_unwhitened**2 + noise_amp_sin_unwhitened**2) * wc.Nt * wc.Nf * 2
    noise_noise_matched_expect = np.sum(noise_amp_mag_unwhitened / noise_exp_point, axis=-1) * 2 * wc.dt

    assert_allclose(noise_noise_matched, noise_noise_matched_expect, atol=1.e-100, rtol=2.e-1)

    signal_noise_amp_mag = (signal_amp_cos * noise_amp_cos + signal_amp_sin * noise_amp_sin) * wc.Nt * wc.Nf * 2
    signal_noise_matched = np.sum(signal_noise_amp_mag, axis=-1)

    data_noise = signal_noise + noise_noise
    data_noise_time = signal_noise_time + noise_noise_time
    data_noise_freq = signal_noise_freq + noise_noise_freq
    data_noise_matched = signal_noise_matched + noise_noise_matched

    data_signal = signal_signal + signal_noise
    data_signal_time = signal_signal_time + signal_noise_time
    data_signal_freq = signal_signal_freq + signal_noise_freq
    data_signal_matched = signal_signal_matched + signal_noise_matched

    data_data = signal_signal + 2 * signal_noise + noise_noise
    data_data_time = signal_signal_time + 2 * signal_noise_time + noise_noise_time
    data_data_freq = signal_signal_freq + 2 * signal_noise_freq + noise_noise_freq
    data_data_matched = signal_signal_matched + 2 * signal_noise_matched + noise_noise_matched

    log_likelihood = data_signal - signal_signal / 2
    log_likelihood_time = data_signal_time - signal_signal_time / 2
    log_likelihood_freq = data_signal_freq - signal_signal_freq / 2
    log_likelihood_matched = data_signal_matched - signal_signal_matched / 2

    snr_channel = noise_manager.get_sparse_snrs(wavelet_waveform, nt_lim_snr)
    snr_tot: float = float(np.linalg.norm(snr_channel))

    n_set_tot: int = int(np.sum(wavelet_waveform.n_set))
    std_data_data: float = float(np.sqrt(4 * snr_tot ** 2 + 2 * n_set_tot))

    n_set_tot_time: int = int((wc.Nf - 1) * wc.Nt * nc_snr)
    std_data_data_time: float = float(np.sqrt(4 * snr_tot ** 2 + 2 * n_set_tot_time))

    n_set_tot_freq: int = int((wc.Nf * wc.Nt - wc.Nt) * nc_snr)
    std_data_data_freq: float = float(np.sqrt(4 * snr_tot ** 2 + 2 * n_set_tot_freq))

    n_set_tot_matched: int = 6
    std_data_data_matched: float = float(np.sqrt(4 * snr_tot ** 2 + 2 * n_set_tot_matched))

    assert_allclose(signal_signal, signal_signal_time, atol=1.e-100, rtol=3.e-7)
    assert_allclose(signal_signal, signal_signal_freq, atol=1.e-100, rtol=3.e-7)
    assert_allclose(signal_signal, signal_signal_matched, atol=1.e-100, rtol=9.e-3)
    assert_allclose(signal_signal_freq, signal_signal_time, atol=1.e-100, rtol=3.e-7)
    assert_allclose(signal_signal_matched, signal_signal_time, atol=1.e-100, rtol=9.e-3)
    assert_allclose(signal_signal_freq, signal_signal_matched, atol=1.e-100, rtol=9.e-3)

    assert_allclose(log_likelihood, log_likelihood_matched, atol=1.e-100, rtol=9.e-3)
    assert_allclose(log_likelihood, log_likelihood_freq, atol=1.e-100, rtol=1.e-5)
    assert_allclose(log_likelihood, log_likelihood_time, atol=1.e-100, rtol=1.e-5)
    assert_allclose(log_likelihood_time, log_likelihood_freq, atol=1.e-100, rtol=1.e-5)
    assert_allclose(log_likelihood_time, log_likelihood_matched, atol=1.e-100, rtol=9.e-3)
    assert_allclose(log_likelihood_freq, log_likelihood_matched, atol=1.e-100, rtol=9.e-3)

    _ = unit_normal_battery(noise_noise_matched - n_set_tot_matched, mult=float(np.sqrt(2 * n_set_tot_matched)), do_assert=True)
    # _ = unit_normal_battery(log_likelihood_matched - signal_signal_matched / 2, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(signal_noise_matched, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(data_signal_matched - signal_signal_matched, mult=snr_tot, do_assert=True)

    _ = unit_normal_battery(data_noise_matched - n_set_tot_matched, mult=float(np.sqrt(2 * n_set_tot_matched + snr_tot**2)), do_assert=True)
    _ = unit_normal_battery(data_data_matched - n_set_tot_matched - signal_signal_matched, mult=std_data_data_matched, do_assert=True)

    _ = unit_normal_battery(noise_noise_freq - n_set_tot_freq, mult=float(np.sqrt(2 * n_set_tot_freq)), do_assert=True)
    _ = unit_normal_battery(log_likelihood_freq - signal_signal_freq / 2, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(signal_noise_freq, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(data_signal_freq - signal_signal_freq, mult=snr_tot, do_assert=True)

    _ = unit_normal_battery(data_noise_freq - n_set_tot_freq, mult=float(np.sqrt(2 * n_set_tot_freq + snr_tot**2)), do_assert=True)
    _ = unit_normal_battery(data_data_freq - n_set_tot_freq - signal_signal_freq, mult=std_data_data_freq, do_assert=True)

    _ = unit_normal_battery(signal_noise_time, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(data_signal_time - signal_signal_time, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(log_likelihood_time - signal_signal_time / 2, mult=snr_tot, do_assert=True)

    _ = unit_normal_battery(noise_noise_time - n_set_tot_time, mult=float(np.sqrt(2 * n_set_tot_time)), do_assert=True)
    _ = unit_normal_battery(data_noise_time - n_set_tot_time, mult=float(np.sqrt(2 * n_set_tot_time + snr_tot**2)), do_assert=True)
    _ = unit_normal_battery(data_data_time - n_set_tot_time - signal_signal_time, mult=std_data_data_time, do_assert=True)

    _ = unit_normal_battery(signal_noise, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(data_signal - signal_signal, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(noise_noise - n_set_tot, mult=float(np.sqrt(2 * n_set_tot)), do_assert=True)
    _ = unit_normal_battery(data_noise - n_set_tot, mult=float(np.sqrt(2 * n_set_tot + snr_tot**2)), do_assert=True)
    _ = unit_normal_battery(data_data - n_set_tot - signal_signal, mult=std_data_data, do_assert=True)
    _ = unit_normal_battery(log_likelihood - signal_signal / 2, mult=snr_tot, do_assert=True)


# scaling on (Nf, Nt, dt, mult) in the second configuration
@pytest.mark.parametrize(
    'channel_mult',
    [
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 4.0, 1.0, 1.0),
        (1.0, 0.25, 1.0, 1.0),
        # (0.5, 2.0, 1.0, 1.0),
        # (0.5, 1.0, 2.0, 1.0),
        # (1.0, 0.5, 2.0, 1.0),
        # (1.0, 2.0, 0.5, 1.0),
        # (2.0, 0.5, 1.0, 1.0),
        # (1.0, 1.0, 0.5, 1.0),
    ],
)
@pytest.mark.parametrize('amp_mult', [1000000.0, 0.0000001, 1.0, 0.1, 10.0])
def test_noise_generation_scaling_direct(channel_mult: tuple[float, float, float, float], amp_mult: float) -> None:
    """Test the scaling between (Nf, Nt, dt, mult) and SNR^2"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    noise_curve_mode = 0
    if noise_curve_mode == 0:
        response_mode = 0
        amp0_use = 1.0e-20 * amp_mult
    else:
        response_mode = 2
        amp0_use = 1.0

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    config_in['lisa_constants']['noise_curve_mode'] = noise_curve_mode
    config_in['wavelet_constants']['Nst'] = 512

    # replace the Nf and Nt from the file
    config_in['wavelet_constants']['Nf'] = int(config_in['wavelet_constants']['Nf'] * channel_mult[0])
    config_in['wavelet_constants']['Nt'] = int(config_in['wavelet_constants']['Nt'] * channel_mult[1])
    config_in['wavelet_constants']['dt'] = float(config_in['wavelet_constants']['dt'] * channel_mult[2])
    config_in['wavelet_constants']['mult'] = int(config_in['wavelet_constants']['mult'] * channel_mult[3])

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    noise = instrument_noise_AET_wdm_m(lc, wc)

    fs = np.arange(0, wc.Nf) * wc.DF

    noise_exp = np.zeros((fs.size, noise.shape[-1]))

    noise_exp[1:] = instrument_noise_AET(fs[1:], lc)  # /(2 * wc.dt)

    seed = 31415
    nc_snr = int(noise.shape[1])

    noise_manager = DiagonalStationaryDenseNoiseModel(noise, wc, prune=1, nc_snr=nc_snr, seed=seed)

    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=amp0_use,  # amplitude
        phi0=0.3,  # phase at t=0
        F0=1.0e-4,  # initial frequency (Hz)
        FTd0=3.0e-12,  # frequency derivative (Hz/s)
    )

    assert intrinsic.FTd0 < 8 * wc.DF / wc.Tw
    assert intrinsic.FTd0 < wc.dfd * (wc.Nfd - wc.Nfd_negative)
    assert intrinsic.FTd0 >= wc.dfd * (-wc.Nfd_negative)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)

    nt_lim_snr = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)

    waveform = LinearFrequencyWaveletWaveformTime(
        params,
        wc,
        lc,
        nt_lim_waveform,
        table_cache_mode='check',
        table_output_mode='skip',
        response_mode=response_mode,
    )

    wavelet_waveform = waveform.get_unsorted_coeffs()

    waveform_dense = wavelet_sparse_to_dense(wavelet_waveform, wc)

    waveform_dense_white = waveform_dense * noise_manager.get_inv_chol_S()

    waveform_sparse_white = wavelet_dense_select_sparse(waveform_dense_white, wavelet_waveform, wc, inplace_mode=0)

    # get realizations of the noise and mask the non-overlapping part so we can isolate the band-limited noise
    seed_gen = 31415926
    n_seed = 1000
    rng = np.random.default_rng(seed_gen)
    seeds = rng.integers(0, np.iinfo(np.int_).max, n_seed)

    signal_noise = np.zeros(n_seed)

    noise_noise = np.zeros(n_seed)

    signal_signal = np.full(n_seed, np.sum(waveform_sparse_white.wave_value * waveform_sparse_white.wave_value))

    assert_allclose(signal_signal, np.sum(waveform_dense_white * waveform_dense_white), atol=1.0e-100, rtol=1.0e-13)

    # noise_real_white = noise_manager.generate_dense_noise(seed_override=seed_loc, white_mode=1)
    noise_real_white = noise_manager.generate_dense_noise(white_mode=1)

    noise_sparse_white = wavelet_dense_select_sparse(noise_real_white, wavelet_waveform, wc, inplace_mode=0)

    # for itrs, seed_loc in enumerate(seeds):
    for itrs in range(seeds.size):
        # noise_real_white = noise_manager.generate_dense_noise(seed_override=seed_loc, white_mode=1)
        # overwrite sparsely for speed
        for itrc in range(noise.shape[1]):
            n_set_loc: int = int(noise_sparse_white.n_set[itrc])
            noise_sparse_white.wave_value[itrc, :n_set_loc] = rng.normal(0.0, 1.0, n_set_loc)

        signal_noise[itrs] = np.sum(noise_sparse_white.wave_value * waveform_sparse_white.wave_value)

        noise_noise[itrs] = np.sum(noise_sparse_white.wave_value * noise_sparse_white.wave_value)

    data_noise = signal_noise + noise_noise

    data_signal = signal_signal + signal_noise

    data_data = signal_signal + 2 * signal_noise + noise_noise

    log_likelihood = data_signal - signal_signal / 2

    log_likelihood_mean_exp = np.mean(signal_signal / 2.0)
    # assert_allclose(log_likelihood_mean_exp1, log_likelihood_mean_exp, atol=1.e-100, rtol=1.e-10)

    snr_channel = noise_manager.get_sparse_snrs(wavelet_waveform, nt_lim_snr)
    snr_tot: float = float(np.linalg.norm(snr_channel))

    n_set_tot: int = int(np.sum(wavelet_waveform.n_set))
    std_data_data: float = float(np.sqrt(4 * snr_tot ** 2 + 2 * n_set_tot))

    print('likelihood mean, std, min, max', log_likelihood.mean(), log_likelihood.std(), log_likelihood.min(), log_likelihood.max())
    print('likelihood expect', log_likelihood_mean_exp)
    print('signal*noise mean, std, min, max', signal_noise.mean(), signal_noise.std(), signal_noise.min(), signal_noise.max())
    print('noise*noise mean, std, min, max', noise_noise.mean(), noise_noise.std(), noise_noise.min(), noise_noise.max())
    print('data*signal mean, std, min, max', data_signal.mean(), data_signal.std(), data_signal.min(), data_signal.max())
    print('data*noise mean, std, min, max', data_noise.mean(), data_noise.std(), data_noise.min(), data_noise.max())
    print('data*data mean, std, min, max', data_data.mean(), data_data.std(), data_data.min(), data_data.max())
    print('data*data mean diff, std', (data_data.mean() - n_set_tot - signal_signal.mean()) / np.sqrt(2 * n_set_tot + 2 * snr_tot**2), data_data.std() / np.sqrt(2 * n_set_tot + 2 * snr_tot**2))
    print('data*data exp, std', np.sqrt(2 * n_set_tot + 2 * snr_tot**2) / np.sqrt(n_seed), np.sqrt(2.0) / np.sqrt(n_seed) / np.sqrt(2 * n_set_tot + 2 * snr_tot**2))
    print('snr channels, total', snr_channel, snr_tot)
    print('signal*signal', signal_signal.mean())
    print('set, total', wavelet_waveform.n_set, n_set_tot)
    print('(signal* noise std)/(total snr)', signal_noise.std() / snr_tot)

    assert_allclose(snr_tot**2, 2 * log_likelihood_mean_exp, atol=1.0e-10, rtol=1.0e-10)

    _ = unit_normal_battery(signal_noise, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(data_signal - signal_signal, mult=snr_tot, do_assert=True)
    _ = unit_normal_battery(noise_noise - n_set_tot, mult=float(np.sqrt(2 * n_set_tot)), do_assert=True)
    _ = unit_normal_battery(data_noise - n_set_tot, mult=float(np.sqrt(2 * n_set_tot + snr_tot**2)), do_assert=True)
    _ = unit_normal_battery(data_data - n_set_tot - signal_signal, mult=std_data_data, do_assert=True)
    _ = unit_normal_battery(log_likelihood - signal_signal / 2, mult=snr_tot, do_assert=True)


# scaling on (Nf, Nt, dt, mult) in the second configuration
@pytest.mark.parametrize(
    'channel_mult',
    [
        # (1.0, 1.0, 1.0, 1.0),
        (0.5, 2.0, 1.0, 1.0),
        (1.0, 0.5, 1.0, 1.0),
        (1.0, 4.0, 1.0, 1.0),
        # (0.5, 1.0, 2.0, 1.0),
        # (1.0, 0.5, 2.0, 1.0),
        # (1.0, 2.0, 0.5, 1.0),
        # (2.0, 0.5, 1.0, 1.0),
        # (1.0, 1.0, 0.5, 1.0),
    ],
)
@pytest.mark.parametrize('amp_mult', [1.0])
def test_noise_generation_scaling_compare(channel_mult: tuple[float, float, float, float], amp_mult: float) -> None:
    """Test the scaling between (Nf, Nt, dt, mult) and SNR^2"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

    noise_curve_mode = 0
    if noise_curve_mode == 0:
        response_mode = 0
        amp0_use = 1.0e-20 * amp_mult
    else:
        response_mode = 2
        amp0_use = 1.0

    config_in1['lisa_constants']['noise_curve_mode'] = noise_curve_mode
    config_in1['wavelet_constants']['Nst'] = 512

    wc1 = get_wavelet_model(config_in1)
    lc1 = get_lisa_constants(config_in1)

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    config_in2['lisa_constants']['noise_curve_mode'] = noise_curve_mode
    config_in2['wavelet_constants']['Nst'] = 512

    # replace the Nf and Nt from the file
    config_in2['wavelet_constants']['Nf'] = int(wc1.Nf * channel_mult[0])
    config_in2['wavelet_constants']['Nt'] = int(wc1.Nt * channel_mult[1])
    config_in2['wavelet_constants']['dt'] = float(wc1.dt * channel_mult[2])
    config_in2['wavelet_constants']['mult'] = int(wc1.mult * channel_mult[3])

    wc2 = get_wavelet_model(config_in2)
    lc2 = get_lisa_constants(config_in2)

    # check proper set up
    assert lc1.noise_curve_mode == lc2.noise_curve_mode

    noise1 = instrument_noise_AET_wdm_m(lc1, wc1)
    noise2 = instrument_noise_AET_wdm_m(lc2, wc2)

    seed1 = 31415
    seed2 = 31415

    noise_manager1 = DiagonalStationaryDenseNoiseModel(noise1, wc1, prune=1, nc_snr=noise1.shape[1], seed=seed1)
    noise_manager2 = DiagonalStationaryDenseNoiseModel(noise2, wc2, prune=1, nc_snr=noise2.shape[1], seed=seed2)

    # scale the spectra and check the integrated power matches expectation
    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=amp0_use,  # amplitude
        phi0=0.3,  # phase at t=0
        F0=1.0e-4,  # initial frequency (Hz)
        FTd0=3.0e-12,  # frequency derivative (Hz/s)
    )

    assert intrinsic.FTd0 < 8 * wc1.DF / wc1.Tw
    assert intrinsic.FTd0 < 8 * wc2.DF / wc2.Tw
    assert intrinsic.FTd0 < wc1.dfd * (wc1.Nfd - wc1.Nfd_negative)
    assert intrinsic.FTd0 < wc2.dfd * (wc2.Nfd - wc2.Nfd_negative)
    assert intrinsic.FTd0 >= wc1.dfd * (-wc1.Nfd_negative)
    assert intrinsic.FTd0 >= wc2.dfd * (-wc2.Nfd_negative)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    nt_lim_waveform1 = PixelGenericRange(0, wc1.Nt, wc1.DT, 0.0)
    nt_lim_waveform2 = PixelGenericRange(0, wc2.Nt, wc2.DT, 0.0)

    nt_lim_snr1 = PixelGenericRange(0, wc1.Nt, wc1.DT, 0.0)
    nt_lim_snr2 = PixelGenericRange(0, wc2.Nt, wc2.DT, 0.0)

    waveform1 = LinearFrequencyWaveletWaveformTime(
        params,
        wc1,
        lc1,
        nt_lim_waveform1,
        table_cache_mode='check',
        table_output_mode='skip',
        response_mode=response_mode,
    )
    waveform2 = LinearFrequencyWaveletWaveformTime(
        params,
        wc2,
        lc2,
        nt_lim_waveform2,
        table_cache_mode='check',
        table_output_mode='skip',
        response_mode=response_mode,
    )

    wavelet_waveform1 = waveform1.get_unsorted_coeffs()
    wavelet_waveform2 = waveform2.get_unsorted_coeffs()

    waveform_dense1 = wavelet_sparse_to_dense(wavelet_waveform1, wc1)
    waveform_dense2 = wavelet_sparse_to_dense(wavelet_waveform2, wc2)

    waveform_dense1_white = waveform_dense1 * noise_manager1.get_inv_chol_S()
    waveform_dense2_white = waveform_dense2 * noise_manager2.get_inv_chol_S()

    waveform_sparse1_white = wavelet_dense_select_sparse(waveform_dense1_white, wavelet_waveform1, wc1, inplace_mode=0)
    waveform_sparse2_white = wavelet_dense_select_sparse(waveform_dense2_white, wavelet_waveform2, wc2, inplace_mode=0)

    waveform_sparse1_white_alt = whiten_sparse_data(wavelet_waveform1, noise_manager1.get_inv_chol_S(), wc1)
    waveform_sparse2_white_alt = whiten_sparse_data(wavelet_waveform2, noise_manager2.get_inv_chol_S(), wc2)

    assert_allclose(
        waveform_sparse1_white_alt.wave_value, waveform_sparse1_white.wave_value, atol=1.0e-100, rtol=1.0e-14
    )
    assert_allclose(
        waveform_sparse2_white_alt.wave_value, waveform_sparse2_white.wave_value, atol=1.0e-100, rtol=1.0e-14
    )

    assert_array_equal(waveform_sparse1_white_alt.pixel_index, waveform_sparse1_white.pixel_index)
    assert_array_equal(waveform_sparse1_white_alt.n_set, waveform_sparse1_white.n_set)
    assert_array_equal(waveform_sparse1_white_alt.n_pixel_max, waveform_sparse1_white.n_pixel_max)

    assert_array_equal(waveform_sparse2_white_alt.pixel_index, waveform_sparse2_white.pixel_index)
    assert_array_equal(waveform_sparse2_white_alt.n_set, waveform_sparse2_white.n_set)
    assert_array_equal(waveform_sparse2_white_alt.n_pixel_max, waveform_sparse2_white.n_pixel_max)

    # get realizations of the noise and mask the non-overlapping part so we can isolate the band-limited noise
    seed_gen = 31415926
    n_seed = 1000
    rng = np.random.default_rng(seed_gen)
    seeds = rng.integers(0, np.iinfo(np.int_).max, n_seed)

    signal_noise1 = np.zeros(n_seed)
    signal_noise2 = np.zeros(n_seed)

    noise_noise1 = np.zeros(n_seed)
    noise_noise2 = np.zeros(n_seed)

    signal_signal1 = np.full(n_seed, np.sum(waveform_sparse1_white.wave_value * waveform_sparse1_white.wave_value))
    signal_signal2 = np.full(n_seed, np.sum(waveform_sparse2_white.wave_value * waveform_sparse2_white.wave_value))

    assert_allclose(signal_signal1, np.sum(waveform_dense1_white * waveform_dense1_white), atol=1.0e-100, rtol=1.0e-13)
    assert_allclose(signal_signal2, np.sum(waveform_dense2_white * waveform_dense2_white), atol=1.0e-100, rtol=1.0e-13)

    for itrs, seed_loc in enumerate(seeds):
        noise_real1_white = noise_manager1.generate_dense_noise(seed_override=seed_loc, white_mode=1)
        noise_real2_white = noise_manager2.generate_dense_noise(seed_override=seed_loc, white_mode=1)

        noise_sparse1_white = wavelet_dense_select_sparse(noise_real1_white, wavelet_waveform1, wc1, inplace_mode=0)
        noise_sparse2_white = wavelet_dense_select_sparse(noise_real2_white, wavelet_waveform2, wc2, inplace_mode=0)

        signal_noise1[itrs] = np.sum(noise_sparse1_white.wave_value * waveform_sparse1_white.wave_value)
        signal_noise2[itrs] = np.sum(noise_sparse2_white.wave_value * waveform_sparse2_white.wave_value)

        noise_noise1[itrs] = np.sum(noise_sparse1_white.wave_value * noise_sparse1_white.wave_value)
        noise_noise2[itrs] = np.sum(noise_sparse2_white.wave_value * noise_sparse2_white.wave_value)

    data_noise1 = signal_noise1 + noise_noise1
    data_noise2 = signal_noise2 + noise_noise2

    data_signal1 = signal_signal1 + signal_noise1
    data_signal2 = signal_signal2 + signal_noise2

    data_data1 = signal_signal1 + 2 * signal_noise1 + noise_noise1
    data_data2 = signal_signal2 + 2 * signal_noise2 + noise_noise2

    log_likelihood1 = data_signal1 - signal_signal1 / 2
    log_likelihood2 = data_signal2 - signal_signal2 / 2

    log_likelihood_mean_exp1 = np.mean(signal_signal1 / 2.0)
    log_likelihood_mean_exp2 = np.mean(signal_signal2 / 2.0)
    # assert_allclose(log_likelihood_mean_exp1, log_likelihood_mean_exp2, atol=1.e-100, rtol=1.e-10)

    snr_channel1 = noise_manager1.get_sparse_snrs(wavelet_waveform1, nt_lim_snr1)
    snr_tot1 = float(np.linalg.norm(snr_channel1))

    n_set_tot1 = np.sum(wavelet_waveform1.n_set)

    snr_channel2 = noise_manager2.get_sparse_snrs(wavelet_waveform2, nt_lim_snr2)
    snr_tot2 = float(np.linalg.norm(snr_channel2))

    n_set_tot2 = np.sum(wavelet_waveform2.n_set)

    assert_allclose(snr_tot1**2, 2 * log_likelihood_mean_exp1, atol=1.0e-10, rtol=1.0e-10)
    assert_allclose(snr_tot2**2, 2 * log_likelihood_mean_exp2, atol=1.0e-10, rtol=1.0e-10)

    _ = unit_normal_battery(signal_noise1, mult=snr_tot1, do_assert=True)
    _ = unit_normal_battery(signal_noise2, mult=snr_tot2, do_assert=True)
    _ = unit_normal_battery(data_signal1 - signal_signal1, mult=snr_tot1, do_assert=True)
    _ = unit_normal_battery(data_signal2 - signal_signal2, mult=snr_tot2, do_assert=True)
    _ = unit_normal_battery(noise_noise1 - n_set_tot1, mult=float(np.sqrt(2 * n_set_tot1)), do_assert=True)
    _ = unit_normal_battery(noise_noise2 - n_set_tot2, mult=float(np.sqrt(2 * n_set_tot2)), do_assert=True)
    _ = unit_normal_battery(data_noise1 - n_set_tot1, mult=float(np.sqrt(2 * n_set_tot1 + snr_tot1**2)), do_assert=True)
    _ = unit_normal_battery(data_noise2 - n_set_tot2, mult=float(np.sqrt(2 * n_set_tot2 + snr_tot2**2)), do_assert=True)
    _ = unit_normal_battery(
        data_data1 - n_set_tot1 - signal_signal1, mult=float(np.sqrt(2 * n_set_tot1 + 4 * snr_tot1**2)), do_assert=True
    )
    _ = unit_normal_battery(
        data_data2 - n_set_tot2 - signal_signal2, mult=float(np.sqrt(2 * n_set_tot2 + 4 * snr_tot2**2)), do_assert=True
    )
    _ = unit_normal_battery(log_likelihood1 - signal_signal1 / 2, mult=snr_tot1, do_assert=True)
    _ = unit_normal_battery(log_likelihood2 - signal_signal2 / 2, mult=snr_tot2, do_assert=True)

    # check if means of likelihoods match each other
    assert_allclose(
        log_likelihood1.mean() - signal_signal1 / 2,
        log_likelihood2.mean() - signal_signal2 / 2,
        atol=float(5.0 * np.sqrt(snr_tot1**2 + snr_tot2**2)),
        rtol=1.0e-10,
    )

    # check if stds of likelihoods match each other
    rat1 = signal_noise1.std() / snr_tot1
    rat2 = signal_noise2.std() / snr_tot2
    assert_allclose(
        rat1 - 1.0, rat2 - 1.0, atol=float(5.0 * np.sqrt(2 / n_seed) * np.sqrt(rat1**2 + rat2**2)), rtol=1.0e-10
    )

    # this would also be true if the channels are uncorrelated
    # unit_normal_battery(log_likelihood1 - signal_signal1/2 - log_likelihood2 + signal_signal2/2, mult=np.sqrt(snr_tot1**2 + snr_tot2**2), do_assert=True)


# scaling on (Nf, Nt, dt, mult) in the second configuration
@pytest.mark.parametrize(
    'channel_mult',
    [
        (0.5, 2.0, 1.0, 1.0),
    ],
)
def test_noise_whiten_consistency(channel_mult: tuple[float, float, float, float]) -> None:
    """Test the scaling between (Nf, Nt, dt, mult) and SNR^2"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

    noise_curve_mode = 0
    if noise_curve_mode == 0:
        response_mode = 0
        amp0_use = 1.0e-20
    else:
        response_mode = 2
        amp0_use = 1.0

    config_in1['lisa_constants']['noise_curve_mode'] = noise_curve_mode
    config_in1['wavelet_constants']['Nst'] = 512

    wc1 = get_wavelet_model(config_in1)
    lc1 = get_lisa_constants(config_in1)

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    config_in2['lisa_constants']['noise_curve_mode'] = noise_curve_mode
    config_in2['wavelet_constants']['Nst'] = 512

    # replace the Nf and Nt from the file
    config_in2['wavelet_constants']['Nf'] = int(wc1.Nf * channel_mult[0])
    config_in2['wavelet_constants']['Nt'] = int(wc1.Nt * channel_mult[1])
    config_in2['wavelet_constants']['dt'] = float(wc1.dt * channel_mult[2])
    config_in2['wavelet_constants']['mult'] = int(wc1.mult * channel_mult[3])

    wc2 = get_wavelet_model(config_in2)
    lc2 = get_lisa_constants(config_in2)

    # check proper set up
    assert lc1.noise_curve_mode == lc2.noise_curve_mode

    noise1 = instrument_noise_AET_wdm_m(lc1, wc1)
    noise2 = instrument_noise_AET_wdm_m(lc2, wc2)

    seed1 = 31415
    seed2 = 31415

    noise_manager1 = DiagonalStationaryDenseNoiseModel(noise1, wc1, prune=1, nc_snr=noise1.shape[1], seed=seed1)
    noise_manager2 = DiagonalStationaryDenseNoiseModel(noise2, wc2, prune=1, nc_snr=noise2.shape[1], seed=seed2)

    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=amp0_use,  # amplitude
        phi0=0.3,  # phase at t=0
        F0=1.0e-4,  # initial frequency (Hz)
        FTd0=3.0e-12,  # frequency derivative (Hz/s)
    )

    assert intrinsic.FTd0 < 8 * wc1.DF / wc1.Tw
    assert intrinsic.FTd0 < 8 * wc2.DF / wc2.Tw
    assert intrinsic.FTd0 < wc1.dfd * (wc1.Nfd - wc1.Nfd_negative)
    assert intrinsic.FTd0 < wc2.dfd * (wc2.Nfd - wc2.Nfd_negative)
    assert intrinsic.FTd0 >= wc1.dfd * (-wc1.Nfd_negative)
    assert intrinsic.FTd0 >= wc2.dfd * (-wc2.Nfd_negative)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    nt_lim_waveform1 = PixelGenericRange(0, wc1.Nt, wc1.DT, 0.0)
    nt_lim_waveform2 = PixelGenericRange(0, wc2.Nt, wc2.DT, 0.0)

    nt_lim_snr1 = PixelGenericRange(0, wc1.Nt, wc1.DT, 0.0)
    nt_lim_snr2 = PixelGenericRange(0, wc2.Nt, wc2.DT, 0.0)

    waveform1 = LinearFrequencyWaveletWaveformTime(
        params,
        wc1,
        lc1,
        nt_lim_waveform1,
        table_cache_mode='check',
        table_output_mode='skip',
        response_mode=response_mode,
    )
    waveform2 = LinearFrequencyWaveletWaveformTime(
        params,
        wc2,
        lc2,
        nt_lim_waveform2,
        table_cache_mode='check',
        table_output_mode='skip',
        response_mode=response_mode,
    )

    wavelet_waveform1 = waveform1.get_unsorted_coeffs()
    wavelet_waveform2 = waveform2.get_unsorted_coeffs()

    waveform_dense1 = wavelet_sparse_to_dense(wavelet_waveform1, wc1)
    waveform_dense2 = wavelet_sparse_to_dense(wavelet_waveform2, wc2)

    waveform_dense1_white = waveform_dense1 * noise_manager1.get_inv_chol_S()
    waveform_dense2_white = waveform_dense2 * noise_manager2.get_inv_chol_S()

    waveform_sparse1_white = wavelet_dense_select_sparse(waveform_dense1_white, wavelet_waveform1, wc1, inplace_mode=0)
    waveform_sparse2_white = wavelet_dense_select_sparse(waveform_dense2_white, wavelet_waveform2, wc2, inplace_mode=0)

    waveform_sparse1_white_alt = whiten_sparse_data(wavelet_waveform1, noise_manager1.get_inv_chol_S(), wc1)
    waveform_sparse2_white_alt = whiten_sparse_data(wavelet_waveform2, noise_manager2.get_inv_chol_S(), wc2)

    assert_allclose(
        waveform_sparse1_white_alt.wave_value, waveform_sparse1_white.wave_value, atol=1.0e-100, rtol=1.0e-14
    )
    assert_allclose(
        waveform_sparse2_white_alt.wave_value, waveform_sparse2_white.wave_value, atol=1.0e-100, rtol=1.0e-14
    )

    assert_array_equal(waveform_sparse1_white_alt.pixel_index, waveform_sparse1_white.pixel_index)
    assert_array_equal(waveform_sparse1_white_alt.n_set, waveform_sparse1_white.n_set)
    assert_array_equal(waveform_sparse1_white_alt.n_pixel_max, waveform_sparse1_white.n_pixel_max)

    assert_array_equal(waveform_sparse2_white_alt.pixel_index, waveform_sparse2_white.pixel_index)
    assert_array_equal(waveform_sparse2_white_alt.n_set, waveform_sparse2_white.n_set)
    assert_array_equal(waveform_sparse2_white_alt.n_pixel_max, waveform_sparse2_white.n_pixel_max)

    # get realizations of the noise and mask the non-overlapping part so we can isolate the band-limited noise
    seed_gen = 31415926
    n_seed = 1
    rng = np.random.default_rng(seed_gen)
    seeds = rng.integers(0, np.iinfo(np.int_).max, n_seed)

    signal_noise1 = np.zeros(n_seed)
    signal_noise2 = np.zeros(n_seed)

    noise_noise1 = np.zeros(n_seed)
    noise_noise2 = np.zeros(n_seed)

    data_signal1 = np.zeros(n_seed)
    data_signal2 = np.zeros(n_seed)

    data_noise1 = np.zeros(n_seed)
    data_noise2 = np.zeros(n_seed)

    data_data1 = np.zeros(n_seed)
    data_data2 = np.zeros(n_seed)

    log_likelihood1 = np.zeros(n_seed)
    log_likelihood2 = np.zeros(n_seed)

    signal_signal1 = np.full(n_seed, np.sum(waveform_sparse1_white.wave_value * waveform_sparse1_white.wave_value))
    signal_signal2 = np.full(n_seed, np.sum(waveform_sparse2_white.wave_value * waveform_sparse2_white.wave_value))

    assert_allclose(signal_signal1, np.sum(waveform_dense1_white * waveform_dense1_white), atol=1.0e-100, rtol=1.0e-13)
    assert_allclose(signal_signal2, np.sum(waveform_dense2_white * waveform_dense2_white), atol=1.0e-100, rtol=1.0e-13)

    for itrs, seed_loc in enumerate(seeds):
        noise_real1_white = noise_manager1.generate_dense_noise(seed_override=seed_loc, white_mode=1)
        noise_real2_white = noise_manager2.generate_dense_noise(seed_override=seed_loc, white_mode=1)

        noise_real1_white_alt = (
            noise_manager1.generate_dense_noise(seed_override=seed_loc, white_mode=0) * noise_manager1.get_inv_chol_S()
        )
        noise_real2_white_alt = (
            noise_manager2.generate_dense_noise(seed_override=seed_loc, white_mode=0) * noise_manager2.get_inv_chol_S()
        )

        assert_allclose(noise_real1_white[:, :], noise_real1_white_alt[:, :], atol=1.0e-100, rtol=1.0e-13)
        assert_allclose(noise_real2_white[:, :], noise_real2_white_alt[:, :], atol=1.0e-100, rtol=1.0e-13)

        noise_sparse1_white = wavelet_dense_select_sparse(noise_real1_white, wavelet_waveform1, wc1, inplace_mode=0)
        noise_sparse2_white = wavelet_dense_select_sparse(noise_real2_white, wavelet_waveform2, wc2, inplace_mode=0)

        pixel_dist1 = noise_sparse1_white.wave_value[:, : noise_sparse1_white.n_set.min()].flatten()
        pixel_dist2 = noise_sparse2_white.wave_value[:, : noise_sparse2_white.n_set.min()].flatten()

        _ = unit_normal_battery(pixel_dist1, do_assert=True)
        _ = unit_normal_battery(pixel_dist2, do_assert=True)

        signal_noise1[itrs] = np.sum(noise_sparse1_white.wave_value * waveform_sparse1_white.wave_value)
        signal_noise2[itrs] = np.sum(noise_sparse2_white.wave_value * waveform_sparse2_white.wave_value)

        assert_allclose(
            signal_noise1[itrs], np.sum(waveform_dense1_white * noise_real1_white), atol=1.0e-100, rtol=1.0e-10
        )
        assert_allclose(
            signal_noise2[itrs], np.sum(waveform_dense2_white * noise_real2_white), atol=1.0e-100, rtol=1.0e-10
        )

        noise_noise1[itrs] = np.sum(noise_sparse1_white.wave_value * noise_sparse1_white.wave_value)
        noise_noise2[itrs] = np.sum(noise_sparse2_white.wave_value * noise_sparse2_white.wave_value)

        data_dense1_white = noise_real1_white + waveform_dense1_white
        data_dense2_white = noise_real2_white + waveform_dense2_white

        data_sparse1_white_alt = wavelet_dense_select_sparse(data_dense1_white, wavelet_waveform1, wc1, inplace_mode=0)
        data_sparse2_white_alt = wavelet_dense_select_sparse(data_dense2_white, wavelet_waveform2, wc2, inplace_mode=0)

        data_sparse1_value_white = noise_sparse1_white.wave_value + waveform_sparse1_white.wave_value
        data_sparse2_value_white = noise_sparse2_white.wave_value + waveform_sparse2_white.wave_value

        assert_allclose(data_sparse1_white_alt.wave_value, data_sparse1_value_white, atol=1.0e-100, rtol=1.0e-14)
        assert_allclose(data_sparse2_white_alt.wave_value, data_sparse2_value_white, atol=1.0e-100, rtol=1.0e-14)

        data_signal1[itrs] = np.sum(data_sparse1_value_white * waveform_sparse1_white.wave_value)
        data_signal2[itrs] = np.sum(data_sparse2_value_white * waveform_sparse2_white.wave_value)

        data_noise1[itrs] = np.sum(data_sparse1_value_white * noise_sparse1_white.wave_value)
        data_noise2[itrs] = np.sum(data_sparse2_value_white * noise_sparse2_white.wave_value)

        data_data1[itrs] = np.sum(data_sparse1_value_white * data_sparse1_value_white)
        data_data2[itrs] = np.sum(data_sparse2_value_white * data_sparse2_value_white)

        log_likelihood1[itrs] = np.sum(
            get_sparse_likelihood_helper_prewhitened(
                wavelet_waveform1,
                data_dense1_white,
                nt_lim_snr1,
                wc1,
                noise_manager1.get_inv_chol_S(),
                noise_manager1.get_nc_snr(),
            )
        )
        log_likelihood2[itrs] = np.sum(
            get_sparse_likelihood_helper_prewhitened(
                wavelet_waveform2,
                data_dense2_white,
                nt_lim_snr2,
                wc2,
                noise_manager2.get_inv_chol_S(),
                noise_manager2.get_nc_snr(),
            )
        )

    data_noise1_alt = signal_noise1 + noise_noise1
    data_noise2_alt = signal_noise2 + noise_noise2

    assert_allclose(data_noise1, data_noise1_alt, atol=1.0e-100, rtol=1.0e-10)
    assert_allclose(data_noise2, data_noise2_alt, atol=1.0e-100, rtol=1.0e-10)

    data_signal1_alt = signal_signal1 + signal_noise1
    data_signal2_alt = signal_signal2 + signal_noise2

    assert_allclose(data_signal1, data_signal1_alt, atol=1.0e-100, rtol=1.0e-10)
    assert_allclose(data_signal2, data_signal2_alt, atol=1.0e-100, rtol=1.0e-10)

    data_data1_alt = signal_signal1 + 2 * signal_noise1 + noise_noise1
    data_data2_alt = signal_signal2 + 2 * signal_noise2 + noise_noise2

    assert_allclose(data_data1, data_data1_alt, atol=1.0e-100, rtol=1.0e-10)
    assert_allclose(data_data2, data_data2_alt, atol=1.0e-100, rtol=1.0e-10)

    log_likelihood1_alt = data_signal1 - signal_signal1 / 2
    log_likelihood2_alt = data_signal2 - signal_signal2 / 2

    assert_allclose(log_likelihood1, log_likelihood1_alt, atol=1.0e-100, rtol=1.0e-10)
    assert_allclose(log_likelihood2, log_likelihood2_alt, atol=1.0e-100, rtol=1.0e-10)
