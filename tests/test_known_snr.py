"""Test that the computed SNR scales as expected with changes in (Nf, Nt, dt, mult)."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose, assert_array_less
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import welch
from WDMWaveletTransforms.transform_freq_funcs import tukey
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_time

from LisaWaveformTools.instrument_noise import instrument_noise_AET, instrument_noise_AET_wdm_m
from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams, LinearFrequencyWaveletWaveformTime
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalStationaryDenseNoiseModel
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from WaveletWaveforms.sparse_waveform_functions import (
    PixelGenericRange,
    wavelet_sparse_to_dense,
)
from WaveletWaveforms.wdm_config import get_wavelet_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.mark.parametrize('amp0_mult', [1.0, 5.0])
@pytest.mark.parametrize('var_mult', [1.0, 7.0])
def test_snr_known_sinusoid(amp0_mult: float, var_mult: float) -> None:
    """Test recover known snr for sinusoid"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    noise_curve_mode = 1
    response_mode = 2
    amp0_use = 1.0 * amp0_mult

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    config_in['lisa_constants']['noise_curve_mode'] = noise_curve_mode

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    # pick a multiplier that will result in unit noise in the time and wavelet domains
    noise_mult = 2 * wc.dt * var_mult

    # the snr we are injecting; leading order result assuming T is large
    snr_expect: NDArray[np.floating] = np.full(lc.nc_snr, float(np.sqrt(1. / 2. * wc.Nf * wc.Nt * amp0_use**2 / var_mult)))

    noise = noise_mult * instrument_noise_AET_wdm_m(lc, wc)

    ts = np.arange(0, wc.Nf * wc.Nt) * wc.dt

    fs_full = np.fft.rfftfreq(wc.Nf * wc.Nt, d=wc.dt)

    noise_exp_full = np.zeros((fs_full.size, noise.shape[-1]))

    noise_exp_full[1:] = noise_mult * instrument_noise_AET(fs_full[1:], lc)

    seed = 31415
    nc_snr = int(noise.shape[1])

    noise_manager = DiagonalStationaryDenseNoiseModel(noise, wc, prune=1, nc_snr=nc_snr, seed=seed)
    noise_realization_wave = noise_manager.generate_dense_noise()

    # check the noise normalization is as expected for a flat spectrum
    assert_allclose(var_mult, np.var(noise_realization_wave), atol=1.e-100, rtol=4.e-3)

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

    signal_time = np.zeros((wc.Nf * wc.Nt, nc_snr))

    for itrc in range(nc_snr):
        signal_time[:, itrc] = inverse_wavelet_time(waveform_dense[:, :, itrc], wc.Nf, wc.Nt)

    noise_realization_time = np.zeros((wc.Nf * wc.Nt, nc_snr))

    for itrc in range(nc_snr):
        noise_realization_time[:, itrc] = inverse_wavelet_time(noise_realization_wave[:, :, itrc], wc.Nf, wc.Nt)

    signal_time_expect = np.zeros((wc.Nf * wc.Nt, nc_snr))
    for itrc in range(nc_snr):
        signal_time_expect[:, itrc] = amp0_use * np.cos(2 * np.pi * intrinsic.F0 * ts - intrinsic.phi0)

    # tukey(noise_realization_time, tukey_alpha, signal_time_expect.shape[0])

    # tukey(signal_time_expect, tukey_alpha, signal_time_expect.shape[0])

    # tukey(signal_time, tukey_alpha, signal_time.shape[0])

    # noise_realization_time = np.asarray(filtfilt(b, a, noise_realization_time, axis=0), dtype=np.float64)

    # check converted signal matches expectation
    assert_allclose(np.mean(signal_time, axis=0), np.mean(signal_time_expect, axis=0), atol=amp0_use * 3.e-6, rtol=1.e-14)
    assert_allclose(np.linalg.norm(signal_time - signal_time_expect, axis=0) / np.linalg.norm(signal_time_expect), 0., atol=amp0_use * 4.e-2, rtol=1.e-14)
    for itrc in range(nc_snr):
        assert_array_less(0.998, np.corrcoef(signal_time[:, itrc], signal_time_expect[:, itrc])[:, 0])

    signal_freq = np.fft.rfft(signal_time, axis=0) / float(np.sqrt(wc.Nt * wc.Nf / 2))

    fpsd, psd_noise = welch(noise_realization_time, fs=1 / wc.dt, nperseg=2 * wc.Nt, scaling='density', window='tukey', axis=0)
    fpsd, psd_signal = welch(signal_time, fs=1 / wc.dt, nperseg=2 * wc.Nt, scaling='density', window='tukey', axis=0)
    # psd_noise = psd_noise[np.argsort(fpsd)]
    # fpsd = fpsd[np.argsort(fpsd)]

    psd_noise_full = np.zeros((fs_full.size, lc.nc_snr))
    for itrc in range(lc.nc_snr):
        psd_noise_full[:, itrc] = np.abs(InterpolatedUnivariateSpline(fpsd, psd_noise[:, itrc], k=1, ext=1)(fs_full))

    # check normalization using the definition of the psd
    assert_allclose(np.var(noise_realization_time, axis=0), np.trapezoid(psd_noise, fpsd, axis=0), atol=1.e-100, rtol=7.e-3)
    assert_allclose(np.var(noise_realization_time, axis=0), np.trapezoid(psd_noise_full, fs_full, axis=0), atol=1.e-100, rtol=7.e-3)
    assert_allclose(np.var(noise_realization_time, axis=0), np.trapezoid(noise_exp_full, fs_full, axis=0), atol=1.e-100, rtol=7.e-3)
    assert_allclose(np.trapezoid(psd_noise, fpsd, axis=0), np.trapezoid(noise_exp_full, fs_full, axis=0), atol=1.e-100, rtol=7.e-3)
    assert_allclose(np.trapezoid(psd_noise_full, fs_full, axis=0), np.trapezoid(noise_exp_full, fs_full, axis=0), atol=1.e-100, rtol=7.e-3)
    assert_allclose(np.trapezoid(psd_noise, fpsd, axis=0), np.trapezoid(psd_noise_full, fs_full, axis=0), atol=1.e-100, rtol=7.e-3)

    assert_allclose(np.var(signal_time, axis=0), np.trapezoid(psd_signal, fpsd, axis=0), atol=1.e-100, rtol=4.e-3)

    snr_freq_direct: NDArray[np.floating] = np.zeros(lc.nc_snr)
    snr_freq_direct2: NDArray[np.floating] = np.zeros(lc.nc_snr)
    snr_freq_direct3: NDArray[np.floating] = np.zeros(lc.nc_snr)
    for itrc in range(lc.nc_snr):
        mask = (psd_noise_full[:, itrc] > 0.) & (np.isfinite(psd_noise_full[:, itrc]))
        snr_freq_direct[itrc] = np.sqrt((2 * wc.Tobs) * np.trapezoid((np.abs(np.sqrt(wc.dt) * signal_freq[mask, itrc])**2 / psd_noise_full[mask, itrc]), fs_full[mask]))
        mask = (noise_exp_full[:, itrc] > 0.) & (np.isfinite(noise_exp_full[:, itrc]))
        snr_freq_direct2[itrc] = np.sqrt((2 * wc.Tobs) * np.trapezoid((np.abs(np.sqrt(wc.dt) * signal_freq[mask, itrc])**2 / noise_exp_full[mask, itrc]), fs_full[mask]))
        mask = (psd_noise[:, itrc] > 0.) & (np.isfinite(psd_noise[:, itrc]))
        snr_freq_direct3[itrc] = np.sqrt((2 * wc.Tobs) * np.trapezoid(psd_signal[mask, itrc] / psd_noise[mask, itrc], fpsd[mask]))

    snr_channel: NDArray[np.floating] = noise_manager.get_sparse_snrs(wavelet_waveform, nt_lim_snr)
    # definition of snr of sum of n random variables
    snr_var: NDArray[np.floating] = np.sqrt(wc.Nf * wc.Nt * np.var(signal_time, axis=0) / np.var(noise_realization_time, axis=0))
    snr_mean: NDArray[np.floating] = np.sqrt(wc.Nf * wc.Nt * np.mean(signal_time**2, axis=0) / np.mean(noise_realization_time**2, axis=0))

    snr_var_expect: NDArray[np.floating] = np.sqrt(wc.Nf * wc.Nt * np.var(signal_time_expect, axis=0) / np.var(noise_realization_time, axis=0))

    snr_var_wave: NDArray[np.floating] = np.sqrt((wc.Nf - 1) * wc.Nt * np.var(waveform_dense[:, 1:], axis=(0, 1)) / np.var(noise_realization_time[:, 1:], axis=(0, 1)))

    print(snr_expect)
    print(snr_var)
    print(snr_var_wave)
    print(snr_mean)
    print(snr_freq_direct)
    print(snr_freq_direct2)
    print(snr_freq_direct3)
    print(snr_channel)

    assert_allclose(snr_var, snr_mean, atol=1.e-100, rtol=1.e-10)

    # check variances match
    assert_allclose(snr_var, snr_var_wave, atol=1.e-100, rtol=7.e-3)
    assert_allclose(snr_var, snr_var_expect, atol=1.e-100, rtol=7.e-3)

    # check variance snr matches injected
    assert_allclose(snr_var, snr_expect, atol=1.e-100, rtol=5.e-3)
    assert_allclose(snr_var_expect, snr_expect, atol=1.e-100, rtol=5.e-3)

    # check snr computed by pipeline matches defintion
    assert_allclose(snr_var, snr_channel, atol=1.e-100, rtol=6.e-3)
    assert_allclose(snr_expect, snr_channel, atol=1.e-100, rtol=6.e-3)
    assert_allclose(snr_var_expect, snr_channel, atol=1.e-100, rtol=6.e-3)

    # check snr computed by psds match definition
    assert_allclose(snr_var, snr_freq_direct, atol=1.e-100, rtol=3.e-2)
    assert_allclose(snr_var, snr_freq_direct2, atol=1.e-100, rtol=3.e-3)
    assert_allclose(snr_var, snr_freq_direct3, atol=1.e-100, rtol=3.e-2)

    assert_allclose(snr_expect, snr_freq_direct, atol=1.e-100, rtol=3.e-2)
    assert_allclose(snr_expect, snr_freq_direct2, atol=1.e-100, rtol=6.e-3)
    assert_allclose(snr_expect, snr_freq_direct3, atol=1.e-100, rtol=3.e-2)

    # check snr computed by psd matches method
    assert_allclose(snr_channel, snr_freq_direct, atol=1.e-100, rtol=3.e-2)
    assert_allclose(snr_channel, snr_freq_direct2, atol=1.e-100, rtol=7.e-3)
    assert_allclose(snr_channel, snr_freq_direct3, atol=1.e-100, rtol=3.e-2)


@pytest.mark.parametrize('amp0_mult', [1.0])
@pytest.mark.parametrize('var_mult', [1.0])
@pytest.mark.parametrize('ftd0', [-2.e-12, -5.148215847186539e-13, -1.0296431694373077e-17, 0., 1.0296431694373077e-17, 2.5741079235932694e-13, 5.148215847186539e-13, 1.0296431694373078e-12, 1.e-10])
@pytest.mark.parametrize('f0_mult', [4.99999, 5., 5.00001, 5.5, 5.99999, 6., 6.00001, 6.5, 6.159375])
@pytest.mark.parametrize('phi0', [0., 0.3, np.sqrt(2.), np.pi / 4., np.pi / 2., np.pi])
def test_snr_known_linear(amp0_mult: float, var_mult: float, ftd0: float, f0_mult: float, phi0: float) -> None:
    """Test recover known snr for sinusoid"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    noise_curve_mode = 1
    response_mode = 2
    amp0_use = 1.0 * amp0_mult

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    config_in['lisa_constants']['noise_curve_mode'] = noise_curve_mode

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)
    tukey_alpha = 0.02

    # pick a multiplier that will result in unit noise in the time and wavelet domains
    noise_mult = 2 * wc.dt * var_mult

    # the snr we are injecting; leading order result assuming T is large
    snr_expect: NDArray[np.floating] = np.full(lc.nc_snr, float(np.sqrt(1. / 2. * wc.Nf * wc.Nt * amp0_use**2 / var_mult)))

    noise = noise_mult * instrument_noise_AET_wdm_m(lc, wc)

    ts = np.arange(0, wc.Nf * wc.Nt) * wc.dt

    seed = 31415
    nc_snr = int(noise.shape[1])

    noise_manager = DiagonalStationaryDenseNoiseModel(noise, wc, prune=1, nc_snr=nc_snr, seed=seed)
    noise_realization_wave = noise_manager.generate_dense_noise()

    # check the noise normalization is as expected for a flat spectrum
    assert_allclose(var_mult, np.var(noise_realization_wave), atol=1.e-100, rtol=4.e-3)

    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=amp0_use,  # amplitude
        phi0=phi0,  # phase at t=0
        F0=wc.DF * f0_mult,  # initial frequency (Hz)
        FTd0=ftd0,  # frequency derivative (Hz/s)
    )

    assert_array_less(intrinsic.FTd0, 8 * wc.DF / wc.Tw)
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

    signal_time = np.zeros((wc.Nf * wc.Nt, nc_snr))

    for itrc in range(nc_snr):
        signal_time[:, itrc] = inverse_wavelet_time(waveform_dense[:, :, itrc], wc.Nf, wc.Nt)

    noise_realization_time = np.zeros((wc.Nf * wc.Nt, nc_snr))

    for itrc in range(nc_snr):
        noise_realization_time[:, itrc] = inverse_wavelet_time(noise_realization_wave[:, :, itrc], wc.Nf, wc.Nt)

    signal_time_expect = np.zeros((wc.Nf * wc.Nt, nc_snr))
    for itrc in range(nc_snr):
        signal_time_expect[:, itrc] = amp0_use * np.cos(2 * np.pi * intrinsic.F0 * ts + np.pi * intrinsic.FTd0 * ts**2 - intrinsic.phi0)

    # tukey(noise_realization_time, tukey_alpha, signal_time_expect.shape[0])

    tukey(signal_time_expect, tukey_alpha, signal_time_expect.shape[0])

    tukey(signal_time, tukey_alpha, signal_time.shape[0])

    # noise_realization_time = np.asarray(filtfilt(b, a, noise_realization_time, axis=0), dtype=np.float64)
    # import matplotlib.pyplot as plt
    # plt.plot(signal_time[:,0])
    # plt.plot(signal_time_expect[:,0])
    # plt.show()

    # check converted signal matches expectation
    assert_allclose(np.mean(signal_time, axis=0), np.mean(signal_time_expect, axis=0), atol=amp0_use * 3.e-6, rtol=1.e-14)
    assert_allclose(np.linalg.norm(signal_time - signal_time_expect, axis=0) / np.linalg.norm(signal_time_expect), 0., atol=amp0_use * 4.e-2, rtol=1.e-14)
    for itrc in range(nc_snr):
        assert_array_less(0.998, np.corrcoef(signal_time[:, itrc], signal_time_expect[:, itrc])[:, 0])

    fpsd, psd_noise = welch(noise_realization_time, fs=1 / wc.dt, nperseg=2 * wc.Nt, scaling='density', window='tukey', axis=0)
    fpsd, psd_signal = welch(signal_time, fs=1 / wc.dt, nperseg=2 * wc.Nt, scaling='density', window='tukey', axis=0)

    # check normalization using the definition of the psd
    assert_allclose(np.var(noise_realization_time, axis=0), np.trapezoid(psd_noise, fpsd, axis=0), atol=1.e-100, rtol=7.e-3)

    assert_allclose(np.var(signal_time, axis=0), np.trapezoid(psd_signal, fpsd, axis=0), atol=1.e-100, rtol=5.e-3)

    snr_freq_direct3: NDArray[np.floating] = np.zeros(lc.nc_snr)
    for itrc in range(lc.nc_snr):
        mask = (psd_noise[:, itrc] > 0.) & (np.isfinite(psd_noise[:, itrc]))
        snr_freq_direct3[itrc] = np.sqrt((2 * wc.Tobs) * np.trapezoid(psd_signal[mask, itrc] / psd_noise[mask, itrc], fpsd[mask]))

    snr_channel: NDArray[np.floating] = noise_manager.get_sparse_snrs(wavelet_waveform, nt_lim_snr)
    # definition of snr of sum of n random variables
    snr_var: NDArray[np.floating] = np.sqrt(wc.Nf * wc.Nt * np.var(signal_time, axis=0) / np.var(noise_realization_time, axis=0))

    snr_var_expect: NDArray[np.floating] = np.sqrt(wc.Nf * wc.Nt * np.var(signal_time_expect, axis=0) / np.var(noise_realization_time, axis=0))

    snr_var_wave: NDArray[np.floating] = np.sqrt((wc.Nf - 1) * wc.Nt * np.var(waveform_dense[:, 1:], axis=(0, 1)) / np.var(noise_realization_time[:, 1:], axis=(0, 1)))

    print(snr_expect)
    print(snr_var)
    print(snr_var_wave)
    print(snr_freq_direct3)
    print(snr_channel)

    # check variances match
    assert_allclose(snr_var, snr_var_wave, atol=1.e-100, rtol=9.e-3)
    assert_allclose(snr_var, snr_var_expect, atol=1.e-100, rtol=7.e-3)

    # check variance snr matches injected
    assert_allclose(snr_var, snr_expect, atol=1.e-100, rtol=6.e-3)
    assert_allclose(snr_var_expect, snr_expect, atol=1.e-100, rtol=6.e-3)

    # check snr computed by pipeline matches defintion
    assert_allclose(snr_var, snr_channel, atol=1.e-100, rtol=8.e-3)
    assert_allclose(snr_expect, snr_channel, atol=1.e-100, rtol=6.e-3)
    assert_allclose(snr_var_expect, snr_channel, atol=1.e-100, rtol=8.e-3)

    # check snr computed by psds match definition
    assert_allclose(snr_var, snr_freq_direct3, atol=1.e-100, rtol=7.e-2)

    assert_allclose(snr_expect, snr_freq_direct3, atol=1.e-100, rtol=7.e-2)

    # check snr computed by psd matches method
    assert_allclose(snr_channel, snr_freq_direct3, atol=1.e-100, rtol=7.e-2)
