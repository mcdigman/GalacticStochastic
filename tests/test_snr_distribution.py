"""Test that the computed SNR scales as expected with changes in (Nf, Nt, dt, mult)."""

from pathlib import Path

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose, assert_array_equal

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

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    config_in2['lisa_constants']['noise_curve_mode'] = noise_curve_mode
    config_in2['wavelet_constants']['Nst'] = 512

    # replace the Nf and Nt from the file
    config_in2['wavelet_constants']['Nf'] = int(config_in1['wavelet_constants']['Nf'] * channel_mult[0])
    config_in2['wavelet_constants']['Nt'] = int(config_in1['wavelet_constants']['Nt'] * channel_mult[1])
    config_in2['wavelet_constants']['dt'] = float(config_in1['wavelet_constants']['dt'] * channel_mult[2])
    config_in2['wavelet_constants']['mult'] = int(config_in1['wavelet_constants']['mult'] * channel_mult[3])

    wc2 = get_wavelet_model(config_in2)
    lc2 = get_lisa_constants(config_in2)

    print(wc2.Nf, wc2.Nt, wc2.dt, wc2.mult)

    noise2 = instrument_noise_AET_wdm_m(lc2, wc2)

    fs2 = np.arange(0, wc2.Nf) * wc2.DF

    noise2_exp = np.zeros((fs2.size, noise2.shape[-1]))

    noise2_exp[1:] = instrument_noise_AET(fs2[1:], lc2)  # /(2 * wc2.dt)

    seed2 = 31415

    noise_manager2 = DiagonalStationaryDenseNoiseModel(noise2, wc2, prune=1, nc_snr=noise2.shape[1], seed=seed2)

    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=amp0_use,  # amplitude
        phi0=0.3,  # phase at t=0
        F0=1.0e-4,  # initial frequency (Hz)
        FTd0=3.0e-12,  # frequency derivative (Hz/s)
    )

    assert intrinsic.FTd0 < 8 * wc2.DF / wc2.Tw
    assert intrinsic.FTd0 < wc2.dfd * (wc2.Nfd - wc2.Nfd_negative)
    assert intrinsic.FTd0 >= wc2.dfd * (-wc2.Nfd_negative)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    nt_lim_waveform2 = PixelGenericRange(0, wc2.Nt, wc2.DT, 0.0)

    nt_lim_snr2 = PixelGenericRange(0, wc2.Nt, wc2.DT, 0.0)

    waveform2 = LinearFrequencyWaveletWaveformTime(
        params,
        wc2,
        lc2,
        nt_lim_waveform2,
        table_cache_mode='check',
        table_output_mode='skip',
        response_mode=response_mode,
    )

    wavelet_waveform2 = waveform2.get_unsorted_coeffs()

    waveform_dense2 = wavelet_sparse_to_dense(wavelet_waveform2, wc2)

    waveform_dense2_white = waveform_dense2 * noise_manager2.get_inv_chol_S()

    waveform_sparse2_white = wavelet_dense_select_sparse(waveform_dense2_white, wavelet_waveform2, wc2, inplace_mode=0)

    waveform_sparse2_white_alt = whiten_sparse_data(wavelet_waveform2, noise_manager2.get_inv_chol_S(), wc2)

    assert_allclose(
        waveform_sparse2_white_alt.wave_value, waveform_sparse2_white.wave_value, atol=1.0e-100, rtol=1.0e-14
    )

    assert_array_equal(waveform_sparse2_white_alt.pixel_index, waveform_sparse2_white.pixel_index)
    assert_array_equal(waveform_sparse2_white_alt.n_set, waveform_sparse2_white.n_set)
    assert_array_equal(waveform_sparse2_white_alt.n_pixel_max, waveform_sparse2_white.n_pixel_max)

    # get realizations of the noise and mask the non-overlapping part so we can isolate the band-limited noise
    seed_gen = 31415926
    n_seed = 1000
    rng = np.random.default_rng(seed_gen)
    seeds = rng.integers(0, np.iinfo(np.int_).max, n_seed)

    signal_noise2 = np.zeros(n_seed)

    noise_noise2 = np.zeros(n_seed)

    signal_signal2 = np.full(n_seed, np.sum(waveform_sparse2_white.wave_value * waveform_sparse2_white.wave_value))
    signal_noise_noise_noise2 = np.full(
        n_seed, np.sum(waveform_sparse2_white.wave_value * waveform_sparse2_white.wave_value)
    )

    assert_allclose(signal_signal2, np.sum(waveform_dense2_white * waveform_dense2_white), atol=1.0e-100, rtol=1.0e-13)

    # noise_real2_white = noise_manager2.generate_dense_noise(seed_override=seed_loc, white_mode=1)
    noise_real2_white = noise_manager2.generate_dense_noise(white_mode=1)

    noise_sparse2_white = wavelet_dense_select_sparse(noise_real2_white, wavelet_waveform2, wc2, inplace_mode=0)

    # for itrs, seed_loc in enumerate(seeds):
    for itrs in range(seeds.size):
        # noise_real2_white = noise_manager2.generate_dense_noise(seed_override=seed_loc, white_mode=1)
        # overwrite sparsely for speed
        for itrc in range(noise2.shape[1]):
            n_set_loc: int = int(noise_sparse2_white.n_set[itrc])
            noise_sparse2_white.wave_value[itrc, :n_set_loc] = rng.normal(0.0, 1.0, n_set_loc)

        signal_noise2[itrs] = np.sum(noise_sparse2_white.wave_value * waveform_sparse2_white.wave_value)

        noise_noise2[itrs] = np.sum(noise_sparse2_white.wave_value * noise_sparse2_white.wave_value)

    data_noise2 = signal_noise2 + noise_noise2

    # cov_need = 2 * signal_noise_noise_noise2.mean()  # mean of signal_noise is zero

    data_signal2 = signal_signal2 + signal_noise2

    data_data2 = signal_signal2 + 2 * signal_noise2 + noise_noise2

    log_likelihood2 = data_signal2 - signal_signal2 / 2

    log_likelihood_mean_exp2 = np.mean(signal_signal2 / 2.0)
    # assert_allclose(log_likelihood_mean_exp1, log_likelihood_mean_exp2, atol=1.e-100, rtol=1.e-10)

    snr_channel2 = noise_manager2.get_sparse_snrs(wavelet_waveform2, nt_lim_snr2)
    snr_tot2: float = float(np.linalg.norm(snr_channel2))

    n_set_tot2: int = int(np.sum(wavelet_waveform2.n_set))

    print('2 likelihood mean, std, min, max', log_likelihood2.mean(), log_likelihood2.std(), log_likelihood2.min(), log_likelihood2.max())
    print('2 likelihood expect', log_likelihood_mean_exp2)
    print('2 signal*noise mean, std, min, max', signal_noise2.mean(), signal_noise2.std(), signal_noise2.min(), signal_noise2.max())
    print('2 noise*noise mean, std, min, max', noise_noise2.mean(), noise_noise2.std(), noise_noise2.min(), noise_noise2.max())
    print(
        '2 data*signal mean, std, min, max',
        data_signal2.mean(),
        data_signal2.std(),
        data_signal2.min(),
        data_signal2.max(),
    )
    print(
        '2 data*noise mean, std, min, max',
        data_noise2.mean(),
        data_noise2.std(),
        data_noise2.min(),
        data_noise2.max(),
    )
    print(
        '2 data*data mean, std, min, max',
        data_data2.mean(),
        data_data2.std(),
        data_data2.min(),
        data_data2.max(),
    )
    print(
        '2 data*data mean diff, std',
        (data_data2.mean() - n_set_tot2 - signal_signal2.mean()) / np.sqrt(2 * n_set_tot2 + 2 * snr_tot2**2),
        data_data2.std() / np.sqrt(2 * n_set_tot2 + 2 * snr_tot2**2),
    )
    print(
        '2 signal noise noise noise mean, std, min, max',
        signal_noise_noise_noise2.mean(),
        signal_noise_noise_noise2.std(),
        signal_noise_noise_noise2.min(),
        signal_noise_noise_noise2.max(),
    )
    print(
        '2 data*data exp, std',
        np.sqrt(2 * n_set_tot2 + 2 * snr_tot2**2) / np.sqrt(n_seed),
        np.sqrt(2.0) / np.sqrt(n_seed) / np.sqrt(2 * n_set_tot2 + 2 * snr_tot2**2),
    )
    print('2 snr channels, total', snr_channel2, snr_tot2)
    print('2 signal*signal', signal_signal2.mean())
    print('2 set, total', wavelet_waveform2.n_set, n_set_tot2)
    print('2 (signal* noise std)/(total snr)', signal_noise2.std() / snr_tot2)
    std_data_data: float = float(np.sqrt(4 * snr_tot2**2 + 2 * n_set_tot2))
    # print('2 cov', cov_need, 2 * snr_tot2 * 15.0, cov_got, cov_got2, std_data_data)

    # import matplotlib.pyplot as plt
    # plt.hist(log_likelihood2 - log_likelihood_mean_exp2, 100, alpha=0.8, density=True)
    # plt.show()

    # plt.hist(signal_noise2, 100, alpha=0.8, density=True)
    # plt.show()

    # plt.hist((noise_noise2 - n_set_tot2)/np.sqrt(2*n_set_tot2), 100, alpha=0.8, density=True)
    # plt.show()

    # plt.hist((data_noise2 - n_set_tot2)/np.sqrt(2*n_set_tot2 + snr_tot2**2), 100, alpha=0.8, density=True)
    # plt.show()

    # plt.hist((data_data2 - n_set_tot2 - signal_signal2)/std_data_data, 100, alpha=0.8, density=True)
    # plt.show()
    # var_data_data = 2*var(signal noise) + var(noise noise) + 4 * cov(signal noise, noise noise)
    # std_data_data = np.sqrt(2*snr_tot2**2 + 2*n_set_tot2 + 4 * cov(signal noise, noise noise))
    # print(std_data_data, np.sqrt(2*n_set_tot2 + 4*snr_tot2**2 + 4 * cov_got), data_data2.std())

    assert_allclose(snr_tot2**2, 2 * log_likelihood_mean_exp2, atol=1.0e-10, rtol=1.0e-10)

    _ = unit_normal_battery(signal_noise2, mult=snr_tot2, do_assert=True)
    _ = unit_normal_battery(data_signal2 - signal_signal2, mult=snr_tot2, do_assert=True)
    _ = unit_normal_battery(noise_noise2 - n_set_tot2, mult=float(np.sqrt(2 * n_set_tot2)), do_assert=True)
    _ = unit_normal_battery(data_noise2 - n_set_tot2, mult=float(np.sqrt(2 * n_set_tot2 + snr_tot2**2)), do_assert=True)
    _ = unit_normal_battery(data_data2 - n_set_tot2 - signal_signal2, mult=std_data_data, do_assert=True)
    _ = unit_normal_battery(log_likelihood2 - signal_signal2 / 2, mult=snr_tot2, do_assert=True)


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

    print(wc1.Nf, wc1.Nt, wc1.dt, wc1.mult)
    print(wc2.Nf, wc2.Nt, wc2.dt, wc2.mult)
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

    print(
        intrinsic.FTd0,
        8 * wc1.DF / wc1.Tw,
        8 * wc2.DF / wc2.Tw,
        wc1.DF**2 / 8,
        wc2.DF**2 / 8,
        wc1.dfd * (wc1.Nfd - wc1.Nfd_negative),
        wc2.dfd * (wc2.Nfd - wc2.Nfd_negative),
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

    snr_channel2 = noise_manager2.get_sparse_snrs(wavelet_waveform2, nt_lim_snr2)
    snr_tot2 = float(np.linalg.norm(snr_channel2))

    print('1 likelihood mean, std, min, max', log_likelihood1.mean(), log_likelihood1.std(), log_likelihood1.min(), log_likelihood1.max())
    print('1 likelihood expect', log_likelihood_mean_exp1)
    print('1 signal*noise mean, std, min, max', signal_noise1.mean(), signal_noise1.std(), signal_noise1.min(), signal_noise1.max())
    print('1 noise*noise mean, std, min, max', noise_noise1.mean(), noise_noise1.std(), noise_noise1.min(), noise_noise1.max())
    print('1 data*signal mean, std, min, max', data_signal1.mean(), data_signal1.std(), data_signal1.min(), data_signal1.max())
    print('1 data*noise mean, std, min, max', data_noise1.mean(), data_noise1.std(), data_noise1.min(), data_noise1.max())
    print('1 data*data mean, std, min, max', data_data1.mean(), data_data1.std(), data_data1.min(), data_data1.max())
    print('1 snr channels, total', snr_channel1, snr_tot1)
    print('1 signal*signal', signal_signal1.mean())
    n_set_tot1 = np.sum(wavelet_waveform1.n_set)
    print('1 set, total', wavelet_waveform1.n_set, n_set_tot1)
    print('1 (signal* noise std)/(total snr)', signal_noise1.std() / snr_tot1)

    print('2 likelihood mean, std, min, max', log_likelihood2.mean(), log_likelihood2.std(), log_likelihood2.min(), log_likelihood2.max())
    print('2 likelihood expect', log_likelihood_mean_exp2)
    print('2 signal*noise mean, std, min, max', signal_noise2.mean(), signal_noise2.std(), signal_noise2.min(), signal_noise2.max())
    print('2 noise*noise mean, std, min, max', noise_noise2.mean(), noise_noise2.std(), noise_noise2.min(), noise_noise2.max())
    print('2 data*signal mean, std, min, max', data_signal2.mean(), data_signal2.std(), data_signal2.min(), data_signal2.max())
    print('2 data*noise mean, std, min, max', data_noise2.mean(), data_noise2.std(), data_noise2.min(), data_noise2.max())
    print('2 data*data mean, std, min, max', data_data2.mean(), data_data2.std(), data_data2.min(), data_data2.max())
    print('2 snr channels, total', snr_channel2, snr_tot2)
    print('2 signal*signal', signal_signal2.mean())
    n_set_tot2 = np.sum(wavelet_waveform2.n_set)
    print('2 set, total', wavelet_waveform2.n_set, n_set_tot2)
    print('2 (signal* noise std)/(total snr)', signal_noise2.std() / snr_tot2)

    print('12 snr rat^2, exp likelihood rat', (snr_tot1 / snr_tot2) ** 2, log_likelihood_mean_exp1 / log_likelihood_mean_exp2)

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

    print(wc1.Nf, wc1.Nt, wc1.dt, wc1.mult)
    print(wc2.Nf, wc2.Nt, wc2.dt, wc2.mult)
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

    print(
        intrinsic.FTd0,
        8 * wc1.DF / wc1.Tw,
        8 * wc2.DF / wc2.Tw,
        wc1.DF**2 / 8,
        wc2.DF**2 / 8,
        wc1.dfd * (wc1.Nfd - wc1.Nfd_negative),
        wc2.dfd * (wc2.Nfd - wc2.Nfd_negative),
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
