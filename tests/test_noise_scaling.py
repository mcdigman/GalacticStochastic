"""Test that the computed SNR scales as expected with changes in (Nf, Nt, dt, mult)."""

from pathlib import Path

import numpy as np
import pytest
import scipy.signal
import tomllib
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import InterpolatedUnivariateSpline
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_time

from LisaWaveformTools.instrument_noise import instrument_noise_AET, instrument_noise_AET_wdm_m
from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams, LinearFrequencyWaveletWaveformTime
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalStationaryDenseNoiseModel
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.wdm_config import get_wavelet_model

# TODO add higher precision test using identical noise generated natively in time domain
# TODO add test with large enough slope to cross multiple pixels


# scaling on (Nf, Nt, dt, mult) in the second configuration
@pytest.mark.parametrize(
    'channel_mult',
    [
        (0.5, 0.5, 0.5, 1.0),
        (0.5, 0.5, 1.0, 1.0),
        (0.5, 0.5, 2.0, 1.0),
        (0.5, 1.0, 0.5, 1.0),
        (0.5, 1.0, 1.0, 1.0),
        (0.5, 1.0, 2.0, 1.0),
        (0.5, 2.0, 0.5, 1.0),
        (0.5, 2.0, 1.0, 1.0),
        (0.5, 2.0, 2.0, 1.0),
        (1.0, 0.5, 0.5, 1.0),
        (1.0, 0.5, 1.0, 1.0),
        (1.0, 0.5, 2.0, 1.0),
        (1.0, 1.0, 0.5, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 2.0, 1.0),
        (1.0, 2.0, 0.5, 1.0),
        (1.0, 2.0, 1.0, 1.0),
        (1.0, 2.0, 2.0, 1.0),
        (2.0, 0.5, 0.5, 1.0),
        (2.0, 0.5, 1.0, 1.0),
        (2.0, 0.5, 2.0, 1.0),
        (2.0, 1.0, 0.5, 1.0),
        (2.0, 1.0, 1.0, 1.0),
        (2.0, 1.0, 2.0, 1.0),
        (2.0, 2.0, 0.5, 1.0),
        (2.0, 2.0, 1.0, 1.0),
        (2.0, 2.0, 2.0, 1.0),
        (0.5, 0.5, 4.0, 1.0),
        (0.5, 4.0, 0.5, 1.0),
        (4.0, 0.5, 0.5, 1.0),
        (1.0, 3.0, 1.0, 1.0),
        (0.25, 4.0, 1.0, 1.0),
        (0.75, 1.0, 0.75, 1.0),
        (0.5, 8.0, 0.25, 1.0),
        (8.0, 0.125, 1.0, 0.5),
        (4.0, 0.25, 1.0, 0.5),
        (2.0, 0.5, 1.0, 0.5),
        (0.5, 0.5, 0.5, 0.5),
        (0.125, 8.0, 1.0, 2.0),
        (1.0, 1.0, 0.75, 2.0),
        (1.0, 0.5, 1.0, 2.0),
        (2.0, 2.0, 2.0, 2.0),
        (1.0, 1.0, 1.0, 2.0),
        (0.5, 1.0, 1.0, 2.0),
        (0.5, 1.0, 1.0, 4.0),
        (1.0, 1.0, 0.5, 4.0),
        (0.5, 1.0, 0.5, 16.0),
    ],
)
def test_noise_generation_scaling_curve(channel_mult: tuple[float, float, float, float]) -> None:
    """Test the scaling between (Nf, Nt, dt, mult) and SNR^2"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

    config_in1['lisa_constants']['noise_curve_mode'] = 0
    config_in1['wavelet_constants']['Nst'] = 512

    wc1 = get_wavelet_model(config_in1)
    lc1 = get_lisa_constants(config_in1)

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    config_in2['lisa_constants']['noise_curve_mode'] = 0
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

    t_obs1 = wc1.Tobs
    t_obs2 = wc2.Tobs
    time_rat_waveform = t_obs1 / t_obs2

    if wc1.Nf * wc1.Nt == wc2.Nf * wc2.Nt:
        assert channel_mult[0] * channel_mult[1] == 1.0
        n_pix_equal = True
        if channel_mult[2] == 1.0:
            assert t_obs1 == t_obs2
    else:
        n_pix_equal = False
        assert channel_mult[0] * channel_mult[1] != 1.0

    if channel_mult[0] * channel_mult[1] * channel_mult[2] == 1.0:
        time_equal = True
        assert t_obs1 == t_obs2
        assert time_rat_waveform == 1.0
    else:
        time_equal = False
        assert t_obs1 != t_obs2
        assert time_rat_waveform != 1.0

    if channel_mult[0] * channel_mult[2] == 1.0:
        width_equal = True
        assert wc1.DT == wc2.DT
        assert wc1.DF == wc2.DF
        assert wc1.DOM == wc2.DOM
        assert wc1.insDOM == wc2.insDOM
        assert wc1.B == wc2.B
        assert wc1.A == wc2.A
        assert wc1.BW == wc2.BW
        if wc1.Nsf == wc2.Nsf:
            assert wc1.df_bw == wc2.df_bw
    else:
        width_equal = False
        if wc1.Nsf == wc2.Nsf:
            assert wc1.df_bw != wc2.df_bw
        assert wc1.BW != wc2.BW
        assert wc1.A != wc2.A
        assert wc1.B != wc2.B
        assert wc1.insDOM != wc2.insDOM
        assert wc1.DOM != wc2.DOM
        assert wc1.DF != wc2.DF
        assert wc1.DT != wc2.DT

    if channel_mult[2] == 1.0:
        fmax_equal = True
        assert wc1.dt == wc2.dt
        assert wc1.Nf * wc1.DF == wc2.Nf * wc2.DF
        assert wc1.DOM * wc1.Nf == wc2.DOM * wc2.Nf  # OM = OM
    else:
        fmax_equal = False
        assert wc1.dt != wc2.dt
        assert wc1.Nf * wc1.DF != wc2.Nf * wc2.DF
        assert wc1.DOM * wc1.Nf != wc2.DOM * wc2.Nf

    if channel_mult[2] == 1.0 and channel_mult[0] == 1.0:
        spectrum_equal = True
        assert wc1.dt == wc2.dt
        assert wc1.Nf == wc2.Nf
        assert width_equal
        assert fmax_equal
        assert wc1.Nf == wc2.Nf
    else:
        spectrum_equal = False

    if channel_mult[0] * channel_mult[3] == 1.0:
        k_equal = True
        assert wc1.K == wc2.K
    else:
        k_equal = False
        assert wc1.K != wc2.K

    if channel_mult[0] * channel_mult[3] * channel_mult[2] == 1.0:
        tw_equal = True
        assert wc1.Tw == wc2.Tw
        assert wc1.dom == wc2.dom
    else:
        tw_equal = False
        assert wc1.Tw != wc2.Tw
        assert wc1.dom != wc2.dom

    if channel_mult[2] ** 2 * channel_mult[3] * channel_mult[0] ** 2 == 1.0 and wc1.dfdot == wc2.dfdot:
        dfd_equal = True
        assert wc1.dfd == wc2.dfd
    else:
        dfd_equal = False
        assert wc1.dfd != wc2.dfd

    if width_equal and k_equal:
        assert dfd_equal
        assert tw_equal

    noise1 = instrument_noise_AET_wdm_m(lc1, wc1)
    noise2 = instrument_noise_AET_wdm_m(lc2, wc2)

    fs1 = np.arange(0, wc1.Nf) * wc1.DF
    fs2 = np.arange(0, wc2.Nf) * wc2.DF

    noise1_exp = np.zeros((fs1.size, noise1.shape[-1]))
    noise2_exp = np.zeros((fs2.size, noise2.shape[-1]))

    noise1_exp[1:] = instrument_noise_AET(fs1[1:], lc1)  # /(2 * wc1.dt)
    noise2_exp[1:] = instrument_noise_AET(fs2[1:], lc2)  # /(2 * wc2.dt)

    # test the noise curve itself does not scale with observing time
    if spectrum_equal:
        assert_allclose(noise1[1:], noise2[1:], rtol=1.0e-13, atol=1.0e-40)

    # test all channels same if noise model flat
    if lc1.noise_curve_mode == 1:
        for itrc in range(noise1.shape[1]):
            assert_array_equal(noise1[1:, 0], noise1[1:, itrc])
    if lc2.noise_curve_mode == 1:
        for itrc in range(noise2.shape[1]):
            assert_array_equal(noise2[1:, 0], noise2[1:, itrc])

    seed1 = 31415
    seed2 = 31415

    noise_manager1 = DiagonalStationaryDenseNoiseModel(noise1, wc1, prune=1, nc_snr=noise1.shape[1], seed=seed1)
    noise_manager2 = DiagonalStationaryDenseNoiseModel(noise2, wc2, prune=1, nc_snr=noise2.shape[1], seed=seed2)

    fmin = max(wc1.DF, wc2.DF)
    fmax = min(wc1.DF * wc1.Nf, wc2.DF * wc2.Nf)

    mask1 = (fs1 >= fmin) & (fs1 <= fmax)
    mask2 = (fs2 >= fmin) & (fs2 <= fmax)

    fmin1 = np.min(fs1[mask1])
    fmax1 = np.max(fs1[mask1])
    frange1 = fmax1 - fmin1

    fmin2 = np.min(fs2[mask2])
    fmax2 = np.max(fs2[mask2])
    frange2 = fmax2 - fmin2

    # getting the integrated power in the overlapping band
    pow1_spect = np.sum(noise_manager1.get_S(), axis=0)
    pow2_spect = np.sum(noise_manager2.get_S(), axis=0)

    if spectrum_equal:
        assert_allclose(pow1_spect, time_rat_waveform * pow2_spect, atol=1.0e-40, rtol=1.0e-10)
        assert_allclose(
            pow1_spect[1:] / noise1_exp[1:],
            time_rat_waveform * pow2_spect[1:] / noise2_exp[1:],
            atol=1.0e-40,
            rtol=1.0e-8,
        )

    pow1 = np.sum(pow1_spect[mask1])
    pow2 = np.sum(pow2_spect[mask2])

    pow1_white = np.sum(pow1_spect[mask1] / noise1_exp[mask1])
    pow2_white = np.sum(pow2_spect[mask2] / noise2_exp[mask2])

    range_rat = frange1 / frange2

    print(frange1, frange2, range_rat)
    print(pow1, pow2)
    print(
        wc1.Nf * wc1.Nt / (wc2.Nf * wc2.Nt),
        time_rat_waveform,
        wc1.dt / wc2.dt,
        wc1.DF / wc2.DF,
        wc1.DT / wc2.DT,
        pow1 / pow2,
        pow1 / pow2 / time_rat_waveform,
    )
    print(wc1.Tw / wc2.Tw)
    print(wc1.Tobs / wc1.Tw, wc1.Tobs / wc2.Tw)
    # scale the spectra and check the integrated power matches expectation
    correct_fac = (wc2.dt / wc1.dt) * time_rat_waveform
    assert_allclose(correct_fac, (wc1.Nf * wc1.Nt) / (wc2.Nf * wc2.Nt))
    pow2_rescale = correct_fac * pow2
    pow2_white_rescale = correct_fac * pow2_white
    if n_pix_equal:
        if spectrum_equal:
            assert_allclose(pow1, pow2_rescale, atol=1.0e-40, rtol=5.0e-1)
        assert_allclose(pow1_white, pow2_white_rescale, atol=1.0e-40, rtol=4.0e-2)
    else:
        if spectrum_equal:
            assert_allclose(pow1, pow2_rescale, atol=1.0e-40, rtol=5.0e-1)
        assert_allclose(pow1_white, pow2_white_rescale, atol=1.0e-40, rtol=6.0e-2)

    # get realizations of the noise and mask the non-overlapping part so we can isolate the band-limited noise
    noise_real1 = noise_manager1.generate_dense_noise()
    noise_real2 = noise_manager2.generate_dense_noise()
    noise_real1[:, ~mask1, :] = 0.0
    noise_real2[:, ~mask2, :] = 0.0

    noise_real1_white = noise_real1.copy()
    noise_real1_white[:, 1:, :] = noise_real1_white[:, 1:] / np.sqrt(noise1_exp[1:])
    noise_real2_white = noise_real2.copy()
    noise_real2_white[:, 1:, :] = noise_real2_white[:, 1:] / np.sqrt(noise2_exp[1:])

    pow_real1 = np.sum(noise_real1[:, 1:] ** 2)
    pow_real2 = np.sum(noise_real2[:, 1:] ** 2)
    pow_real2_rescale = pow_real2 * correct_fac

    pow_real1_white = np.sum(noise_real1[:, 1:] ** 2 / noise1_exp[1:])
    pow_real2_white = np.sum(noise_real2[:, 1:] ** 2 / noise2_exp[1:])
    pow_real2_white_rescale = pow_real2_white * correct_fac

    pow_real1_white2 = np.sum(noise_real1_white[:, 1:] ** 2)
    pow_real2_white2 = np.sum(noise_real2_white[:, 1:] ** 2)
    pow_real2_white_rescale2 = pow_real2_white2 * correct_fac
    assert_allclose(pow_real1_white, pow_real1_white2)
    assert_allclose(pow_real2_white, pow_real2_white2)
    assert_allclose(pow_real2_white_rescale, pow_real2_white_rescale2)

    print(pow1, pow_real1, pow2, pow_real2, pow_real2_rescale)
    # check the power in the realizations matches what we expect
    assert_allclose(pow_real1, pow1, atol=1.0e-40, rtol=7.0e-3)
    assert_allclose(pow_real2, pow2, atol=1.0e-40, rtol=7.0e-3)
    assert_allclose(pow_real1 / pow_real2_rescale, pow1 / pow2_rescale, atol=1.0e-40, rtol=1.0e-1)

    assert_allclose(pow_real1_white, pow1_white, atol=1.0e-40, rtol=7.0e-3)
    assert_allclose(pow_real2_white, pow2_white, atol=1.0e-40, rtol=1.0e-2)
    assert_allclose(
        pow_real1_white / pow_real2_white_rescale, pow1_white / pow2_white_rescale, atol=1.0e-40, rtol=1.0e-1
    )

    # transform realized noise to time domain

    noise_time1 = np.zeros((wc1.Nt * wc1.Nf, noise_real1.shape[-1]))
    noise_time2 = np.zeros((wc2.Nt * wc2.Nf, noise_real2.shape[-1]))

    noise_time1_white = np.zeros((wc1.Nt * wc1.Nf, noise_real1.shape[-1]))
    noise_time2_white = np.zeros((wc2.Nt * wc2.Nf, noise_real2.shape[-1]))

    for itrc in range(noise_real1.shape[-1]):
        noise_time1[:, itrc] = inverse_wavelet_time(noise_real1[:, :, itrc], wc1.Nf, wc1.Nt)
        noise_time1_white[:, itrc] = inverse_wavelet_time(noise_real1_white[:, :, itrc], wc1.Nf, wc1.Nt)
    for itrc in range(noise_real2.shape[-1]):
        noise_time2[:, itrc] = inverse_wavelet_time(noise_real2[:, :, itrc], wc2.Nf, wc2.Nt)
        noise_time2_white[:, itrc] = inverse_wavelet_time(noise_real2_white[:, :, itrc], wc2.Nf, wc2.Nt)

    pow_real1_time = np.sum(noise_time1**2)
    pow_real2_time = np.sum(noise_time2**2)
    pow_real2_time_rescale = pow_real2_time * correct_fac

    pow_real1_time_white = np.sum(noise_time1_white**2)
    pow_real2_time_white = np.sum(noise_time2_white**2)
    pow_real2_time_white_rescale = pow_real2_time_white * correct_fac

    # check band limited power in time domain matches (parseval's theorem requires this)
    assert_allclose(pow_real1, pow_real1_time, atol=1.0e-40, rtol=1.0e-8)
    assert_allclose(pow_real2, pow_real2_time, atol=1.0e-40, rtol=1.0e-8)
    assert_allclose(pow_real1 / pow_real2_rescale, pow_real1_time / pow_real2_time_rescale, atol=1.0e-40, rtol=1.0e-8)

    assert_allclose(pow_real1_white, pow_real1_time_white, atol=1.0e-40, rtol=1.0e-8)
    assert_allclose(pow_real2_white, pow_real2_time_white, atol=1.0e-40, rtol=1.0e-8)
    assert_allclose(
        pow_real1_white / pow_real2_white_rescale,
        pow_real1_time_white / pow_real2_time_white_rescale,
        atol=1.0e-40,
        rtol=1.0e-8,
    )

    # somewhat narrower range than originally specified to trim edge effects
    mask1_lim = (fs1 >= fmin1 + 2 * wc1.DF + 2.0 / wc1.DT) & (fs1 <= fmax1 - 2 * wc1.DF - 2.0 / wc1.DT)
    mask2_lim = (fs2 >= fmin2 + 2 * wc2.DF + 2.0 / wc2.DT) & (fs2 <= fmax2 - 2 * wc2.DF - 2.0 / wc2.DT)

    assert np.any(mask1_lim)
    assert np.any(mask2_lim)

    noise_real1_spect = 2 * wc1.dt * np.mean(noise_real1**2, axis=0)
    noise_real2_spect = 2 * wc2.dt * np.mean(noise_real2**2, axis=0)

    noise_real1_spect_white = np.zeros_like(noise_real1_spect)
    noise_real2_spect_white = np.zeros_like(noise_real2_spect)

    noise_real1_spect_white[1:] = noise_real1_spect[1:] / noise1_exp[1:]
    noise_real2_spect_white[1:] = noise_real2_spect[1:] / noise2_exp[1:]

    for itrc in range(noise_time1.shape[1]):
        fpsd1, psd1 = scipy.signal.welch(
            noise_time1[:, itrc], fs=1.0 / wc1.dt, nperseg=2 * wc1.Nf, scaling='density', window='tukey'
        )
        fpsd2, psd2 = scipy.signal.welch(
            noise_time2[:, itrc], fs=1.0 / wc2.dt, nperseg=2 * wc2.Nf, scaling='density', window='tukey'
        )

        _, psd1_white2 = scipy.signal.welch(
            noise_time1_white[:, itrc], fs=1.0 / wc1.dt, nperseg=2 * wc1.Nf, scaling='density', window='tukey'
        )
        _, psd2_white2 = scipy.signal.welch(
            noise_time2_white[:, itrc], fs=1.0 / wc2.dt, nperseg=2 * wc2.Nf, scaling='density', window='tukey'
        )

        fpsd1 = fpsd1[: wc1.Nf]
        fpsd2 = fpsd2[: wc2.Nf]
        assert psd1.size == wc1.Nf + 1
        assert psd2.size == wc2.Nf + 1

        psd1 = psd1[: wc1.Nf]
        psd2 = psd2[: wc2.Nf]

        psd1_white2 = psd1_white2[: wc1.Nf]
        psd2_white2 = psd2_white2[: wc2.Nf]

        psd1_white = psd1.copy()
        psd2_white = psd2.copy()

        psd1_white[1:] = psd1_white[1:] / noise1_exp[1:, itrc]
        psd2_white[1:] = psd2_white[1:] / noise2_exp[1:, itrc]

        assert_allclose(fpsd1, fs1)
        assert_allclose(fpsd2, fs2)

        print(psd1.shape)
        print(psd2.shape)
        print(np.mean(noise_real1_spect) / np.mean(noise_real2_spect))

        # import matplotlib.pyplot as plt
        # if False and itrc == 0:
        #    # plt.loglog(fpsd1[mask1_lim],psd1[mask1_lim])
        #    # plt.loglog(fs1[mask1_lim], noise_real1_spect[mask1_lim, itrc])
        #    # plt.show()

        #    # plt.loglog(psd2[mask2_lim])
        #    # plt.loglog(noise_real2_spect[mask2_lim, itrc])
        #    # plt.show()

        #    plt.plot(psd1_white[mask1_lim])
        #    # plt.plot(noise_real1_spect_white[mask1_lim,0])
        #    plt.plot(psd1_white2[mask1_lim])
        #    plt.show()

        #    plt.plot(psd2_white[mask2_lim])
        #    # plt.plot(noise_real2_spect_white[mask2_lim,0])
        #    plt.plot(psd2_white2[mask2_lim])
        #    plt.show()

        assert_allclose(psd1_white[mask1_lim], psd1_white2[mask1_lim], atol=1.0e-100, rtol=5.0e-2)
        assert_allclose(psd2_white[mask2_lim], psd2_white2[mask2_lim], atol=1.0e-100, rtol=5.0e-2)

        # later bins have less loss of precision
        assert_allclose(
            psd1_white[wc1.Nf // 2:][mask1_lim[wc1.Nf // 2:]],
            psd1_white2[wc1.Nf // 2:][mask1_lim[wc1.Nf // 2:]],
            atol=1.0e-100,
            rtol=5.0e-3,
        )
        if wc2.Nf > 32:
            assert_allclose(
                psd2_white[wc2.Nf // 2:][mask2_lim[wc2.Nf // 2:]],
                psd2_white2[wc2.Nf // 2:][mask2_lim[wc2.Nf // 2:]],
                atol=1.0e-100,
                rtol=5.0e-3,
            )

        print(1.0 / wc1.Tw, 1.0 / wc2.Tw, 1.0 / wc1.DT, 1.0 / wc2.DT, wc1.DF, wc2.DF)

        assert_allclose(psd1[mask1_lim], noise_real1_spect[mask1_lim, itrc], atol=1.0e-100, rtol=1.0e-1)
        assert_allclose(psd2[mask2_lim], noise_real2_spect[mask2_lim, itrc], atol=1.0e-100, rtol=4.0e-1)

        assert_allclose(psd1_white[mask1_lim], noise_real1_spect_white[mask1_lim, itrc], atol=1.0e-100, rtol=1.0e-1)
        assert_allclose(psd2_white[mask2_lim], noise_real2_spect_white[mask2_lim, itrc], atol=1.0e-1, rtol=2.0e-1)

        assert_allclose(
            np.mean(psd1_white[mask1_lim]), np.mean(noise_real1_spect_white[mask1_lim, itrc]), atol=3.0e-3, rtol=3.0e-3
        )
        assert_allclose(
            np.mean(psd2_white[mask2_lim]), np.mean(noise_real2_spect_white[mask2_lim, itrc]), atol=9.0e-3, rtol=9.0e-3
        )

        assert_allclose(
            np.sqrt(3.0 / 2.0) * np.std(psd1_white[mask1_lim]),
            np.std(noise_real1_spect_white[mask1_lim, itrc]),
            atol=1.0e-100,
            rtol=1.0e-1,
        )
        assert_allclose(
            np.sqrt(3.0 / 2.0) * np.std(psd2_white[mask2_lim]),
            np.std(noise_real2_spect_white[mask2_lim, itrc]),
            atol=1.0e-100,
            rtol=4.0e-1,
        )

        assert_allclose(np.mean(psd1[mask1_lim] / noise_real1_spect[mask1_lim, itrc]), 1.0, atol=3.0e-3, rtol=3.0e-3)
        assert_allclose(np.mean(psd2[mask2_lim] / noise_real2_spect[mask2_lim, itrc]), 1.0, atol=9.0e-3, rtol=9.0e-3)

        # variation in the spectra generated from the same noise realization should be correlated
        corr1 = np.corrcoef(psd1_white[mask1_lim], noise_real1_spect_white[mask1_lim, itrc])
        corr2 = np.corrcoef(psd2_white[mask2_lim], noise_real2_spect_white[mask2_lim, itrc])

        corr1_2 = np.corrcoef(psd1_white2[mask1_lim], noise_real1_spect_white[mask1_lim, itrc])
        corr2_2 = np.corrcoef(psd2_white2[mask2_lim], noise_real2_spect_white[mask2_lim, itrc])

        corr1_3 = np.corrcoef(psd1_white[mask1_lim], psd1_white2[mask1_lim])
        corr2_3 = np.corrcoef(psd2_white[mask2_lim], psd2_white2[mask2_lim])

        assert corr1[0, 1] > 0.84
        assert corr2[0, 1] > 0.7

        assert corr1_2[0, 1] > 0.85
        assert corr2_2[0, 1] > 0.75

        assert corr1_3[0, 1] > 0.98
        assert corr2_3[0, 1] > 0.97

        # case where we known the expected answers
        assert_allclose(np.mean(noise_real1_spect_white[mask1_lim, itrc]), 1.0, atol=1.0e-2, rtol=1.0e-2)
        assert_allclose(np.mean(psd1_white[mask1_lim]), 1.0, atol=1.0e-2, rtol=1.0e-2)
        assert_allclose(np.mean(noise_real2_spect_white[mask2_lim, itrc]), 1.0, atol=1.0e-2, rtol=1.0e-2)
        assert_allclose(np.mean(psd2_white[mask2_lim]), 1.0, atol=1.0e-2, rtol=1.0e-2)

        assert_allclose(noise_real1_spect_white[mask1_lim, itrc], 1.0, atol=1.0e-1, rtol=1.0e-1)
        assert_allclose(psd1_white[mask1_lim], 1.0, atol=1.0e-1, rtol=1.0e-1)
        assert_allclose(noise_real2_spect_white[mask2_lim, itrc], 1.0, atol=3.0e-1, rtol=3.0e-1)
        assert_allclose(psd2_white[mask2_lim], 1.0, atol=3.0e-1, rtol=3.0e-1)

        if wc1.Nf == wc2.Nf:
            # can compare the spectra directly, and the cross terms
            mask12_lim = mask1_lim & mask2_lim
            assert_allclose(psd1_white[mask12_lim], psd2_white[mask12_lim], atol=7.0e-2, rtol=1.0e-1)
            assert_allclose(
                noise_real1_spect_white[mask12_lim, itrc],
                noise_real2_spect_white[mask12_lim, itrc],
                atol=7.0e-2,
                rtol=1.0e-1,
            )
            assert_allclose(psd2_white[mask12_lim], noise_real1_spect_white[mask12_lim, itrc], atol=7.0e-2, rtol=1.0e-1)
            assert_allclose(psd1_white[mask12_lim], noise_real2_spect_white[mask12_lim, itrc], atol=7.0e-2, rtol=1.0e-1)

        psd1_2 = InterpolatedUnivariateSpline(fs1, psd1_white, k=1, ext=1)(fs2)
        psd2_1 = InterpolatedUnivariateSpline(fs2, psd2_white, k=1, ext=1)(fs1)
        mask1_2 = InterpolatedUnivariateSpline(fs1, 1.0 * mask1_lim, k=1, ext=1)(fs2)
        mask2_1 = InterpolatedUnivariateSpline(fs2, 1.0 * mask2_lim, k=1, ext=1)(fs1)

        noise_real1_spect_2 = InterpolatedUnivariateSpline(fs1, noise_real1_spect_white[:, itrc], k=1, ext=1)(fs2)
        noise_real2_spect_1 = InterpolatedUnivariateSpline(fs2, noise_real2_spect_white[:, itrc], k=1, ext=1)(fs1)

        mask12_lim_2 = (mask1_2 > 0.99) & mask2_lim
        mask12_lim_1 = (mask2_1 > 0.99) & mask1_lim
        print(mask12_lim_2.size, mask12_lim_2.sum())
        # plt.plot(psd1_2[mask12_lim_2])
        # plt.plot(psd2[mask12_lim_2])
        # plt.plot(noise_real1_spect_2[mask12_lim_2])
        # plt.plot((mask1_2*mask2_lim)[mask12_lim_2])
        # plt.show()

        # plt.plot(psd2_1[mask12_lim_1])
        # plt.plot(psd1[mask12_lim_1])
        # plt.plot(noise_real2_spect_1[mask12_lim_1])
        # plt.plot((mask2_1*mask1_lim)[mask12_lim_1])
        # plt.show()
        print(np.sqrt(wc2.Nf / wc1.Nf))
        # scale tolerances based on how different the binning is
        if time_equal and spectrum_equal:
            # noise should have generated identically in this case, so tolerance is much tighter
            rtol12: float = 1.0e-20 * float(np.max([np.sqrt(wc1.Nt / wc2.Nt), np.sqrt(wc2.Nt / wc1.Nt)]))
            atol12: float = 1.0e-20 * float(np.max([np.sqrt(wc1.Nf / wc2.Nf), np.sqrt(wc2.Nf / wc1.Nf)]))
        else:
            rtol12 = 1.4e-1 * float(np.max([np.sqrt(wc1.Nt / wc2.Nt), np.sqrt(wc2.Nt / wc1.Nt)]))
            atol12 = 1.4e-1 * float(np.max([np.sqrt(wc1.Nf / wc2.Nf), np.sqrt(wc2.Nf / wc1.Nf)]))
        # these terms won't be exactly same unless the noise generated the same
        assert_allclose(psd1_2[mask12_lim_2], psd2_white[mask12_lim_2], atol=atol12, rtol=rtol12)
        assert_allclose(psd2_1[mask12_lim_1], psd1_white[mask12_lim_1], atol=atol12, rtol=rtol12)
        assert_allclose(
            noise_real2_spect_1[mask12_lim_1], noise_real1_spect_white[mask12_lim_1, itrc], atol=atol12, rtol=rtol12
        )
        assert_allclose(
            noise_real1_spect_2[mask12_lim_2], noise_real2_spect_white[mask12_lim_2, itrc], atol=atol12, rtol=rtol12
        )
        assert_allclose(
            np.mean(psd1_2[mask12_lim_2]), np.mean(psd2_white[mask12_lim_2]), atol=6.0e-2 * atol12, rtol=6.0e-2 * rtol12
        )
        assert_allclose(
            np.mean(psd2_1[mask12_lim_1]), np.mean(psd1_white[mask12_lim_1]), atol=6.0e-2 * atol12, rtol=6.0e-2 * rtol12
        )
        assert_allclose(
            np.mean(noise_real2_spect_1[mask12_lim_1]),
            np.mean(noise_real1_spect_white[mask12_lim_1, itrc]),
            atol=6.0e-2 * atol12,
            rtol=6.0e-2 * rtol12,
        )
        assert_allclose(
            np.mean(noise_real1_spect_2[mask12_lim_2]),
            np.mean(noise_real2_spect_white[mask12_lim_2, itrc]),
            atol=6.0e-2 * atol12,
            rtol=6.0e-2 * rtol12,
        )

        # cross terms also have extra tolerance loss from using different methods
        if time_equal and spectrum_equal:
            # tigther tolerance case
            rtol12 = 4.0e-2 * float(np.max([np.sqrt(wc1.Nt / wc2.Nt), np.sqrt(wc2.Nt / wc1.Nt)]))
            atol12 = 4.0e-2 * float(np.max([np.sqrt(wc1.Nf / wc2.Nf), np.sqrt(wc2.Nf / wc1.Nf)]))
        else:
            rtol12 = 1.2e-1 * float(np.max([np.sqrt(wc1.Nt / wc2.Nt), np.sqrt(wc2.Nt / wc1.Nt)]))
            atol12 = 1.2e-1 * float(np.max([np.sqrt(wc1.Nf / wc2.Nf), np.sqrt(wc2.Nf / wc1.Nf)]))
        assert_allclose(noise_real2_spect_1[mask12_lim_1], psd1_white[mask12_lim_1], atol=atol12, rtol=rtol12)
        assert_allclose(noise_real1_spect_2[mask12_lim_2], psd2_white[mask12_lim_2], atol=atol12, rtol=rtol12)

        assert_allclose(
            np.mean(noise_real2_spect_1[mask12_lim_1]),
            np.mean(psd1_white[mask12_lim_1]),
            atol=5.0e-2 * atol12,
            rtol=5.0e-2 * rtol12,
        )
        assert_allclose(
            np.mean(noise_real1_spect_2[mask12_lim_2]),
            np.mean(psd2_white[mask12_lim_2]),
            atol=8.0e-2 * atol12,
            rtol=8.0e-2 * rtol12,
        )


# hi


# scaling on (Nf, Nt, dt, mult) in the second configuration
@pytest.mark.parametrize(
    'channel_mult',
    [
        (0.5, 0.5, 0.5, 1.0),
        (0.5, 0.5, 1.0, 1.0),
        (0.5, 0.5, 2.0, 1.0),
        (0.5, 1.0, 0.5, 1.0),
        (0.5, 1.0, 1.0, 1.0),
        (0.5, 1.0, 2.0, 1.0),
        (0.5, 2.0, 0.5, 1.0),
        (0.5, 2.0, 1.0, 1.0),
        (0.5, 2.0, 2.0, 1.0),
        (1.0, 0.5, 0.5, 1.0),
        (1.0, 0.5, 1.0, 1.0),
        (1.0, 0.5, 2.0, 1.0),
        (1.0, 1.0, 0.5, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 2.0, 1.0),
        (1.0, 2.0, 0.5, 1.0),
        (1.0, 2.0, 1.0, 1.0),
        (1.0, 2.0, 2.0, 1.0),
        (2.0, 0.5, 0.5, 1.0),
        (2.0, 0.5, 1.0, 1.0),
        (2.0, 0.5, 2.0, 1.0),
        (2.0, 1.0, 0.5, 1.0),
        (2.0, 1.0, 1.0, 1.0),
        (2.0, 1.0, 2.0, 1.0),
        (2.0, 2.0, 0.5, 1.0),
        (2.0, 2.0, 1.0, 1.0),
        (2.0, 2.0, 2.0, 1.0),
        (0.5, 0.5, 4.0, 1.0),
        (0.5, 4.0, 0.5, 1.0),
        (4.0, 0.5, 0.5, 1.0),
        (1.0, 3.0, 1.0, 1.0),
        (0.25, 4.0, 1.0, 1.0),
        (0.75, 1.0, 0.75, 1.0),
        (0.5, 8.0, 0.25, 1.0),
        (8.0, 0.125, 1.0, 0.5),
        (4.0, 0.25, 1.0, 0.5),
        (2.0, 0.5, 1.0, 0.5),
        (0.5, 0.5, 0.5, 0.5),
        (0.125, 8.0, 1.0, 2.0),
        (1.0, 1.0, 0.75, 2.0),
        (1.0, 0.5, 1.0, 2.0),
        (2.0, 2.0, 2.0, 2.0),
        (1.0, 1.0, 1.0, 2.0),
        (0.5, 1.0, 1.0, 2.0),
        (0.5, 1.0, 1.0, 4.0),
        (1.0, 1.0, 0.5, 4.0),
        (0.5, 1.0, 0.5, 16.0),
    ],
)
def test_noise_generation_scaling_flat(channel_mult: tuple[float, float, float, float]) -> None:
    """Test the scaling between (Nf, Nt, dt, mult) and SNR^2"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

    config_in1['lisa_constants']['noise_curve_mode'] = 1
    config_in1['wavelet_constants']['Nst'] = 512

    wc1 = get_wavelet_model(config_in1)
    lc1 = get_lisa_constants(config_in1)

    # get the config for the second (Nf, Nt) pair
    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    config_in2['lisa_constants']['noise_curve_mode'] = 1
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

    t_obs1 = wc1.Tobs
    t_obs2 = wc2.Tobs
    time_rat_waveform = t_obs1 / t_obs2

    if wc1.Nf * wc1.Nt == wc2.Nf * wc2.Nt:
        assert channel_mult[0] * channel_mult[1] == 1.0
        n_pix_equal = True
        if channel_mult[2] == 1.0:
            assert t_obs1 == t_obs2
    else:
        n_pix_equal = False
        assert channel_mult[0] * channel_mult[1] != 1.0

    if channel_mult[0] * channel_mult[1] * channel_mult[2] == 1.0:
        time_equal = True
        assert t_obs1 == t_obs2
        assert time_rat_waveform == 1.0
    else:
        time_equal = False
        assert t_obs1 != t_obs2
        assert time_rat_waveform != 1.0

    if channel_mult[0] * channel_mult[2] == 1.0:
        width_equal = True
        assert wc1.DT == wc2.DT
        assert wc1.DF == wc2.DF
        assert wc1.DOM == wc2.DOM
        assert wc1.insDOM == wc2.insDOM
        assert wc1.B == wc2.B
        assert wc1.A == wc2.A
        assert wc1.BW == wc2.BW
        if wc1.Nsf == wc2.Nsf:
            assert wc1.df_bw == wc2.df_bw
    else:
        width_equal = False
        if wc1.Nsf == wc2.Nsf:
            assert wc1.df_bw != wc2.df_bw
        assert wc1.BW != wc2.BW
        assert wc1.A != wc2.A
        assert wc1.B != wc2.B
        assert wc1.insDOM != wc2.insDOM
        assert wc1.DOM != wc2.DOM
        assert wc1.DF != wc2.DF
        assert wc1.DT != wc2.DT

    if channel_mult[2] == 1.0:
        fmax_equal = True
        assert wc1.dt == wc2.dt
        assert wc1.Nf * wc1.DF == wc2.Nf * wc2.DF
        assert wc1.DOM * wc1.Nf == wc2.DOM * wc2.Nf  # OM = OM
    else:
        fmax_equal = False
        assert wc1.dt != wc2.dt
        assert wc1.Nf * wc1.DF != wc2.Nf * wc2.DF
        assert wc1.DOM * wc1.Nf != wc2.DOM * wc2.Nf

    if channel_mult[2] == 1.0 and channel_mult[0] == 1.0:
        spectrum_equal = True
        assert wc1.dt == wc2.dt
        assert wc1.Nf == wc2.Nf
        assert width_equal
        assert fmax_equal
        assert wc1.Nf == wc2.Nf
    else:
        spectrum_equal = False

    if channel_mult[0] * channel_mult[3] == 1.0:
        k_equal = True
        assert wc1.K == wc2.K
    else:
        k_equal = False
        assert wc1.K != wc2.K

    if channel_mult[0] * channel_mult[3] * channel_mult[2] == 1.0:
        tw_equal = True
        assert wc1.Tw == wc2.Tw
        assert wc1.dom == wc2.dom
    else:
        tw_equal = False
        assert wc1.Tw != wc2.Tw
        assert wc1.dom != wc2.dom

    if channel_mult[2] ** 2 * channel_mult[3] * channel_mult[0] ** 2 == 1.0 and wc1.dfdot == wc2.dfdot:
        dfd_equal = True
        assert wc1.dfd == wc2.dfd
    else:
        dfd_equal = False
        assert wc1.dfd != wc2.dfd

    if width_equal and k_equal:
        assert dfd_equal
        assert tw_equal

    noise1 = instrument_noise_AET_wdm_m(lc1, wc1)
    noise2 = instrument_noise_AET_wdm_m(lc2, wc2)

    # test the noise curve itself does not scale with observing time
    if spectrum_equal:
        assert_allclose(noise1[1:], noise2[1:], rtol=1.0e-13, atol=1.0e-40)

    # test all channels same if noise model flat
    if lc1.noise_curve_mode == 1:
        for itrc in range(noise1.shape[1]):
            assert_array_equal(noise1[1:, 0], noise1[1:, itrc])
    if lc2.noise_curve_mode == 1:
        for itrc in range(noise2.shape[1]):
            assert_array_equal(noise2[1:, 0], noise2[1:, itrc])

    seed1 = 31415
    seed2 = 31415

    noise_manager1 = DiagonalStationaryDenseNoiseModel(noise1, wc1, prune=1, nc_snr=noise1.shape[1], seed=seed1)
    noise_manager2 = DiagonalStationaryDenseNoiseModel(noise2, wc2, prune=1, nc_snr=noise2.shape[1], seed=seed2)

    fs1 = np.arange(0, wc1.Nf) * wc1.DF
    fs2 = np.arange(0, wc2.Nf) * wc2.DF

    fmin = max(wc1.DF, wc2.DF)
    fmax = min(wc1.DF * wc1.Nf, wc2.DF * wc2.Nf)

    mask1 = (fs1 >= fmin) & (fs1 <= fmax)
    mask2 = (fs2 >= fmin) & (fs2 <= fmax)

    fmin1 = np.min(fs1[mask1])
    fmax1 = np.max(fs1[mask1])
    frange1 = fmax1 - fmin1

    fmin2 = np.min(fs2[mask2])
    fmax2 = np.max(fs2[mask2])
    frange2 = fmax2 - fmin2

    # getting the integrated power in the overlapping band
    pow1_spect = np.sum(noise_manager1.get_S(), axis=(0, 2))
    pow2_spect = np.sum(noise_manager2.get_S(), axis=(0, 2))

    if spectrum_equal:
        assert_allclose(pow1_spect, time_rat_waveform * pow2_spect, atol=1.0e-40, rtol=1.0e-10)

    pow1 = np.sum(pow1_spect[mask1])
    pow2 = np.sum(pow2_spect[mask2])

    range_rat = frange1 / frange2

    print(frange1, frange2, range_rat)
    print(pow1, pow2)
    print(
        wc1.Nf * wc1.Nt / (wc2.Nf * wc2.Nt),
        time_rat_waveform,
        wc1.dt / wc2.dt,
        wc1.DF / wc2.DF,
        wc1.DT / wc2.DT,
        pow1 / pow2,
        pow1 / pow2 / time_rat_waveform,
    )
    # scale the spectra and check the integrated power matches expectation
    correct_fac = (wc2.dt / wc1.dt) * time_rat_waveform
    assert_allclose(correct_fac, (wc1.Nf * wc1.Nt) / (wc2.Nf * wc2.Nt))
    pow2_rescale = correct_fac * pow2
    if n_pix_equal:
        assert_allclose(pow1, pow2_rescale, atol=1.0e-40, rtol=2.0e-2)
    else:
        assert_allclose(pow1, pow2_rescale, atol=1.0e-40, rtol=4.0e-2)

    # get realizations of the noise and mask the non-overlapping part so we can isolate the band-limited noise
    noise_real1 = noise_manager1.generate_dense_noise()
    noise_real2 = noise_manager2.generate_dense_noise()
    noise_real1[:, ~mask1, :] = 0.0
    noise_real2[:, ~mask2, :] = 0.0

    pow_real1 = np.sum(noise_real1**2)
    pow_real2 = np.sum(noise_real2**2)
    pow_real2_rescale = pow_real2 * correct_fac

    print(pow1, pow_real1, pow2, pow_real2, pow_real2_rescale)
    # check the power in the realizations matches what we expect
    assert_allclose(pow_real1, pow1, atol=1.0e-40, rtol=7.0e-3)
    assert_allclose(pow_real2, pow2, atol=1.0e-40, rtol=7.0e-3)
    assert_allclose(pow_real1 / pow_real2_rescale, pow1 / pow2_rescale, atol=1.0e-40, rtol=7.0e-3)

    # transform realized noise to time domain

    noise_time1 = np.zeros((wc1.Nt * wc1.Nf, noise_real1.shape[-1]))
    noise_time2 = np.zeros((wc2.Nt * wc2.Nf, noise_real2.shape[-1]))

    for itrc in range(noise_real1.shape[-1]):
        noise_time1[:, itrc] = inverse_wavelet_time(noise_real1[:, :, itrc], wc1.Nf, wc1.Nt)
    for itrc in range(noise_real2.shape[-1]):
        noise_time2[:, itrc] = inverse_wavelet_time(noise_real2[:, :, itrc], wc2.Nf, wc2.Nt)

    pow_real1_time = np.sum(noise_time1**2)
    pow_real2_time = np.sum(noise_time2**2)
    pow_real2_time_rescale = pow_real2_time * correct_fac

    # check band limited power in time domain matches (parseval's theorem requires this)
    assert_allclose(pow_real1, pow_real1_time, atol=1.0e-40, rtol=1.0e-8)
    assert_allclose(pow_real2, pow_real2_time, atol=1.0e-40, rtol=1.0e-8)
    assert_allclose(pow_real1 / pow_real2_rescale, pow_real1_time / pow_real2_time_rescale, atol=1.0e-40, rtol=1.0e-8)

    # somewhat narrower range than originally specified to trim edge effects
    mask1_lim = (fs1 >= fmin1 + wc1.DF) & (fs1 <= fmax1 - wc1.DF)
    mask2_lim = (fs2 >= fmin2 + wc2.DF) & (fs2 <= fmax2 - wc2.DF)

    noise_real1_spect = 2 * wc1.dt * np.mean(noise_real1**2, axis=0)
    noise_real2_spect = 2 * wc2.dt * np.mean(noise_real2**2, axis=0)

    for itrc in range(noise_time1.shape[1]):
        fpsd1, psd1 = scipy.signal.welch(
            noise_time1[:, itrc], fs=1.0 / wc1.dt, nperseg=2 * wc1.Nf, scaling='density', window='tukey'
        )
        fpsd2, psd2 = scipy.signal.welch(
            noise_time2[:, itrc], fs=1.0 / wc2.dt, nperseg=2 * wc2.Nf, scaling='density', window='tukey'
        )

        fpsd1_lim = fpsd1[: wc1.Nf]
        assert_allclose(fpsd1_lim, fs1)

        fpsd2_lim = fpsd2[: wc2.Nf]
        assert_allclose(fpsd2_lim, fs2)

        print(psd1.shape)
        print(psd2.shape)
        assert psd1.size == wc1.Nf + 1
        assert psd2.size == wc2.Nf + 1
        print(np.mean(noise_real1_spect) / np.mean(noise_real2_spect))

        # plt.plot(fpsd1[:wc1.Nf][mask1_lim],psd1[:wc1.Nf][mask1_lim])
        # plt.plot(fpsd2,psd2)
        # plt.plot(fs1[mask1_lim], noise_real1_spect[:,0][mask1_lim])
        # plt.plot(fs2, noise_real2_spect[:,0])
        # plt.show()

        # plt.plot(psd2[:wc2.Nf][mask2_lim])
        # plt.plot(noise_real2_spect[:,0][mask2_lim])
        # plt.show()

        assert_allclose(psd1[: wc1.Nf][mask1_lim], noise_real1_spect[mask1_lim, itrc], atol=4.0e-2, rtol=1.0e-1)
        assert_allclose(
            psd2[: wc2.Nf][mask2_lim],
            noise_real2_spect[mask2_lim, itrc],
            atol=5.0e-2 * float(np.sqrt(wc2.Nf / wc1.Nf)),
            rtol=2.0e-1,
        )

        assert_allclose(
            np.mean(psd1[: wc1.Nf][mask1_lim]), np.mean(noise_real1_spect[mask1_lim, itrc]), atol=1.0e-10, rtol=3.0e-3
        )
        assert_allclose(
            np.mean(psd2[: wc2.Nf][mask2_lim]), np.mean(noise_real2_spect[mask2_lim, itrc]), atol=1.0e-10, rtol=3.0e-3
        )

        # multiplier due to different effective smoothing
        assert_allclose(
            np.sqrt(3.0 / 2.0) * np.std(psd1[: wc1.Nf][mask1_lim]),
            np.std(noise_real1_spect[mask1_lim, itrc]),
            atol=5.0e-3,
            rtol=5.0e-3,
        )
        assert_allclose(
            np.sqrt(3.0 / 2.0) * np.std(psd2[: wc2.Nf][mask2_lim]),
            np.std(noise_real2_spect[mask2_lim, itrc]),
            atol=5.0e-3,
            rtol=5.0e-3,
        )

        # variation in the spectra generated from the same noise realization should be correlated
        corr1 = np.corrcoef(psd1[: wc1.Nf][mask1_lim], noise_real1_spect[mask1_lim, itrc])
        corr2 = np.corrcoef(psd2[: wc2.Nf][mask2_lim], noise_real2_spect[mask2_lim, itrc])

        assert corr1[0, 1] > 0.82
        assert corr2[0, 1] > 0.7

        # case where we known the expected answers
        if lc1.noise_curve_mode == 1:
            assert_allclose(np.mean(noise_real1_spect[mask1_lim, itrc]), 1.0, atol=1.0e-2, rtol=1.0e-2)
            assert_allclose(np.mean(psd1[: wc1.Nf][mask1_lim]), 1.0, atol=1.0e-2, rtol=1.0e-2)
        if lc2.noise_curve_mode == 1:
            assert_allclose(np.mean(noise_real2_spect[mask2_lim, itrc]), 1.0, atol=1.0e-2, rtol=1.0e-2)
            assert_allclose(np.mean(psd2[: wc2.Nf][mask2_lim]), 1.0, atol=1.0e-2, rtol=1.0e-2)

        if wc1.Nf == wc2.Nf:
            # can compare the spectra directly, and the cross terms
            mask12_lim = mask1_lim & mask2_lim
            assert_allclose(psd1[: wc1.Nf][mask12_lim], psd2[: wc2.Nf][mask12_lim], atol=7.0e-2, rtol=1.0e-1)
            assert_allclose(
                noise_real1_spect[mask12_lim, itrc], noise_real2_spect[mask12_lim, itrc], atol=7.0e-2, rtol=1.0e-1
            )
            assert_allclose(psd2[: wc2.Nf][mask12_lim], noise_real1_spect[mask12_lim, itrc], atol=7.0e-2, rtol=1.0e-1)
            assert_allclose(psd1[: wc2.Nf][mask12_lim], noise_real2_spect[mask12_lim, itrc], atol=7.0e-2, rtol=1.0e-1)

        psd1_2 = InterpolatedUnivariateSpline(fs1, psd1[: wc1.Nf], k=1, ext=1)(fs2)
        psd2_1 = InterpolatedUnivariateSpline(fs2, psd2[: wc2.Nf], k=1, ext=1)(fs1)
        mask1_2 = InterpolatedUnivariateSpline(fs1, 1.0 * mask1_lim, k=1, ext=1)(fs2)
        mask2_1 = InterpolatedUnivariateSpline(fs2, 1.0 * mask2_lim, k=1, ext=1)(fs1)

        noise_real1_spect_2 = InterpolatedUnivariateSpline(fs1, noise_real1_spect[:, itrc], k=1, ext=1)(fs2)
        noise_real2_spect_1 = InterpolatedUnivariateSpline(fs2, noise_real2_spect[:, itrc], k=1, ext=1)(fs1)

        mask12_lim_2 = (mask1_2 > 0.99) & mask2_lim
        mask12_lim_1 = (mask2_1 > 0.99) & mask1_lim
        print(mask12_lim_2.size, mask12_lim_2.sum())
        # plt.plot(psd1_2[mask12_lim_2])
        # plt.plot(psd2[:wc2.Nf][mask12_lim_2])
        # plt.plot(noise_real1_spect_2[mask12_lim_2])
        # plt.plot((mask1_2*mask2_lim)[mask12_lim_2])
        # plt.show()

        # plt.plot(psd2_1[mask12_lim_1])
        # plt.plot(psd1[:wc1.Nf][mask12_lim_1])
        # plt.plot(noise_real2_spect_1[mask12_lim_1])
        # plt.plot((mask2_1*mask1_lim)[mask12_lim_1])
        # plt.show()
        print(np.sqrt(wc2.Nf / wc1.Nf))
        # scale tolerances based on how different the binning is
        if time_equal and spectrum_equal:
            # noise should have generated identically in this case, so tolerance is much tighter
            rtol12: float = 1.0e-20 * float(np.max([np.sqrt(wc1.Nt / wc2.Nt), np.sqrt(wc2.Nt / wc1.Nt)]))
            atol12: float = 1.0e-20 * float(np.max([np.sqrt(wc1.Nf / wc2.Nf), np.sqrt(wc2.Nf / wc1.Nf)]))
        else:
            rtol12 = 1.4e-1 * float(np.max([np.sqrt(wc1.Nt / wc2.Nt), np.sqrt(wc2.Nt / wc1.Nt)]))
            atol12 = 1.4e-1 * float(np.max([np.sqrt(wc1.Nf / wc2.Nf), np.sqrt(wc2.Nf / wc1.Nf)]))
        # these terms won't be exactly same unless the noise generated the same
        assert_allclose(psd1_2[mask12_lim_2], psd2[: wc2.Nf][mask12_lim_2], atol=atol12, rtol=rtol12)
        assert_allclose(psd2_1[mask12_lim_1], psd1[: wc1.Nf][mask12_lim_1], atol=atol12, rtol=rtol12)
        assert_allclose(
            noise_real2_spect_1[mask12_lim_1], noise_real1_spect[mask12_lim_1, itrc], atol=atol12, rtol=rtol12
        )
        assert_allclose(
            noise_real1_spect_2[mask12_lim_2], noise_real2_spect[mask12_lim_2, itrc], atol=atol12, rtol=rtol12
        )
        assert_allclose(
            np.mean(psd1_2[mask12_lim_2]),
            np.mean(psd2[: wc2.Nf][mask12_lim_2]),
            atol=4.0e-2 * atol12,
            rtol=4.0e-2 * rtol12,
        )
        assert_allclose(
            np.mean(psd2_1[mask12_lim_1]),
            np.mean(psd1[: wc1.Nf][mask12_lim_1]),
            atol=4.0e-2 * atol12,
            rtol=4.0e-2 * rtol12,
        )
        assert_allclose(
            np.mean(noise_real2_spect_1[mask12_lim_1]),
            np.mean(noise_real1_spect[mask12_lim_1, itrc]),
            atol=4.0e-2 * atol12,
            rtol=4.0e-2 * rtol12,
        )
        assert_allclose(
            np.mean(noise_real1_spect_2[mask12_lim_2]),
            np.mean(noise_real2_spect[mask12_lim_2, itrc]),
            atol=4.0e-2 * atol12,
            rtol=4.0e-2 * rtol12,
        )

        # cross terms also have extra tolerance loss from using different methods
        if time_equal and spectrum_equal:
            # tigther tolerance case
            rtol12 = 4.0e-2 * float(np.max([np.sqrt(wc1.Nt / wc2.Nt), np.sqrt(wc2.Nt / wc1.Nt)]))
            atol12 = 4.0e-2 * float(np.max([np.sqrt(wc1.Nf / wc2.Nf), np.sqrt(wc2.Nf / wc1.Nf)]))
        else:
            rtol12 = 1.2e-1 * float(np.max([np.sqrt(wc1.Nt / wc2.Nt), np.sqrt(wc2.Nt / wc1.Nt)]))
            atol12 = 1.2e-1 * float(np.max([np.sqrt(wc1.Nf / wc2.Nf), np.sqrt(wc2.Nf / wc1.Nf)]))
        assert_allclose(noise_real2_spect_1[mask12_lim_1], psd1[: wc1.Nf][mask12_lim_1], atol=atol12, rtol=rtol12)
        assert_allclose(noise_real1_spect_2[mask12_lim_2], psd2[: wc2.Nf][mask12_lim_2], atol=atol12, rtol=rtol12)

        assert_allclose(
            np.mean(noise_real2_spect_1[mask12_lim_1]),
            np.mean(psd1[: wc1.Nf][mask12_lim_1]),
            atol=4.0e-2 * atol12,
            rtol=4.0e-2 * rtol12,
        )
        assert_allclose(
            np.mean(noise_real1_spect_2[mask12_lim_2]),
            np.mean(psd2[: wc2.Nf][mask12_lim_2]),
            atol=4.0e-2 * atol12,
            rtol=4.0e-2 * rtol12,
        )


# scaling on (Nf, Nt, dt, mult) in the second configuration
@pytest.mark.parametrize(
    'channel_mult',
    [
        (1.0, 1.0, 1.0, 1.0),
        (2.0, 0.5, 1.0, 1.0),
        (0.5, 2.0, 1.0, 1.0),
        (1.0, 0.5, 2.0, 1.0),
        (2.0, 1.0, 0.5, 1.0),
        (0.5, 1.0, 2.0, 1.0),
        (0.5, 0.5, 4.0, 1.0),
        (0.5, 0.5, 1.0, 1.0),
        (1.0, 1.0, 0.5, 1.0),
        (2.0, 0.5, 1.0, 0.5),
        (8.0, 0.125, 1.0, 0.5),
        (4.0, 0.25, 1.0, 0.5),
        (0.125, 8.0, 1.0, 2.0),
        (0.25, 4.0, 1.0, 1.0),
        (0.5, 1.0, 1.0, 4.0),
        (1.0, 1.0, 0.5, 4.0),
        (1.0, 1.0, 0.5, 2.0),
        (1.0, 0.5, 1.0, 2.0),
        (2.0, 2.0, 2.0, 2.0),
        (0.5, 1.0, 0.5, 16.0),
        (0.75, 1.0, 0.75, 1.0),
        (0.5, 0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0, 2.0),
        (0.5, 1.0, 1.0, 2.0),
        (0.5, 1.0, 0.5, 1.0),
        (0.5, 1.0, 1.0, 1.0),
        (0.5, 8.0, 0.25, 1.0),
        (0.5, 4.0, 0.5, 1.0),
        (0.5, 0.5, 0.5, 1.0),
        (1.0, 2.0, 1.0, 1.0),
        (1.0, 0.5, 1.0, 1.0),
    ],
)
def test_noise_snr_scaling(channel_mult: tuple[float, float, float, float]) -> None:
    """Test the scaling between (Nf, Nt, dt, mult) and SNR^2"""
    toml_filename_in = 'tests/wavemaket_test_config1.toml'

    noise_curve_mode = 0
    if noise_curve_mode == 0:
        amp_use = 1.0e-27
        response_mode = 0
    elif noise_curve_mode == 1:
        amp_use = 1.0
        response_mode = 2
    else:
        msg = 'Unrecognized option for noise curve mode'
        raise ValueError(msg)

    with Path(toml_filename_in).open('rb') as f:
        config_in1 = tomllib.load(f)

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

    t_obs1 = wc1.Tobs
    t_obs2 = wc2.Tobs
    time_rat_waveform = t_obs1 / t_obs2

    if wc1.Nf * wc1.Nt == wc2.Nf * wc2.Nt:
        assert channel_mult[0] * channel_mult[1] == 1.0
        n_pix_equal = True
        if channel_mult[2] == 1.0:
            assert t_obs1 == t_obs2
    else:
        n_pix_equal = False
        assert channel_mult[0] * channel_mult[1] != 1.0

    if channel_mult[0] * channel_mult[1] * channel_mult[2] == 1.0:
        time_equal = True
        if wc1.dt == wc2.dt:
            assert n_pix_equal
        else:
            assert not n_pix_equal
        assert t_obs1 == t_obs2
        assert time_rat_waveform == 1.0
    else:
        time_equal = False
        assert t_obs1 != t_obs2
        assert time_rat_waveform != 1.0

    if channel_mult[0] * channel_mult[2] == 1.0:
        width_equal = True
        assert wc1.DT == wc2.DT
        assert wc1.DF == wc2.DF
        assert wc1.DOM == wc2.DOM
        assert wc1.insDOM == wc2.insDOM
        assert wc1.B == wc2.B
        assert wc1.A == wc2.A
        assert wc1.BW == wc2.BW
        if wc1.Nsf == wc2.Nsf:
            assert wc1.df_bw == wc2.df_bw
    else:
        width_equal = False
        if wc1.Nsf == wc2.Nsf:
            assert wc1.df_bw != wc2.df_bw
        assert wc1.BW != wc2.BW
        assert wc1.A != wc2.A
        assert wc1.B != wc2.B
        assert wc1.insDOM != wc2.insDOM
        assert wc1.DOM != wc2.DOM
        assert wc1.DF != wc2.DF
        assert wc1.DT != wc2.DT

    if channel_mult[2] == 1.0:
        fmax_equal = True
        assert wc1.dt == wc2.dt
        assert wc1.Nf * wc1.DF == wc2.Nf * wc2.DF
        assert wc1.DOM * wc1.Nf == wc2.DOM * wc2.Nf  # OM = OM
    else:
        fmax_equal = False
        assert wc1.dt != wc2.dt
        assert wc1.Nf * wc1.DF != wc2.Nf * wc2.DF
        assert wc1.DOM * wc1.Nf != wc2.DOM * wc2.Nf

    if channel_mult[2] == 1.0 and channel_mult[0] == 1.0:
        spectrum_equal = True
        assert wc1.dt == wc2.dt
        assert wc1.Nf == wc2.Nf
        assert width_equal
        assert fmax_equal
        assert wc1.Nf == wc2.Nf
    else:
        spectrum_equal = False

    if channel_mult[0] * channel_mult[3] == 1.0:
        k_equal = True
        assert wc1.K == wc2.K
    else:
        k_equal = False
        assert wc1.K != wc2.K

    if channel_mult[0] * channel_mult[3] * channel_mult[2] == 1.0:
        tw_equal = True
        assert wc1.Tw == wc2.Tw
        assert wc1.dom == wc2.dom
    else:
        tw_equal = False
        assert wc1.Tw != wc2.Tw
        assert wc1.dom != wc2.dom

    if channel_mult[2] ** 2 * channel_mult[3] * channel_mult[0] ** 2 == 1.0 and wc1.dfdot == wc2.dfdot:
        dfd_equal = True
        assert wc1.dfd == wc2.dfd
    else:
        dfd_equal = False
        assert wc1.dfd != wc2.dfd

    if width_equal and k_equal:
        assert dfd_equal
        assert tw_equal

    noise1 = instrument_noise_AET_wdm_m(lc1, wc1)
    noise2 = instrument_noise_AET_wdm_m(lc2, wc2)

    # test the noise curve itself does not scale with observing time
    if spectrum_equal:
        assert_allclose(noise1[1:], noise2[1:], rtol=1.0e-13, atol=1.0e-40)

    # test all channels same if noise model flat
    if lc1.noise_curve_mode == 1:
        for itrc in range(noise1.shape[1]):
            assert_array_equal(noise1[1:, 0], noise1[1:, itrc])
    if lc2.noise_curve_mode == 1:
        for itrc in range(noise2.shape[1]):
            assert_array_equal(noise2[1:, 0], noise2[1:, itrc])

    seed1 = 31415
    seed2 = 31415

    noise_manager1 = DiagonalStationaryDenseNoiseModel(noise1, wc1, prune=1, nc_snr=noise1.shape[1], seed=seed1)
    noise_manager2 = DiagonalStationaryDenseNoiseModel(noise2, wc2, prune=1, nc_snr=noise2.shape[1], seed=seed2)

    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=amp_use,  # amplitude
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
    # TODO need this check in iterative fit
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

    t_obs_snr1 = (nt_lim_snr1.nx_max - nt_lim_snr1.nx_min) * nt_lim_snr1.dx
    t_obs_snr2 = (nt_lim_snr2.nx_max - nt_lim_snr2.nx_min) * nt_lim_snr2.dx

    t_rat_snr = t_obs_snr1 / t_obs_snr2

    waveform1 = LinearFrequencyWaveletWaveformTime(
        params,
        wc1,
        lc1,
        nt_lim_waveform1,
        table_cache_mode='check',
        table_output_mode='hf',
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

    snrs1 = noise_manager1.get_sparse_snrs(wavelet_waveform1, nt_lim_snr1)
    snrs2 = noise_manager2.get_sparse_snrs(wavelet_waveform2, nt_lim_snr2)

    snr_tot1 = np.linalg.norm(snrs1)
    snr_tot2 = np.linalg.norm(snrs2)
    print(snrs1, snr_tot1)
    print(snrs2, snr_tot2)
    print(t_rat_snr, time_rat_waveform)
    print(
        (wc1.DF * wc1.Nf) / intrinsic.F0, (wc2.DF * wc2.Nf) / intrinsic.F0, intrinsic.F0 / wc1.DF, intrinsic.F0 / wc2.DF
    )

    if waveform1.source_waveform.response_mode == response_mode and lc1.noise_curve_mode == 1:
        assert_array_equal(snrs1[0], snrs1)
    if waveform2.source_waveform.response_mode == response_mode and lc2.noise_curve_mode == 1:
        assert_array_equal(snrs2[0], snrs2)

    # snr^2 should linearly scale with observing time
    if spectrum_equal and time_equal and tw_equal:
        # precision case
        assert_allclose(snr_tot1**2, snr_tot2**2 * t_rat_snr, atol=1.0e-30, rtol=1.0e-14)
        assert_allclose(snrs1**2, snrs2**2 * t_rat_snr, atol=1.0e-30, rtol=1.0e-14)
    elif spectrum_equal:
        # somewhat less precise case
        assert_allclose(snr_tot1**2, snr_tot2**2 * t_rat_snr, atol=1.0e-11, rtol=2.0e-3)
        assert_allclose(snrs1**2, snrs2**2 * t_rat_snr, atol=1.0e-11, rtol=2.0e-3)
    elif width_equal:
        # somewhat less precise case
        assert_allclose(snr_tot1**2, snr_tot2**2 * t_rat_snr, atol=1.0e-30, rtol=5.0e-2)
        assert_allclose(snrs1**2, snrs2**2 * t_rat_snr, atol=1.0e-30, rtol=5.0e-2)
    else:
        assert_allclose(snr_tot1**2, snr_tot2**2 * t_rat_snr, atol=1.0e-9, rtol=5.0e-2)
        assert_allclose(snrs1**2, snrs2**2 * t_rat_snr, atol=1.0e-9, rtol=5.0e-2)

    # import matplotlib.pyplot as plt
    # plt.plot(fs1[1:], noise1[1:,0])
    # plt.plot(fs2[1:], noise2[1:,0])
    # plt.show()
