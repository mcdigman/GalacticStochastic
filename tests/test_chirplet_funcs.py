"""test the Chirp_WDM functions"""

from pathlib import Path

import numpy as np
import pytest
import tomllib
import WDMWaveletTransforms.fft_funcs as fft
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from WDMWaveletTransforms.transform_freq_funcs import tukey
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq, transform_wavelet_time

from LisaWaveformTools.chirplet_source_time import LinearChirpletSourceWaveformTime
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.stationary_source_waveform import ExtrinsicParams, SourceParams
from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wdm_config import get_wavelet_model

# TODO none of these functions actually call the chirplet functions


@pytest.mark.parametrize('m', [1])
@pytest.mark.parametrize('use_tukey', [True, False])
@pytest.mark.parametrize('use_cos', [True, False])
def test_sincos_low_wdm_match(m, use_tukey, use_cos):
    """Test for match for known pure sinusoidal signal at low frequency"""
    toml_filename_in = 'tests/sparse_wdm_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)

    ND = wc.Nf * wc.Nt

    ts = np.arange(0, ND) * wc.dt

    if use_cos:
        hs = np.cos(2 * np.pi * m * ts / (2 * wc.DT))
    else:
        hs = np.sin(2 * np.pi * m * ts / (2 * wc.DT))

    if use_tukey:
        alpha = 8 * (2.0 * (4.0 * wc.DT) / wc.Tobs)
        tukey(hs, alpha, ND)

    hs_freq = fft.rfft(hs)

    wave_got_time = transform_wavelet_time(hs, wc.Nf, wc.Nt)
    wave_got_freq = transform_wavelet_freq(hs_freq, wc.Nf, wc.Nt)
    print(wc.dt, wc.Nt, wc.Nf, wave_got_time[0, m], wave_got_freq[0, m], np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2)

    matchsigf_sigt = 1 - np.sum(wave_got_freq * wave_got_time) / np.sqrt(np.sum(wave_got_freq**2) * np.sum(wave_got_time**2))

    assert matchsigf_sigt < 1.e-7
    assert_allclose(wave_got_time, wave_got_freq, atol=1.e-4, rtol=1.e-4)

    if not use_tukey:
        assert_allclose(wave_got_time[:, 0:m], 0., atol=1.e-8)
        assert_allclose(wave_got_time[:, m + 1:], 0., atol=1.e-8)
        assert_allclose(wave_got_freq[:, 0:m], 0., atol=1.e-8)
        assert_allclose(wave_got_freq[:, m + 1:], 0., atol=1.e-8)

        alt1 = not (m % 2 == 1) ^ use_cos  # or (not (m%2==1) and not use_cos)
        if alt1:
            assert_allclose(wave_got_time[::2, m], 0., atol=1.e-8)
            assert_allclose(wave_got_time[1::2, m], wave_got_time[wc.Nt // 2 + 1, m], atol=1.e-8, rtol=1.e-8)
            assert_allclose(wave_got_freq[::2, m], 0., atol=1.e-8)
            assert_allclose(wave_got_freq[1::2, m], wave_got_freq[wc.Nt // 2 + 1, m], atol=1.e-8, rtol=1.e-8)
            sign = (-1)**(use_cos)
            assert_allclose(wave_got_time[1::2, m], sign * np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2, atol=1.e-8, rtol=1.e-8)
            assert_allclose(wave_got_freq[1::2, m], sign * np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2, atol=1.e-8, rtol=1.e-8)
        else:
            assert_allclose(wave_got_time[1::2, m], 0., atol=1.e-8)
            assert_allclose(wave_got_time[::2, m], wave_got_time[wc.Nt // 2, m], atol=1.e-8, rtol=1.e-8)
            assert_allclose(wave_got_freq[1::2, m], 0., atol=1.e-8)
            assert_allclose(wave_got_freq[::2, m], wave_got_freq[wc.Nt // 2, m], atol=1.e8, rtol=1.e-8)
            assert_allclose(wave_got_time[::2, m], np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2, atol=1.e-8, rtol=1.e-8)
            assert_allclose(wave_got_freq[::2, m], np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2, atol=1.e-8, rtol=1.e-8)


@pytest.mark.parametrize('dt_loc', [7.5, 15., 30.])
@pytest.mark.parametrize('Nt_loc', [256, 512])
@pytest.mark.parametrize('Nf_loc', [1024, 2048])
@pytest.mark.parametrize('m', [7, 8])
@pytest.mark.parametrize('use_cos', [True, False])
def test_sincos_wdm_match(dt_loc, Nt_loc, Nf_loc, m, use_cos):
    """Test for known sinusoidal signal match"""
    toml_filename_in = 'tests/sparse_wdm_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    config_in['wavelet_constants']['Nf'] = int(Nf_loc)
    config_in['wavelet_constants']['Nt'] = int(Nt_loc)
    config_in['wavelet_constants']['dt'] = float(dt_loc)

    wc = get_wavelet_model(config_in)

    ND = wc.Nf * wc.Nt
    ts = np.arange(0, ND) * wc.dt

    if use_cos:
        hs = np.cos(2 * np.pi * m * ts / (2 * wc.DT))
    else:
        hs = np.sin(2 * np.pi * m * ts / (2 * wc.DT))

    hs_freq = fft.rfft(hs)

    wave_got_time = transform_wavelet_time(hs, wc.Nf, wc.Nt)
    wave_got_freq = transform_wavelet_freq(hs_freq, wc.Nf, wc.Nt)
    print(wc.dt, wc.Nt, wc.Nf, wave_got_time[0, m], wave_got_freq[0, m], np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2)

    matchsigf_sigt = 1 - np.sum(wave_got_freq * wave_got_time) / np.sqrt(np.sum(wave_got_freq**2) * np.sum(wave_got_time**2))

    assert matchsigf_sigt < 1.e-7
    assert_allclose(wave_got_time, wave_got_freq, atol=1.e-8, rtol=1.e-8)
    assert_allclose(wave_got_time[:, 0:m], 0., atol=1.e-8)
    assert_allclose(wave_got_time[:, m + 1:], 0., atol=1.e-8)
    assert_allclose(wave_got_freq[:, 0:m], 0., atol=1.e-8)
    assert_allclose(wave_got_freq[:, m + 1:], 0., atol=1.e-8)

    alt1 = not (m % 2 == 1) ^ use_cos  # or (not (m%2==1) and not use_cos)
    if alt1:
        assert_allclose(wave_got_time[::2, m], 0., atol=1.e-8)
        assert_allclose(wave_got_time[1::2, m], wave_got_time[wc.Nt // 2 + 1, m], atol=1.e-8, rtol=1.e-8)
        assert_allclose(wave_got_freq[::2, m], 0., atol=1.e-8)
        assert_allclose(wave_got_freq[1::2, m], wave_got_freq[wc.Nt // 2 + 1, m], atol=1.e-8, rtol=1.e-8)
        sign = (-1)**(use_cos)
        assert_allclose(wave_got_time[1::2, m], sign * np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2, atol=1.e-8, rtol=1.e-8)
        assert_allclose(wave_got_freq[1::2, m], sign * np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2, atol=1.e-8, rtol=1.e-8)
    else:
        assert_allclose(wave_got_time[1::2, m], 0., atol=1.e-8)
        assert_allclose(wave_got_time[::2, m], wave_got_time[wc.Nt // 2, m], atol=1.e-8, rtol=1.e-8)
        assert_allclose(wave_got_freq[1::2, m], 0., atol=1.e-8)
        assert_allclose(wave_got_freq[::2, m], wave_got_freq[wc.Nt // 2, m], atol=1.e8, rtol=1.e-8)
        assert_allclose(wave_got_time[::2, m], np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2, atol=1.e-8, rtol=1.e-8)
        assert_allclose(wave_got_freq[::2, m], np.sum(hs**2) / (np.sqrt(wc.Nf) * wc.Nt) * 2, atol=1.e-8, rtol=1.e-8)


def test_intrinsic_update_consistent1_time():
    """Test the non TDI parts of the waveform update consistently."""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    nt_lim_waveform = PixelTimeRange(0, wc.Nt)

    fdot_mult1 = 0.1
    fdot1 = 3.105 * fdot_mult1 * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples

    fdot_mult2 = 0.15
    fdot2 = 3.105 * fdot_mult2 * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples

    f0 = 16 * wc.DF

    tp1 = wc.Tobs / 2.
    tp2 = 1.05 * wc.Tobs / 2.

    fp1 = f0 + fdot1 * tp1  # ensures that frequency starts positive
    fp2 = f0 + fdot2 * tp1  # ensures that frequency starts positive

    gamma1 = fp1 / 8.
    tau1 = gamma1 / fdot1

    gamma2 = fp2 / 8.
    tau2 = gamma2 / fdot2

    # Setup the intrinsic parameters for the binary source
    intrinsic1 = LinearChirpletIntrinsicParams(10000., 0.1, fp1, tp1, tau1, gamma1)
    intrinsic2 = LinearChirpletIntrinsicParams(15000., 0.2, fp2, tp2, tau2, gamma2)

    extrinsic1 = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)  # Replace this with a real extrinsic param object if needed
    extrinsic2 = ExtrinsicParams(costh=0.2, phi=0.3, cosi=0.1, psi=0.2)  # Replace this with a real extrinsic param object if needed

    # Bundle parameters
    params1 = SourceParams(
        intrinsic=intrinsic1,
        extrinsic=extrinsic1,
    )

    params2 = SourceParams(
        intrinsic=intrinsic2,
        extrinsic=extrinsic2,
    )

    lf_waveform_time11 = LinearChirpletSourceWaveformTime(
        params=params1,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
        wc=wc,
    )

    lf_waveform_time22 = LinearChirpletSourceWaveformTime(
        params=params2,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
        wc=wc,
    )

    assert not np.all(lf_waveform_time11.wavefront_time == lf_waveform_time22.wavefront_time)
    assert not np.all(lf_waveform_time11.intrinsic_waveform.AT == lf_waveform_time22.intrinsic_waveform.AT)
    assert not np.all(lf_waveform_time11.intrinsic_waveform.PT == lf_waveform_time22.intrinsic_waveform.PT)
    assert not np.all(lf_waveform_time11.intrinsic_waveform.FT == lf_waveform_time22.intrinsic_waveform.FT)
    assert not np.all(lf_waveform_time11.intrinsic_waveform.FTd == lf_waveform_time22.intrinsic_waveform.FTd)
    assert not np.all(lf_waveform_time11.tdi_waveform.AT == lf_waveform_time22.tdi_waveform.AT)
    assert not np.all(lf_waveform_time11.tdi_waveform.PT == lf_waveform_time22.tdi_waveform.PT)
    assert not np.all(lf_waveform_time11.tdi_waveform.FT == lf_waveform_time22.tdi_waveform.FT)
    assert not np.all(lf_waveform_time11.tdi_waveform.FTd == lf_waveform_time22.tdi_waveform.FTd)

    assert_equal(lf_waveform_time11.nc_waveform, lf_waveform_time22.nc_waveform)
    assert_equal(lf_waveform_time11.intrinsic_waveform.T, lf_waveform_time22.intrinsic_waveform.T)
    assert_equal(lf_waveform_time11.tdi_waveform.T, lf_waveform_time22.tdi_waveform.T)

    lf_waveform_time12 = LinearChirpletSourceWaveformTime(
        params=params1,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
        wc=wc,
    )
    lf_waveform_time21 = LinearChirpletSourceWaveformTime(
        params=params2,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
        wc=wc,
    )

    def assert_match_time(wave1, wave2):
        assert_array_equal(wave1.wavefront_time, wave2.wavefront_time)
        assert_equal(wave1.nc_waveform, wave2.nc_waveform)
        assert_equal(wave1.intrinsic_waveform.T, wave2.intrinsic_waveform.T)
        assert_array_equal(wave1.intrinsic_waveform.AT, wave2.intrinsic_waveform.AT)
        assert_array_equal(wave1.intrinsic_waveform.PT, wave2.intrinsic_waveform.PT)
        assert_array_equal(wave1.intrinsic_waveform.FT, wave2.intrinsic_waveform.FT)
        assert_array_equal(wave1.intrinsic_waveform.FTd, wave2.intrinsic_waveform.FTd)
        assert_equal(wave1.tdi_waveform.T, wave2.tdi_waveform.T)
        assert_array_equal(wave1.tdi_waveform.AT, wave2.tdi_waveform.AT)
        assert_array_equal(wave1.tdi_waveform.PT, wave2.tdi_waveform.PT)
        assert_array_equal(wave1.tdi_waveform.FT, wave2.tdi_waveform.FT)
        assert_array_equal(wave1.tdi_waveform.FTd, wave2.tdi_waveform.FTd)

    assert_match_time(lf_waveform_time11, lf_waveform_time12)
    assert_match_time(lf_waveform_time22, lf_waveform_time21)

    # check idempotence from start

    lf_waveform_time12.update_params(params1)
    lf_waveform_time21.update_params(params2)

    assert_match_time(lf_waveform_time11, lf_waveform_time12)
    assert_match_time(lf_waveform_time22, lf_waveform_time21)

    # check swap parameters

    lf_waveform_time12.update_params(params2)
    lf_waveform_time21.update_params(params1)

    assert_match_time(lf_waveform_time22, lf_waveform_time12)
    assert_match_time(lf_waveform_time11, lf_waveform_time21)

    # check idempotence from swap

    lf_waveform_time12.update_params(params2)
    lf_waveform_time21.update_params(params1)

    assert_match_time(lf_waveform_time22, lf_waveform_time12)
    assert_match_time(lf_waveform_time11, lf_waveform_time21)

    # check return to start
    lf_waveform_time12.update_params(params1)
    lf_waveform_time21.update_params(params2)

    assert_match_time(lf_waveform_time11, lf_waveform_time12)
    assert_match_time(lf_waveform_time22, lf_waveform_time21)

    # check idempotence from return to start
    lf_waveform_time12.update_params(params1)
    lf_waveform_time21.update_params(params2)

    assert_match_time(lf_waveform_time11, lf_waveform_time12)
    assert_match_time(lf_waveform_time22, lf_waveform_time21)
