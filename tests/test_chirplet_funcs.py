"""test the Chirp_WDM functions"""

from pathlib import Path

import numpy as np
import pytest
import tomllib
import WDMWaveletTransforms.fft_funcs as fft
from numpy.testing import assert_allclose
from WDMWaveletTransforms.transform_freq_funcs import tukey
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq, transform_wavelet_time

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
