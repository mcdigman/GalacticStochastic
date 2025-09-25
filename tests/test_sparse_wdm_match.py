"""test the Chirp_WDM functions"""

from pathlib import Path

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq, transform_wavelet_time

from LisaWaveformTools.chirplet_source_time import LinearChirpletWaveletSparseTime, LinearChirpletWaveletTaylorTime
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.stationary_source_waveform import ExtrinsicParams, SourceParams, StationaryWaveformFreq, StationaryWaveformTime
from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams, chirplet_freq_intrinsic, chirplet_time_intrinsic
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, wavelet_sparse_to_dense
from WaveletWaveforms.wdm_config import get_wavelet_model


@pytest.mark.parametrize('fdot_mult', [0.8])
def test_ts_match_TT_TS(fdot_mult):
    """Test for match between different methods of getting wavelet transform"""
    toml_filename_in = 'tests/sparse_wdm_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

    # Want gamma*tau large so that the SPA is accurate
    # Pick fdot so that both the Taylor expanded time and frequency domain transforms are valid
    # => fdot < 8 DF/T_w and fdot > DF^2/8 = DF/(16 DT)
    # Also need to ensure that the sparse time domain transform is valid
    # => fdot > DF/T_w
    # fdot = gamma/tau

    fdot = 3.105 * fdot_mult * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples
    print(fdot, 8 * wc.DF / wc.Tw, wc.DF**2 / 8)
    assert fdot < 8 * wc.DF / wc.Tw
    # assert fdot > _wc.DF**2/8
    assert fdot > wc.DF / wc.Tw
    assert -wc.dfd * wc.Nfd_negative < fdot
    assert fdot < wc.dfd * (wc.Nfd - wc.Nfd_negative)
    tp = wc.Tobs / 2.
    f0 = 16 * wc.DF
    fp = f0 + fdot * tp  # ensures that frequency starts positive

    print('%f %f' % (fp / wc.DF, tp / wc.DT))
    gamma = fp / 8.
    tau = gamma / fdot
    Phi0 = 0.
    A0 = 10000.

    intrinsic = LinearChirpletIntrinsicParams(A0, Phi0, fp, tp, tau, gamma)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2,
                                psi=0.3)  # Replace this with a real extrinsic param object if needed

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    print('computing time domain waveforms')

    # Time domain Taylor expansion transform
    waveletTT = LinearChirpletWaveletTaylorTime(params, wc, lc, nt_lim_waveform, wavelet_mode=1, response_mode=2)

    waveTT = wavelet_sparse_to_dense(waveletTT.wavelet_waveform, wc)[:, :, 0]

    # Time domain sparse transform
    waveletTS = LinearChirpletWaveletSparseTime(params, wc, lc, nt_lim_waveform, response_mode=2)
    waveTS = wavelet_sparse_to_dense(waveletTS.wavelet_waveform, wc)[:, :, 0]

    maskTS = waveTS != 0.
    maskTT = waveTT != 0.
    maskTTTS = maskTS & maskTT

    # import matplotlib.pyplot as plt
    print(waveTT.shape)
    print(waveTS.shape)

    assert_allclose(waveTT[maskTTTS], waveTS[maskTTTS], atol=1.e-2, rtol=1.e-10)

    pow_TT = np.sum(waveTT**2)
    pow_TS = np.sum(waveTS**2)
    # check powers match

    assert_allclose(pow_TT, pow_TS, atol=1.e-11, rtol=7.e-4)

    waveTT_res = waveTT  # waveTT.reshape(_wc.Nf,_wc.Nt).T
    waveTS_res = waveTS  # waveTS.reshape(_wc.Nf,_wc.Nt).T

    matchTTTS = 1 - np.sum(waveTT_res * waveTS_res) / np.sqrt(np.sum(waveTT_res**2) * np.sum(waveTS_res**2))

    assert matchTTTS < 1.e-4


@pytest.mark.parametrize('fdot_mult', [0.8])
def test_ts_match_TTexact_TS(fdot_mult):
    """Test for match between different methods of getting wavelet transform"""
    toml_filename_in = 'tests/sparse_wdm_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

    # Want gamma*tau large so that the SPA is accurate
    # Pick fdot so that both the Taylor expanded time and frequency domain transforms are valid
    # => fdot < 8 DF/T_w and fdot > DF^2/8 = DF/(16 DT)
    # Also need to ensure that the sparse time domain transform is valid
    # => fdot > DF/T_w
    # fdot = gamma/tau

    fdot = 3.105 * fdot_mult * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples
    print(fdot, 8 * wc.DF / wc.Tw, wc.DF**2 / 8)
    assert fdot < 8 * wc.DF / wc.Tw
    # assert fdot > _wc.DF**2/8
    assert fdot > wc.DF / wc.Tw
    assert -wc.dfd * wc.Nfd_negative < fdot
    assert fdot < wc.dfd * (wc.Nfd - wc.Nfd_negative)
    tp = wc.Tobs / 2.
    f0 = 16 * wc.DF
    fp = f0 + fdot * tp  # ensures that frequency starts positive

    print('%f %f' % (fp / wc.DF, tp / wc.DT))
    gamma = fp / 8.
    tau = gamma / fdot
    Phi0 = 0.
    A0 = 10000.

    intrinsic = LinearChirpletIntrinsicParams(A0, Phi0, fp, tp, tau, gamma)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2,
                                psi=0.3)  # Replace this with a real extrinsic param object if needed

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    print('computing time domain waveforms')

    # Time domain Taylor expansion transform
    waveletTT = LinearChirpletWaveletTaylorTime(params, wc, lc, nt_lim_waveform, wavelet_mode=0, response_mode=2)

    waveTT = wavelet_sparse_to_dense(waveletTT.wavelet_waveform, wc)[:, :, 0]

    # Time domain sparse transform
    waveletTS = LinearChirpletWaveletSparseTime(params, wc, lc, nt_lim_waveform, response_mode=2)
    waveTS = wavelet_sparse_to_dense(waveletTS.wavelet_waveform, wc)[:, :, 0]

    maskTS = waveTS != 0.
    maskTT = waveTT != 0.
    maskTTTS = maskTS & maskTT

    # import matplotlib.pyplot as plt
    print(waveTT.shape)
    print(waveTS.shape)

    assert_allclose(waveTT[maskTTTS], waveTS[maskTTTS], atol=1.e-2, rtol=1.e-10)

    pow_TT = np.sum(waveTT**2)
    pow_TS = np.sum(waveTS**2)
    # check powers match

    assert_allclose(pow_TT, pow_TS, atol=1.e-11, rtol=7.e-4)

    waveTT_res = waveTT  # waveTT.reshape(_wc.Nf,_wc.Nt).T
    waveTS_res = waveTS  # waveTS.reshape(_wc.Nf,_wc.Nt).T

    matchTTTS = 1 - np.sum(waveTT_res * waveTS_res) / np.sqrt(np.sum(waveTT_res**2) * np.sum(waveTS_res**2))

    assert matchTTTS < 1.e-4


@pytest.mark.parametrize('fdot_mult', [0.8])
def test_Chirp_wdm_match_TT_TTexact(fdot_mult):
    """Test for match between different methods of getting wavelet transform"""
    toml_filename_in = 'tests/sparse_wdm_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

    # Want gamma*tau large so that the SPA is accurate
    # Pick fdot so that both the Taylor expanded time and frequency domain transforms are valid
    # => fdot < 8 DF/T_w and fdot > DF^2/8 = DF/(16 DT)
    # Also need to ensure that the sparse time domain transform is valid
    # => fdot > DF/T_w
    # fdot = gamma/tau

    fdot = 3.105 * fdot_mult * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples
    print(fdot, 8 * wc.DF / wc.Tw, wc.DF**2 / 8)
    assert fdot < 8 * wc.DF / wc.Tw
    # assert fdot > _wc.DF**2/8
    assert fdot > wc.DF / wc.Tw
    assert -wc.dfd * wc.Nfd_negative < fdot
    assert fdot < wc.dfd * (wc.Nfd - wc.Nfd_negative)
    tp = wc.Tobs / 2.
    f0 = 16 * wc.DF
    fp = f0 + fdot * tp  # ensures that frequency starts positive

    print('%f %f' % (fp / wc.DF, tp / wc.DT))
    gamma = fp / 8.
    tau = gamma / fdot
    Phi0 = 0.
    A0 = 10000.

    intrinsic = LinearChirpletIntrinsicParams(A0, Phi0, fp, tp, tau, gamma)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2,
                                psi=0.3)  # Replace this with a real extrinsic param object if needed

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    # Time domain Taylor expansion transform
    waveletTT = LinearChirpletWaveletTaylorTime(params, wc, lc, nt_lim_waveform, wavelet_mode=1, response_mode=2)
    waveletTT_exact = LinearChirpletWaveletTaylorTime(params, wc, lc, nt_lim_waveform, wavelet_mode=0, response_mode=2)

    waveTT = wavelet_sparse_to_dense(waveletTT.wavelet_waveform, wc)[:, :, 0]
    waveTT_exact = wavelet_sparse_to_dense(waveletTT_exact.wavelet_waveform, wc)[:, :, 0]

    assert_allclose(waveTT, waveTT_exact, atol=3.e-4, rtol=1.e-15)


@pytest.mark.parametrize('fdot_mult', [0.8])
def test_Chirp_wdm_match_TT_long(fdot_mult):
    """Test for match between different methods of getting wavelet transform"""
    toml_filename_in = 'tests/sparse_wdm_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)
    nt_lim_waveform_long = PixelGenericRange(0, wc.Nt * wc.Nf, wc.dt, 0.)

    # Want gamma*tau large so that the SPA is accurate
    # Pick fdot so that both the Taylor expanded time and frequency domain transforms are valid
    # => fdot < 8 DF/T_w and fdot > DF^2/8 = DF/(16 DT)
    # Also need to ensure that the sparse time domain transform is valid
    # => fdot > DF/T_w
    # fdot = gamma/tau

    fdot = 3.105 * fdot_mult * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples
    print(fdot, 8 * wc.DF / wc.Tw, wc.DF**2 / 8)
    assert fdot < 8 * wc.DF / wc.Tw
    # assert fdot > _wc.DF**2/8
    assert fdot > wc.DF / wc.Tw
    assert -wc.dfd * wc.Nfd_negative < fdot
    assert fdot < wc.dfd * (wc.Nfd - wc.Nfd_negative)
    tp = wc.Tobs / 2.
    f0 = 16 * wc.DF
    fp = f0 + fdot * tp  # ensures that frequency starts positive

    print('%f %f' % (fp / wc.DF, tp / wc.DT))
    gamma = fp / 8.
    tau = gamma / fdot
    Phi0 = 0.
    A0 = 10000.

    intrinsic = LinearChirpletIntrinsicParams(A0, Phi0, fp, tp, tau, gamma)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2,
                                psi=0.3)  # Replace this with a real extrinsic param object if needed

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    print('computing time domain waveforms')
    n_long = wc.Nt * wc.Nf
    T_long = np.arange(0, n_long) * wc.dt
    waveform_long_c = StationaryWaveformTime(T_long, np.zeros(n_long), np.zeros(n_long), np.zeros(n_long), np.zeros(n_long))
    chirplet_time_intrinsic(waveform_long_c, intrinsic, waveform_long_c.T, nt_lim_waveform_long)

    hs_time_c = waveform_long_c.AT * np.cos(waveform_long_c.PT)

    ts = waveform_long_c.T

    print('finished time domain waveforms')

    # Time domain Taylor expansion transform
    waveletTT = LinearChirpletWaveletTaylorTime(params, wc, lc, nt_lim_waveform, wavelet_mode=1, response_mode=2)

    waveTT = wavelet_sparse_to_dense(waveletTT.wavelet_waveform, wc)[:, :, 0]

    wave_got_time = transform_wavelet_time(hs_time_c, wc.Nf, wc.Nt)
    fs_fft = np.arange(0, ts.size // 2 + 1) * 1 / (wc.Tobs)
    n_fft = fs_fft.size
    waveform_long_f = StationaryWaveformFreq(fs_fft, np.zeros(n_fft), np.zeros(n_fft), np.zeros(n_fft),
                                             np.zeros(n_fft))
    chirplet_freq_intrinsic(waveform_long_f, intrinsic, waveform_long_f.F)
    PPfs = waveform_long_f.PF
    AAfs = waveform_long_f.AF
    AAfs = AAfs / (2 * wc.dt)
    hs_freq = np.exp(-1.0j * PPfs) * AAfs.astype(np.complex128)
    wave_got_freq = transform_wavelet_freq(hs_freq, wc.Nf, wc.Nt)

    assert_allclose(wave_got_freq, wave_got_time, atol=3.e-4, rtol=1.e-15)

    assert_allclose(waveTT[(waveTT != 0.)], wave_got_freq[(waveTT != 0.)], atol=2.e-2, rtol=1.e-10)
    assert_allclose(waveTT[(waveTT != 0.)], wave_got_time[(waveTT != 0.)], atol=2.e-2, rtol=1.e-10)

    pow_freq = np.sum(wave_got_freq**2)
    pow_time = np.sum(wave_got_time**2)
    pow_TT = np.sum(waveTT**2)
    # check powers match
    assert_allclose(pow_freq, pow_time, atol=1.e-11, rtol=1.e-9)
    assert_allclose(pow_freq, pow_TT, atol=1.e-11, rtol=7.e-4)

    waveTT_res = waveTT  # waveTT.reshape(_wc.Nf,_wc.Nt).T

    matchsigf_sigt = 1 - np.sum(wave_got_freq * wave_got_time) / np.sqrt(np.sum(wave_got_freq**2) * np.sum(wave_got_time**2))

    matchTTsigt = 1 - np.sum(waveTT_res * wave_got_time) / np.sqrt(np.sum(waveTT_res**2) * np.sum(wave_got_time**2))

    matchTTsigf = 1 - np.sum(waveTT_res * wave_got_freq) / np.sqrt(np.sum(waveTT_res**2) * np.sum(wave_got_freq**2))

    assert matchsigf_sigt < 1.e-6
    assert matchTTsigt < 1.e-3

    assert matchTTsigf < 1.e-3


@pytest.mark.parametrize('fdot_mult', [0.8])
def test_Chirp_wdm_match_TS_long(fdot_mult):
    """Test for match between different methods of getting wavelet transform"""
    toml_filename_in = 'tests/sparse_wdm_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)
    nt_lim_waveform_long = PixelGenericRange(0, wc.Nt * wc.Nf, wc.dt, 0.)

    # Want gamma*tau large so that the SPA is accurate
    # Pick fdot so that both the Taylor expanded time and frequency domain transforms are valid
    # => fdot < 8 DF/T_w and fdot > DF^2/8 = DF/(16 DT)
    # Also need to ensure that the sparse time domain transform is valid
    # => fdot > DF/T_w
    # fdot = gamma/tau

    fdot = 3.105 * fdot_mult * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples
    print(fdot, 8 * wc.DF / wc.Tw, wc.DF**2 / 8)
    assert fdot < 8 * wc.DF / wc.Tw
    # assert fdot > _wc.DF**2/8
    assert fdot > wc.DF / wc.Tw
    assert -wc.dfd * wc.Nfd_negative < fdot
    assert fdot < wc.dfd * (wc.Nfd - wc.Nfd_negative)
    tp = wc.Tobs / 2.
    f0 = 16 * wc.DF
    fp = f0 + fdot * tp  # ensures that frequency starts positive

    print('%f %f' % (fp / wc.DF, tp / wc.DT))
    gamma = fp / 8.
    tau = gamma / fdot
    Phi0 = 0.
    A0 = 10000.

    intrinsic = LinearChirpletIntrinsicParams(A0, Phi0, fp, tp, tau, gamma)

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2,
                                psi=0.3)  # Replace this with a real extrinsic param object if needed

    # Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    print('computing time domain waveforms')
    n_long = wc.Nt * wc.Nf
    T_long = np.arange(0, n_long) * wc.dt
    waveform_long_c = StationaryWaveformTime(T_long, np.zeros(n_long), np.zeros(n_long), np.zeros(n_long), np.zeros(n_long))
    chirplet_time_intrinsic(waveform_long_c, intrinsic, waveform_long_c.T, nt_lim_waveform_long)
    hs_time_c = waveform_long_c.AT * np.cos(waveform_long_c.PT)

    ts = waveform_long_c.T

    print('finished time domain waveforms')

    # Time domain sparse transform
    waveletTS = LinearChirpletWaveletSparseTime(params, wc, lc, nt_lim_waveform, response_mode=2)
    waveTS = wavelet_sparse_to_dense(waveletTS.wavelet_waveform, wc)[:, :, 0]

    fs_fft = np.arange(0, ts.size // 2 + 1) * 1 / (wc.Tobs)
    n_fft = fs_fft.size
    waveform_long_f = StationaryWaveformFreq(fs_fft, np.zeros(n_fft), np.zeros(n_fft), np.zeros(n_fft),
                                             np.zeros(n_fft))
    chirplet_freq_intrinsic(waveform_long_f, intrinsic, waveform_long_f.F)
    PPfs = waveform_long_f.PF
    AAfs = waveform_long_f.AF

    wave_got_time = transform_wavelet_time(hs_time_c, wc.Nf, wc.Nt)

    AAfs = AAfs / (2 * wc.dt)
    hs_freq = np.exp(-1.0j * PPfs) * AAfs.astype(np.complex128)
    wave_got_freq = transform_wavelet_freq(hs_freq, wc.Nf, wc.Nt)

    assert_allclose(wave_got_freq, wave_got_time, atol=3.e-4, rtol=1.e-15)

    assert_allclose(waveTS[(waveTS != 0.)], wave_got_freq[(waveTS != 0.)], atol=2.e-2, rtol=1.e-10)
    assert_allclose(waveTS[(waveTS != 0.)], wave_got_time[(waveTS != 0.)], atol=2.e-2, rtol=1.e-10)

    pow_freq = np.sum(wave_got_freq**2)
    pow_time = np.sum(wave_got_time**2)
    pow_TS = np.sum(waveTS**2)
    # check powers match
    assert_allclose(pow_freq, pow_time, atol=1.e-11, rtol=1.e-9)
    assert_allclose(pow_freq, pow_TS, atol=1.e-11, rtol=6.e-5)

    waveTS_res = waveTS

    matchsigf_sigt = 1 - np.sum(wave_got_freq * wave_got_time) / np.sqrt(np.sum(wave_got_freq**2) * np.sum(wave_got_time**2))

    matchTSsigt = 1 - np.sum(waveTS_res * wave_got_time) / np.sqrt(np.sum(waveTS_res**2) * np.sum(wave_got_time**2))

    matchTSsigf = 1 - np.sum(waveTS_res * wave_got_freq) / np.sqrt(np.sum(waveTS_res**2) * np.sum(wave_got_freq**2))

    assert matchsigf_sigt < 1.e-6
    assert matchTSsigt < 1.e-3

    assert matchTSsigf < 1.e-3
