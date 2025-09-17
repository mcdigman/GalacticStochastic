"""test the Chirp_WDM functions"""

from pathlib import Path

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq, transform_wavelet_time

from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams, amp_phase_f, chirp_time
from WaveletWaveforms.sparse_wavelet_time import wavelet_SparseT, wavelet_TaylorT
from WaveletWaveforms.wdm_config import get_wavelet_model


@pytest.mark.parametrize('fdot_mult', [0.8])
def test_Chirp_wdm_match4(fdot_mult):
    """Test for match between different methods of getting wavelet transform"""
    toml_filename_in = 'tests/sparse_wdm_test_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)

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

    params = LinearChirpletIntrinsicParams(A0, Phi0, fp, tp, tau, gamma)
    # params = np.zeros(10)
    # params[0] = tau  # time spread
    # params[1] = 0.2  # costh
    # params[2] = 1.0  # phi
    # params[3] = A0  # amplitude
    # params[4] = -0.3  # cosi
    # params[5] = 0.8  # psi
    # params[6] = Phi0  # phi0
    # params[7] = gamma  # frequency spread
    # params[8] = tp  # central time
    # params[9] = fp  # central frequency

    print('computing time domain waveforms')
    hs_time, waveform_long = chirp_time(params, wc)
    ts = waveform_long.T
    print('finished time domain waveforms')

    # Time domain Taylor expansion transform
    _TlistT, waveTT = wavelet_TaylorT(params, wc, approximation=1)
    _TlistT_exact, waveTT_exact = wavelet_TaylorT(params, wc, approximation=0)
    # Time domain sparse transform
    _TlistS, waveTS = wavelet_SparseT(params, wc)

    maskTS = (waveTS != 0.)
    maskTT = (waveTT != 0.)
    maskTTTS = maskTS & maskTT

    # import matplotlib.pyplot as plt
    print(waveTT.shape)
    print(waveTS.shape)

    wave_got_time = transform_wavelet_time(hs_time, wc.Nf, wc.Nt)
    fs_fft = np.arange(0, ts.size // 2 + 1) * 1 / (wc.Tobs)
    PPfs, AAfs = amp_phase_f(fs_fft, params)
    AAfs = AAfs / (2 * wc.dt)
    hs_freq = np.exp(-1j * PPfs) * AAfs
    wave_got_freq = transform_wavelet_freq(hs_freq, wc.Nf, wc.Nt)

    # freq_TT = inverse_wavelet_freq(waveTT[:, :], wc.Nf, wc.Nt)
    # freq_TT_exact = inverse_wavelet_freq(waveTT_exact[:, :], wc.Nf, wc.Nt)
    # freq_TS = inverse_wavelet_freq(waveTS[:, :], _wc.Nf, _wc.Nt)
    # freq_f_mask = inverse_wavelet_freq(wave_got_freq * maskTT, wc.Nf, wc.Nt)
    # freq_t_mask = inverse_wavelet_freq(wave_got_time*maskTT, _wc.Nf, _wc.Nt)

    # time_TT = inverse_wavelet_time(waveTT[:, :], wc.Nf, wc.Nt)
    # time_TS = inverse_wavelet_time(waveTS[:, :], _wc.Nf, _wc.Nt)
    # time_f_mask = inverse_wavelet_time(wave_got_freq * maskTT, wc.Nf, wc.Nt)
    # time_t_mask = inverse_wavelet_time(wave_got_time*maskTT, _wc.Nf, _wc.Nt)

    # plt.plot(np.abs(hs_freq))
    # plt.plot(np.abs(freq_f_mask))
    # plt.plot(np.abs(freq_TT))
    # plt.plot(np.abs(freq_TT_exact))
    # plt.plot(np.abs(freq_TS))
    # plt.show()

    # plt.plot(np.abs(hilbert(time_f_mask)))
    # plt.plot(np.abs(hilbert(time_TT)))
    # plt.plot(np.abs(hilbert(time_TS)))
    # plt.show()

    # plt.imshow(np.rot90(wave_got_freq),aspect='auto')
    # plt.show()

    # plt.imshow(np.rot90((waveTT-wave_got_time)**2),aspect='auto')
    # plt.show()

    # plt.imshow(np.rot90((waveTS-wave_got_time)**2),aspect='auto')
    # plt.show()

    # plt.imshow(np.rot90(waveTT*maskTTTS),aspect='auto')
    # plt.show()
    # plt.imshow(np.rot90(waveTS*maskTTTS), aspect='auto')
    # plt.show()
    # plt.imshow(np.rot90((waveTT-waveTS)**2*maskTTTS),aspect='auto')
    # plt.show()

    assert_allclose(wave_got_freq, wave_got_time, atol=3.e-4, rtol=1.e-15)
    assert_allclose(waveTT[maskTT], waveTT_exact[maskTT], atol=3.e-4, rtol=1.e-15)

    assert_allclose(waveTT[maskTTTS], waveTS[maskTTTS], atol=1.e-2, rtol=1.e-10)

    assert_allclose(waveTT[(waveTT != 0.)], wave_got_freq[(waveTT != 0.)], atol=2.e-2, rtol=1.e-10)
    assert_allclose(waveTT[(waveTT != 0.)], wave_got_time[(waveTT != 0.)], atol=2.e-2, rtol=1.e-10)

    assert_allclose(waveTS[(waveTS != 0.)], wave_got_freq[(waveTS != 0.)], atol=2.e-2, rtol=1.e-10)
    assert_allclose(waveTS[(waveTS != 0.)], wave_got_time[(waveTS != 0.)], atol=2.e-2, rtol=1.e-10)

    pow_freq = np.sum(wave_got_freq**2)
    pow_time = np.sum(wave_got_time**2)
    pow_TT = np.sum(waveTT**2)
    pow_TS = np.sum(waveTS**2)
    # check powers match
    assert_allclose(pow_freq, pow_time, atol=1.e-11, rtol=1.e-9)
    assert_allclose(pow_freq, pow_TT, atol=1.e-11, rtol=7.e-4)
    assert_allclose(pow_freq, pow_TS, atol=1.e-11, rtol=6.e-5)
    assert_allclose(pow_TT, pow_TS, atol=1.e-11, rtol=7.e-4)

    waveTT_res = waveTT  # waveTT.reshape(_wc.Nf,_wc.Nt).T
    waveTS_res = waveTS  # waveTS.reshape(_wc.Nf,_wc.Nt).T

    matchsigf_sigt = 1 - np.sum(wave_got_freq * wave_got_time) / np.sqrt(np.sum(wave_got_freq**2) * np.sum(wave_got_time**2))

    matchTTsigt = 1 - np.sum(waveTT_res * wave_got_time) / np.sqrt(np.sum(waveTT_res**2) * np.sum(wave_got_time**2))
    matchTSsigt = 1 - np.sum(waveTS_res * wave_got_time) / np.sqrt(np.sum(waveTS_res**2) * np.sum(wave_got_time**2))

    matchTTsigf = 1 - np.sum(waveTT_res * wave_got_freq) / np.sqrt(np.sum(waveTT_res**2) * np.sum(wave_got_freq**2))
    matchTSsigf = 1 - np.sum(waveTS_res * wave_got_freq) / np.sqrt(np.sum(waveTS_res**2) * np.sum(wave_got_freq**2))

    matchTTTS = 1 - np.sum(waveTT_res * waveTS_res) / np.sqrt(np.sum(waveTT_res**2) * np.sum(waveTS_res**2))

    assert matchsigf_sigt < 1.e-6
    assert matchTTsigt < 1.e-3
    assert matchTSsigt < 1.e-3

    assert matchTTsigf < 1.e-3
    assert matchTSsigf < 1.e-3

    assert matchTTTS < 1.e-4
