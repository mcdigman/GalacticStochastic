"""Test that the computed SNR scales as expected with changes in (Nf, Nt, dt, mult)."""

from pathlib import Path

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose, assert_array_equal

from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams, LinearFrequencyWaveletWaveformTime
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalStationaryDenseNoiseModel
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.wdm_config import get_wavelet_model


# scaling on (Nf, Nt, dt, mult) in the second configuration
# (2.0, 2.0, 2.0, 2.0),
@pytest.mark.parametrize(
    'channel_mult',
    [
        (8.0, 0.125, 1.0, 0.5),
        (4.0, 0.25, 1.0, 0.5),
        (2.0, 0.5, 1.0, 0.5),
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
        (1.0, 1.0, 0.5, 1.0),
        (0.5, 8.0, 0.25, 1.0),
        (0.5, 4.0, 0.5, 1.0),
        (0.5, 0.5, 0.5, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 2.0, 1.0, 1.0),
        (1.0, 0.5, 1.0, 1.0),
        (0.5, 1.0, 2.0, 1.0),
    ],
)
def test_noise_scaling(channel_mult: tuple[float, float, float, float]) -> None:
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
        if channel_mult[2] == 1.0:
            assert t_obs1 == t_obs2
    else:
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

    # fs1 = np.arange(0, wc1.Nf) * wc1.DF
    # fs2 = np.arange(0, wc2.Nf) * wc2.DF
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

    noise_manager1 = DiagonalStationaryDenseNoiseModel(noise1, wc1, prune=1, nc_snr=noise1.shape[1])
    noise_manager2 = DiagonalStationaryDenseNoiseModel(noise2, wc2, prune=1, nc_snr=noise2.shape[1])

    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=1.0,  # amplitude
        phi0=0.3,  # phase at t=0
        F0=1.0e-4,  # initial frequency (Hz)
        FTd0=3.0e-12,  # frequency derivative (Hz/s)
    )

    print(intrinsic.FTd0, 8 * wc1.DF / wc1.Tw, 8 * wc2.DF / wc2.Tw, wc1.DF**2 / 8, wc2.DF**2 / 8, wc1.dfd * (wc1.Nfd - wc1.Nfd_negative), wc2.dfd * (wc2.Nfd - wc2.Nfd_negative))
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

    waveform1 = LinearFrequencyWaveletWaveformTime(params, wc1, lc1, nt_lim_waveform1, table_cache_mode='check', table_output_mode='skip', response_mode=2)
    waveform2 = LinearFrequencyWaveletWaveformTime(params, wc2, lc2, nt_lim_waveform2, table_cache_mode='check', table_output_mode='skip', response_mode=2)

    wavelet_waveform1 = waveform1.get_unsorted_coeffs()
    wavelet_waveform2 = waveform2.get_unsorted_coeffs()

    snrs1 = noise_manager1.get_sparse_snrs(wavelet_waveform1, nt_lim_snr1)
    snrs2 = noise_manager2.get_sparse_snrs(wavelet_waveform2, nt_lim_snr2)

    snr_tot1 = np.linalg.norm(snrs1)
    snr_tot2 = np.linalg.norm(snrs2)
    print(snrs1, snr_tot1)
    print(snrs2, snr_tot2)
    print(t_rat_snr, time_rat_waveform)
    print((wc1.DF * wc1.Nf) / intrinsic.F0, (wc2.DF * wc2.Nf) / intrinsic.F0, intrinsic.F0 / wc1.DF, intrinsic.F0 / wc2.DF)

    if waveform1.source_waveform.response_mode == 2 and lc1.noise_curve_mode == 1:
        assert_array_equal(snrs1[0], snrs1)
    if waveform2.source_waveform.response_mode == 2 and lc2.noise_curve_mode == 1:
        assert_array_equal(snrs2[0], snrs2)

    # snr^2 should linearly scale with observing time
    if spectrum_equal and time_equal and tw_equal:
        # precision case
        assert_allclose(snr_tot1**2, snr_tot2**2 * t_rat_snr, atol=1.0e-30, rtol=1.0e-14)
        assert_allclose(snrs1**2, snrs2**2 * t_rat_snr, atol=1.0e-30, rtol=1.0e-14)
    elif spectrum_equal:
        # somewhat less precise case
        assert_allclose(snr_tot1**2, snr_tot2**2 * t_rat_snr, atol=1.0e-30, rtol=2.0e-3)
        assert_allclose(snrs1**2, snrs2**2 * t_rat_snr, atol=1.0e-30, rtol=2.0e-3)
    elif width_equal:
        # somewhat less precise case
        assert_allclose(snr_tot1**2, snr_tot2**2 * t_rat_snr, atol=1.0e-30, rtol=5.0e-2)
        assert_allclose(snrs1**2, snrs2**2 * t_rat_snr, atol=1.0e-30, rtol=5.0e-2)
    else:
        assert_allclose(snr_tot1**2, snr_tot2**2 * t_rat_snr, atol=1.0e-30, rtol=5.0e-2)
        assert_allclose(snrs1**2, snrs2**2 * t_rat_snr, atol=1.0e-30, rtol=5.0e-2)

    # import matplotlib.pyplot as plt
    # plt.plot(fs1[1:], noise1[1:,0])
    # plt.plot(fs2[1:], noise2[1:,0])
    # plt.show()
