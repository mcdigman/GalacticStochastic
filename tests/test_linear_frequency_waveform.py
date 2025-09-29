from pathlib import Path

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from scipy.interpolate import InterpolatedUnivariateSpline

from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams, LinearFrequencySourceWaveformTime
from LisaWaveformTools.linear_frequency_source_freq import LinearFrequencySourceWaveformFreq
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from LisaWaveformTools.stationary_freq_source import StationarySourceWaveformFreq
from LisaWaveformTools.stationary_time_source import StationarySourceWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.wdm_config import get_wavelet_model


def test_intrinsic_waveform_agreement() -> None:
    """Test the non TDI parts of the waveform agree between the time and frequency domain cases"""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

# Setup the intrinsic parameters for the binary source
    intrinsic = LinearFrequencyIntrinsicParams(
        amp0_t=1.0,    # amplitude
        phi0=0.3,      # phase at t=0
        F0=1.e-3,       # initial frequency (Hz)
        FTd0=0.5e-10,      # frequency derivative (Hz/s)
    )

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)  # Replace this with a real extrinsic param object if needed

# Bundle parameters
    params = SourceParams(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

    NF_lim = PixelGenericRange(0, wc.Nf, wc.DF, 0.)

    freeze_limits = True

    lf_waveform_freq = LinearFrequencySourceWaveformFreq(
        params=params,
        lc=lc,
        T_obs=wc.Tobs,
        nf_lim_absolute=NF_lim,
        freeze_limits=freeze_limits,
        n_pad_F=10,  # optional, default is 10
    )

    print('LinearFrequencySourceWaveformFreq initialized:')
    print(lf_waveform_freq)

    lf_waveform_time = LinearFrequencySourceWaveformTime(
        params=params,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
    )

    t_wave = lf_waveform_time.intrinsic_waveform
    f_wave = lf_waveform_freq.intrinsic_waveform

    FTdF = InterpolatedUnivariateSpline(f_wave.TF, 1. / f_wave.TFp, k=3, ext=1)(t_wave.T)
    TFpT = InterpolatedUnivariateSpline(t_wave.FT, 1. / t_wave.FTd, k=3, ext=1)(f_wave.F)
    AmpTF = InterpolatedUnivariateSpline(f_wave.TF, f_wave.AF / np.sqrt(np.abs(f_wave.TFp)), k=3, ext=1)(t_wave.T)

    # check all the agreements that do not depend on wavefront time
    assert_allclose(FTdF, t_wave.FTd, atol=1.e-20, rtol=1.e-10)
    assert_allclose(np.gradient(f_wave.TF, wc.DF)[2:-1], f_wave.TFp[2:-1], atol=1.e-20, rtol=1.e-10)
    assert_allclose(np.gradient(f_wave.PF, wc.DF)[2:-1] / (2 * np.pi), f_wave.TF[2:-1], atol=1.e-20, rtol=1.e-10)
    assert_allclose(TFpT[(f_wave.TF > 0.) & (wc.Tobs > f_wave.TF)], f_wave.TFp[(f_wave.TF > 0.) & (wc.Tobs > f_wave.TF)], atol=1.e-20, rtol=1.e-10)
    assert_allclose(FTdF, t_wave.FTd, atol=1.e-20, rtol=1.e-10)
    assert_allclose(t_wave.AT, AmpTF, atol=1.e-20, rtol=1.e-10)

    assert_allclose(np.gradient(t_wave.FT, lf_waveform_time.wavefront_time), t_wave.FTd, atol=1.e-20, rtol=1.e-10)
    assert_allclose(np.gradient(t_wave.PT, lf_waveform_time.wavefront_time)[1:-1] / (2 * np.pi), t_wave.FT[1:-1], atol=1.e-20, rtol=1.e-10)

    PTF_base = -f_wave.PF + 2. * np.pi * f_wave.F * f_wave.TF - np.pi / 4.

    PTF = InterpolatedUnivariateSpline(f_wave.TF, PTF_base, k=3, ext=1)(lf_waveform_time.wavefront_time)
    FTF = InterpolatedUnivariateSpline(f_wave.TF, f_wave.F, k=3, ext=1)(lf_waveform_time.wavefront_time)

    PFT_base = -t_wave.PT + 2. * np.pi * lf_waveform_time.wavefront_time * t_wave.FT - np.pi / 4.
    PFT = InterpolatedUnivariateSpline(t_wave.FT, PFT_base, k=3, ext=1)(f_wave.F)
    TFT = InterpolatedUnivariateSpline(t_wave.FT, lf_waveform_time.wavefront_time, k=3, ext=1)(f_wave.F)

    assert_allclose(FTF, t_wave.FT, atol=1.e-20, rtol=1.e-10)
    assert_allclose(PTF, t_wave.PT, atol=1.e-20, rtol=1.e-10)
    mask = (f_wave.TF > 0.) & (wc.Tobs > f_wave.TF)
    assert_allclose(TFT[mask], f_wave.TF[mask], atol=1.e-20, rtol=1.e-10)
    assert_allclose(PFT[mask], f_wave.PF[mask], atol=1.e-20, rtol=1.e-10)


def test_intrinsic_update_consistent1_time() -> None:
    """Test the non TDI parts of the waveform update consistently."""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

    # Setup the intrinsic parameters for the binary source
    intrinsic1 = LinearFrequencyIntrinsicParams(
        amp0_t=1.0,    # amplitude
        phi0=0.3,      # phase at t=0
        F0=1.e-3,       # initial frequency (Hz)
        FTd0=0.5e-10,      # frequency derivative (Hz/s)
    )
    intrinsic2 = LinearFrequencyIntrinsicParams(
        amp0_t=2.0,    # amplitude
        phi0=0.6,      # phase at t=0
        F0=2.e-3,       # initial frequency (Hz)
        FTd0=0.3e-10,      # frequency derivative (Hz/s)
    )

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

    lf_waveform_time11 = LinearFrequencySourceWaveformTime(
        params=params1,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
    )

    lf_waveform_time22 = LinearFrequencySourceWaveformTime(
        params=params2,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
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

    lf_waveform_time12 = LinearFrequencySourceWaveformTime(
        params=params1,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
    )
    lf_waveform_time21 = LinearFrequencySourceWaveformTime(
        params=params2,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
    )

    def assert_match_time(wave1: StationarySourceWaveformTime, wave2: StationarySourceWaveformTime) -> None:
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


@pytest.mark.parametrize('freeze_limits', [True, False])
def test_intrinsic_update_consistent1_freq(freeze_limits: bool) -> None:
    """Test the non TDI parts of the waveform update consistently."""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    # Setup the intrinsic parameters for the binary source
    intrinsic1 = LinearFrequencyIntrinsicParams(
        amp0_t=1.0,    # amplitude
        phi0=0.3,      # phase at t=0
        F0=1.e-3,       # initial frequency (Hz)
        FTd0=0.5e-10,      # frequency derivative (Hz/s)
    )
    intrinsic2 = LinearFrequencyIntrinsicParams(
        amp0_t=2.0,    # amplitude
        phi0=0.6,      # phase at t=0
        F0=2.e-3,       # initial frequency (Hz)
        FTd0=0.3e-10,      # frequency derivative (Hz/s)
    )

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

    NF_lim = PixelGenericRange(0, wc.Nf, wc.DF, 0.)

    lf_waveform_freq11 = LinearFrequencySourceWaveformFreq(
        params=params1,
        lc=lc,
        T_obs=wc.Tobs,
        nf_lim_absolute=NF_lim,
        freeze_limits=freeze_limits,
        n_pad_F=10,  # optional, default is 10
    )

    lf_waveform_freq22 = LinearFrequencySourceWaveformFreq(
        params=params2,
        lc=lc,
        T_obs=wc.Tobs,
        nf_lim_absolute=NF_lim,
        freeze_limits=freeze_limits,
        n_pad_F=10,  # optional, default is 10
    )

    assert not np.all(lf_waveform_freq11.intrinsic_waveform.AF == lf_waveform_freq22.intrinsic_waveform.AF)
    assert not np.all(lf_waveform_freq11.intrinsic_waveform.PF == lf_waveform_freq22.intrinsic_waveform.PF)
    assert not np.all(lf_waveform_freq11.intrinsic_waveform.TF == lf_waveform_freq22.intrinsic_waveform.TF)
    assert not np.all(lf_waveform_freq11.intrinsic_waveform.TFp == lf_waveform_freq22.intrinsic_waveform.TFp)
    assert not np.all(lf_waveform_freq11.tdi_waveform.AF == lf_waveform_freq22.tdi_waveform.AF)
    assert not np.all(lf_waveform_freq11.tdi_waveform.PF == lf_waveform_freq22.tdi_waveform.PF)
    assert not np.all(lf_waveform_freq11.tdi_waveform.TF == lf_waveform_freq22.tdi_waveform.TF)
    assert not np.all(lf_waveform_freq11.tdi_waveform.TFp == lf_waveform_freq22.tdi_waveform.TFp)

    assert_equal(lf_waveform_freq11.nc_waveform, lf_waveform_freq22.nc_waveform)
    assert_equal(lf_waveform_freq11.intrinsic_waveform.F, lf_waveform_freq22.intrinsic_waveform.F)
    assert_equal(lf_waveform_freq11.tdi_waveform.F, lf_waveform_freq22.tdi_waveform.F)
    assert_equal(lf_waveform_freq11.tdi_waveform.F, lf_waveform_freq22.tdi_waveform.F)

    lf_waveform_freq12 = LinearFrequencySourceWaveformFreq(
        params=params1,
        lc=lc,
        T_obs=wc.Tobs,
        nf_lim_absolute=NF_lim,
        freeze_limits=freeze_limits,
        n_pad_F=10,  # optional, default is 10
    )
    lf_waveform_freq21 = LinearFrequencySourceWaveformFreq(
        params=params2,
        lc=lc,
        T_obs=wc.Tobs,
        nf_lim_absolute=NF_lim,
        freeze_limits=freeze_limits,
        n_pad_F=10,  # optional, default is 10
    )

    def assert_match_freq(wave1: StationarySourceWaveformFreq, wave2: StationarySourceWaveformFreq) -> None:
        assert_equal(wave1.nc_waveform, wave2.nc_waveform)
        assert_equal(wave1.intrinsic_waveform.F, wave2.intrinsic_waveform.F)
        assert_equal(wave1.tdi_waveform.F, wave2.tdi_waveform.F)
        assert_equal(wave1.NF, wave2.NF)
        assert_equal(wave1.nf_lim_absolute.nx_min, wave2.nf_lim_absolute.nx_min)
        assert_equal(wave1.nf_lim_absolute.nx_max, wave2.nf_lim_absolute.nx_max)
        assert_array_equal(wave1.FFs, wave2.FFs)

        assert_array_equal(wave1.intrinsic_waveform.AF, wave2.intrinsic_waveform.AF)
        assert_array_equal(wave1.intrinsic_waveform.PF, wave2.intrinsic_waveform.PF)
        assert_array_equal(wave1.intrinsic_waveform.TF, wave2.intrinsic_waveform.TF)
        assert_array_equal(wave1.intrinsic_waveform.TFp, wave2.intrinsic_waveform.TFp)

        low_loc1 = 0
        high_loc1 = wave1.tdi_waveform.TFp.shape[1]
        low_loc2 = 0
        high_loc2 = wave2.tdi_waveform.TFp.shape[1]

        if not freeze_limits:
            assert_equal(wave1.nf_lim.nx_max, wave2.nf_lim.nx_max)
            assert_equal(wave1.nf_lim.nx_min, wave2.nf_lim.nx_min)
            assert_equal(wave1.itrFCut, wave2.itrFCut)
        else:
            nf_low_loc = max(wave1.nf_lim.nx_min, wave2.nf_lim.nx_min)
            nf_high_loc = min(wave1.nf_lim.nx_max, wave2.nf_lim.nx_max)
            nf_high_loc = min(nf_high_loc, nf_low_loc)
            low_loc1 = nf_low_loc - wave1.nf_lim.nx_min
            low_loc2 = nf_low_loc - wave2.nf_lim.nx_min
            high_loc1 = nf_high_loc - wave1.nf_lim.nx_min
            high_loc2 = nf_high_loc - wave2.nf_lim.nx_min

        assert_equal(wave1.kdotx[low_loc1:high_loc1], wave2.kdotx[low_loc2:high_loc2])
        assert_array_equal(wave1.tdi_waveform.AF[:, low_loc1:high_loc1], wave2.tdi_waveform.AF[:, low_loc2:high_loc2])
        assert_array_equal(wave1.tdi_waveform.PF[:, low_loc1:high_loc1], wave2.tdi_waveform.PF[:, low_loc2:high_loc2])
        assert_array_equal(wave1.tdi_waveform.TF[:, low_loc1:high_loc1], wave2.tdi_waveform.TF[:, low_loc2:high_loc2])
        assert_array_equal(wave1.tdi_waveform.TFp[:, low_loc1:high_loc1], wave2.tdi_waveform.TFp[:, low_loc2:high_loc2])

    # check starting match
    assert_match_freq(lf_waveform_freq11, lf_waveform_freq12)
    assert_match_freq(lf_waveform_freq22, lf_waveform_freq21)

    # check idempotence from start
    lf_waveform_freq12.update_params(params1)
    lf_waveform_freq21.update_params(params2)

    assert_match_freq(lf_waveform_freq11, lf_waveform_freq12)
    assert_match_freq(lf_waveform_freq22, lf_waveform_freq21)

    # check swap parameters
    lf_waveform_freq12.update_params(params2)
    lf_waveform_freq21.update_params(params1)

    assert_match_freq(lf_waveform_freq11, lf_waveform_freq21)
    assert_match_freq(lf_waveform_freq22, lf_waveform_freq12)

    # check idempotence from swap
    lf_waveform_freq12.update_params(params2)
    lf_waveform_freq21.update_params(params1)

    assert_match_freq(lf_waveform_freq22, lf_waveform_freq12)
    assert_match_freq(lf_waveform_freq11, lf_waveform_freq21)

    # check return to start
    lf_waveform_freq12.update_params(params1)
    lf_waveform_freq21.update_params(params2)

    assert_match_freq(lf_waveform_freq11, lf_waveform_freq12)
    assert_match_freq(lf_waveform_freq22, lf_waveform_freq21)

    # check idempotence from return to start
    lf_waveform_freq12.update_params(params1)
    lf_waveform_freq21.update_params(params2)

    assert_match_freq(lf_waveform_freq11, lf_waveform_freq12)
    assert_match_freq(lf_waveform_freq22, lf_waveform_freq21)
