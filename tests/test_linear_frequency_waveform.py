from pathlib import Path

import numpy as np
import tomllib
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline

from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams, LinearFrequencySourceWaveformTime
from LisaWaveformTools.linear_frequency_source_freq import LinearFrequencySourceWaveformFreq
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.stationary_source_waveform import ExtrinsicParams, SourceParams
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wdm_config import get_wavelet_model


def test_intrinsic_waveform_agreement():
    """Test the non TDI parts of the waveform agree between the time and frequency domain cases"""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    nt_lim_waveform = PixelTimeRange(0, wc.Nt)

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

    NF_min = 0
    NF_max = wc.Nf

    freeze_limits = True

    lf_waveform_freq = LinearFrequencySourceWaveformFreq(
        params=params,
        lc=lc,
        wc=wc,
        NF_min=NF_min,
        NF_max=NF_max,
        freeze_limits=freeze_limits,
        n_pad_F=10,  # optional, default is 10
    )

    print('LinearFrequencySourceWaveformFreq initialized:')
    print(lf_waveform_freq)

    lf_waveform_time = LinearFrequencySourceWaveformTime(
        params=params,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
        wc=wc,
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
