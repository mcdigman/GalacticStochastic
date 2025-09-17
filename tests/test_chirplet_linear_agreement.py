"""test the Chirp_WDM functions"""

from pathlib import Path

import numpy as np
import tomllib
from numpy.testing import assert_allclose

from LisaWaveformTools.chirplet_source_freq import LinearChirpletSourceWaveformFreq
from LisaWaveformTools.chirplet_source_time import LinearChirpletSourceWaveformTime
from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams, LinearFrequencySourceWaveformTime
from LisaWaveformTools.linear_frequency_source_freq import LinearFrequencySourceWaveformFreq
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.stationary_source_waveform import ExtrinsicParams, SourceParams
from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wdm_config import get_wavelet_model


def test_cross_waveform_agreement_time():
    """Test the linear frequency and chirplet time domain waveforms agree"""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    nt_lim_waveform = PixelTimeRange(0, wc.Nt)

    fdot_mult = 0.1
    fdot = 3.105 * fdot_mult * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples

    f0 = 16 * wc.DF

    tp = 0.0

    fp = f0 + fdot * tp  # ensures that frequency starts positive

    gamma = fp * 800000.0
    tau = gamma / fdot

    phi0 = 0.1
    A0_freq = 1.
    A0_time = np.sqrt(fdot) * A0_freq
    # Setup the intrinsic parameters for the binary source
    intrinsic_chirplet = LinearChirpletIntrinsicParams(A0_freq, phi0, fp, tp, tau, gamma)
    intrinsic_linear = LinearFrequencyIntrinsicParams(
        amp0_t=A0_time,    # amplitude
        phi0=phi0,      # phase at t=0
        F0=f0,       # initial frequency (Hz)
        FTd0=fdot,      # frequency derivative (Hz/s)
    )

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)  # Replace this with a real extrinsic param object if needed

    # Bundle parameters
    params_chirplet = SourceParams(
        intrinsic=intrinsic_chirplet,
        extrinsic=extrinsic,
    )

    params_linear = SourceParams(
        intrinsic=intrinsic_linear,
        extrinsic=extrinsic,
    )

    linear_time = LinearFrequencySourceWaveformTime(
        params=params_linear,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
        wc=wc,
    )

    chirplet_time = LinearChirpletSourceWaveformTime(
        params=params_chirplet,
        nt_lim_waveform=nt_lim_waveform,
        lc=lc,
        wc=wc,
    )

    assert_allclose(linear_time.intrinsic_waveform.T, chirplet_time.intrinsic_waveform.T, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_time.intrinsic_waveform.FTd, chirplet_time.intrinsic_waveform.FTd, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_time.intrinsic_waveform.FT, chirplet_time.intrinsic_waveform.FT,
                    atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_time.intrinsic_waveform.PT, chirplet_time.intrinsic_waveform.PT,
                    atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_time.intrinsic_waveform.AT, chirplet_time.intrinsic_waveform.AT,
                    atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_time.wavefront_time, chirplet_time.wavefront_time, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_time.tdi_waveform.FT, chirplet_time.tdi_waveform.FT, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_time.tdi_waveform.FTd, chirplet_time.tdi_waveform.FTd, atol=1.e-20,
                    rtol=1.e-10)
    assert_allclose(linear_time.tdi_waveform.PT, chirplet_time.tdi_waveform.PT, atol=1.e-20,
                    rtol=1.e-10)
    assert_allclose(linear_time.tdi_waveform.AT, chirplet_time.tdi_waveform.AT, atol=1.e-20,
                    rtol=1.e-10)

    def test_cross_waveform_agreement_time():
        """Test the non TDI parts of the waveform agree between the time and frequency domain cases"""
        toml_filename = 'tests/galactic_fit_test_config1.toml'

        with Path(toml_filename).open('rb') as f:
            config = tomllib.load(f)

        wc = get_wavelet_model(config)
        lc = get_lisa_constants(config)

        nt_lim_waveform = PixelTimeRange(0, wc.Nt)

        fdot_mult = 0.1
        fdot = 3.105 * fdot_mult * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples

        f0 = 16 * wc.DF

        tp = 0.0

        fp = f0 + fdot * tp  # ensures that frequency starts positive

        gamma = fp * 800000.0
        tau = gamma / fdot

        phi0 = 0.1
        A0_freq = 1.
        A0_time = np.sqrt(fdot) * A0_freq
        # Setup the intrinsic parameters for the binary source
        intrinsic_chirplet = LinearChirpletIntrinsicParams(A0_freq, phi0, fp, tp, tau, gamma)
        intrinsic_linear = LinearFrequencyIntrinsicParams(
            amp0_t=A0_time,  # amplitude
            phi0=phi0,  # phase at t=0
            F0=f0,  # initial frequency (Hz)
            FTd0=fdot,  # frequency derivative (Hz/s)
        )

        extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2,
                                    psi=0.3)  # Replace this with a real extrinsic param object if needed

        # Bundle parameters
        params_chirplet = SourceParams(
            intrinsic=intrinsic_chirplet,
            extrinsic=extrinsic,
        )

        params_linear = SourceParams(
            intrinsic=intrinsic_linear,
            extrinsic=extrinsic,
        )

        linear_time = LinearFrequencySourceWaveformTime(
            params=params_linear,
            nt_lim_waveform=nt_lim_waveform,
            lc=lc,
            wc=wc,
        )

        chirplet_time = LinearChirpletSourceWaveformTime(
            params=params_chirplet,
            nt_lim_waveform=nt_lim_waveform,
            lc=lc,
            wc=wc,
        )

        assert_allclose(linear_time.intrinsic_waveform.T, chirplet_time.intrinsic_waveform.T, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.intrinsic_waveform.FTd, chirplet_time.intrinsic_waveform.FTd, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.intrinsic_waveform.FT, chirplet_time.intrinsic_waveform.FT, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.intrinsic_waveform.PT, chirplet_time.intrinsic_waveform.PT, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.intrinsic_waveform.AT, chirplet_time.intrinsic_waveform.AT, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.wavefront_time, chirplet_time.wavefront_time, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.tdi_waveform.FT, chirplet_time.tdi_waveform.FT, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.tdi_waveform.FTd, chirplet_time.tdi_waveform.FTd, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.tdi_waveform.PT, chirplet_time.tdi_waveform.PT, atol=1.e-20, rtol=1.e-10)
        assert_allclose(linear_time.tdi_waveform.AT, chirplet_time.tdi_waveform.AT, atol=1.e-20, rtol=1.e-10)


def test_cross_waveform_agreement_freq():
    """Test the linear frequency and chirplet frequency domain waveforms agree"""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    fdot_mult = 0.1
    fdot = 3.105 * fdot_mult * wc.DF / wc.Tw  # used an irrational fraction to ensure the fdot lands between samples

    f0 = 16 * wc.DF

    tp = 0.0

    fp = f0 + fdot * tp  # ensures that frequency starts positive

    gamma = fp * 80000000.0
    tau = gamma / fdot

    phi0 = 0.1
    A0_freq = 1.
    A0_time = np.sqrt(fdot) * A0_freq
    # Setup the intrinsic parameters for the binary source
    intrinsic_chirplet = LinearChirpletIntrinsicParams(A0_freq, phi0, fp, tp, tau, gamma)
    intrinsic_linear = LinearFrequencyIntrinsicParams(
        amp0_t=A0_time,    # amplitude
        phi0=phi0,      # phase at t=0
        F0=f0,       # initial frequency (Hz)
        FTd0=fdot,      # frequency derivative (Hz/s)
    )

    extrinsic = ExtrinsicParams(costh=0.1, phi=0.1, cosi=0.2, psi=0.3)  # Replace this with a real extrinsic param object if needed

    # Bundle parameters
    params_chirplet = SourceParams(
        intrinsic=intrinsic_chirplet,
        extrinsic=extrinsic,
    )

    params_linear = SourceParams(
        intrinsic=intrinsic_linear,
        extrinsic=extrinsic,
    )

    NF_min = 0
    NF_max = wc.Nf
    freeze_limits = False

    linear_freq = LinearFrequencySourceWaveformFreq(
        params=params_linear,
        lc=lc,
        wc=wc,
        NF_min=NF_min,
        NF_max=NF_max,
        freeze_limits=freeze_limits,
        n_pad_F=10,  # optional, default is 10
    )

    chirplet_freq = LinearChirpletSourceWaveformFreq(
        params=params_chirplet,
        lc=lc,
        wc=wc,
        NF_min=NF_min,
        NF_max=NF_max,
        freeze_limits=freeze_limits,
        n_pad_F=10,  # optional, default is 10
    )

    assert_allclose(linear_freq.intrinsic_waveform.F, chirplet_freq.intrinsic_waveform.F, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.intrinsic_waveform.TFp, chirplet_freq.intrinsic_waveform.TFp, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.intrinsic_waveform.TF, chirplet_freq.intrinsic_waveform.TF, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.intrinsic_waveform.PF, chirplet_freq.intrinsic_waveform.PF, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.intrinsic_waveform.AF, chirplet_freq.intrinsic_waveform.AF, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.kdotx, chirplet_freq.kdotx, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.tdi_waveform.TF, chirplet_freq.tdi_waveform.TF, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.tdi_waveform.TFp, chirplet_freq.tdi_waveform.TFp, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.tdi_waveform.PF, chirplet_freq.tdi_waveform.PF, atol=1.e-20, rtol=1.e-10)
    assert_allclose(linear_freq.tdi_waveform.AF, chirplet_freq.tdi_waveform.AF, atol=1.e-19, rtol=1.e-10)
