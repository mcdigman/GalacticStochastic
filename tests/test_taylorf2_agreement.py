"""Test internal agreement between TaylorF2 waveform variants"""
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose

from LisaWaveformTools.binary_params_manager import M_SUN_SEC, PC_M, BinaryIntrinsicParamsManager
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from LisaWaveformTools.taylorf2_freq_source import TaylorF2WaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.wdm_config import get_wavelet_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.mark.parametrize('model1_idx', [0, 1])
@pytest.mark.parametrize('model2_idx', [1, 2])
@pytest.mark.parametrize('tc_mode', [0, 1])
def test_taylorf2_agreement_zero(model1_idx: int, model2_idx: int, tc_mode: int) -> None:
    """Test agreement between TaylorF2 waveform and LinearFrequency waveform for low-mass, low-fdot sources."""
    if model1_idx >= model2_idx:
        # Nothing to test, or redundant with order
        return
    model_list = ('taylorf2_basic', 'taylorf2_eccentric', 'taylorf2_aligned')
    model1 = model_list[model1_idx]
    model2 = model_list[model2_idx]
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)

    # Define parameters for a low-mass, low-fdot binary system
    intrinsic_params_packed: NDArray[np.floating] = np.array([
        np.log(1.e9 * PC_M),  # Log luminosity distance in meters
        60.0 * M_SUN_SEC,  # Total mass in seconds
        26.0 * M_SUN_SEC,  # Chirp mass in seconds
        1e-3,              # Initial frequency in Hz
        1.0e9,             # Coalescence time in seconds
        0.3,               # Phase at coalescence
        0.0,               # Normalize postnewtonian spin parameter
        0.0,               # Antisymmetric component of aligned spin
        0.0,               # Precessing spin
        0.0,               # Initial eccentricity
    ])
    intrinsic_params_manager = BinaryIntrinsicParamsManager(intrinsic_params_packed)
    intrinsic = intrinsic_params_manager.params
    extrinsic = ExtrinsicParams(
        costh=0.1, phi=0.1, cosi=0.2, psi=0.3
    )

    source_params = SourceParams(intrinsic=intrinsic, extrinsic=extrinsic)

    amplitude_pn_mode = 0
    include_pn_ss3 = 0
    t_obs = wc.Tobs

    # Create TaylorF2 waveform source
    taylorf2_waveform1 = TaylorF2WaveformFreq(
        params=source_params,
        lc=lc,
        nf_lim_absolute=PixelGenericRange(1, wc.Nf, wc.DF, 0.0),
        freeze_limits=1,
        t_obs=t_obs,
        model_select=model1,
        amplitude_pn_mode=amplitude_pn_mode,
        include_pn_ss3=include_pn_ss3,
        tc_mode=tc_mode,
    )
    taylorf2_waveform2 = TaylorF2WaveformFreq(
        params=source_params,
        lc=lc,
        nf_lim_absolute=PixelGenericRange(1, wc.Nf, wc.DF, 0.0),
        freeze_limits=1,
        t_obs=t_obs,
        model_select=model2,
        amplitude_pn_mode=amplitude_pn_mode,
        include_pn_ss3=include_pn_ss3,
        tc_mode=tc_mode,
    )

    waveform1 = taylorf2_waveform1.intrinsic_waveform
    waveform2 = taylorf2_waveform2.intrinsic_waveform
    assert_allclose(waveform1.F, waveform2.F, atol=1.e-100, rtol=1.e-10)
    assert_allclose(waveform1.TF, waveform2.TF, atol=1.e-100, rtol=1.e-10)
    assert_allclose(waveform1.TFp, waveform2.TFp, atol=1.e-100, rtol=1.e-10)
    assert_allclose(waveform1.PF, waveform2.PF, atol=1.e-100, rtol=1.e-10)
    assert_allclose(waveform1.AF, waveform2.AF, atol=1.e-100, rtol=1.e-10)
    assert_allclose(taylorf2_waveform1.TTRef, taylorf2_waveform2.TTRef, atol=1.e-100, rtol=1.e-10)
