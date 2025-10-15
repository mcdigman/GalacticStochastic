"""Test internal agreement between TaylorF2 waveform variants"""
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tomllib

from LisaWaveformTools.binary_params_manager import M_SUN_SEC, PC_M, BinaryIntrinsicParamsManager
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from LisaWaveformTools.taylorf2_freq_source import TaylorF2WaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.wdm_config import get_wavelet_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_taylorf2_agreement() -> None:
    """Test agreement between TaylorF2 waveform and LinearFrequency waveform for low-mass, low-fdot sources."""
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

    # Create TaylorF2 waveform source
    taylorf2_waveform = TaylorF2WaveformFreq(
        params=source_params,
        lc=lc,
        nf_lim_absolute=PixelGenericRange(1, wc.Nf, wc.DF, 0.0),
        freeze_limits=1,
        t_obs=31536000.0  # Observation time of 1 year in seconds
    )

    print(taylorf2_waveform.intrinsic_waveform)
