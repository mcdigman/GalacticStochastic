"""Abstract class for stationary wave approximation-based TDI waveform models."""
from abc import ABC, abstractmethod
from collections import namedtuple

from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wdm_config import WDMWaveletConstants

ExtrinsicParams = namedtuple('ExtrinsicParams', ['costh', 'phi', 'cosi', 'psi'])
ExtrinsicParams.__doc__ = """
Store the extrinsic parameters common to most detector waveform models.

Parameters
----------
costh : float
    Cosine of the source's ecliptic colatitude
phi : float
    Source's ecliptic longitude in radians
cosi : float
    Cosine of the source's inclination angle
psi : float
    Source polarization angle in radians
"""

SourceParams = namedtuple('SourceParams', ['intrinsic', 'extrinsic'])

SourceParams.__doc__ = """
Store both the intrinsic and extrinsic parameters for a source.

Parameters
----------
intrinsic: namedtuple
    The intrinsic parameters for the particular source class, e.g. LinearFrequencyIntrinsicParams
extrinsic: ExtrinsicParams
    The extrinsic parameters for the source
"""


class StationarySourceWaveform(ABC):
    """Abstract base class for waveform models to be used in the stationary wave approximation."""

    @abstractmethod
    def __init__(
        self,
        params: SourceParams,
        nt_lim_waveform: PixelTimeRange,
        lc: LISAConstants,
        wc: WDMWaveletConstants,
    ) -> None:
        """
        Initialize the waveform object for use in the stationary wave approximation.

        Parameters
        ----------
        params : SourceParams
            Model-specific parameters (intrinsic and extrinsic) for the source.
        nt_lim_waveform : PixelTimeRange
            Object describing the range of waveform time samples to generate.
        lc : LISAConstants
            LISA or detector configuration parameters required for the waveform.
        wc : WDMWaveletConstants
            Configuration parameters of the wavelet transform parameters.
        """

    @abstractmethod
    def update_params(self, params: SourceParams) -> None:
        """
        Update the waveform to match a new set of model parameters.

        Typically should call both _update_intrinsic and _update_extrinsic.

        Parameters
        ----------
        params : namedtuple
            Updated parameters (intrinsic and extrinsic) for the source.
        """

    @abstractmethod
    def _update_intrinsic(self) -> None:
        """
        Update source waveform quantities specific to the stored intrinsic source parameters.

        Compute e.g. amplitude, phase, frequency, and frequency derivative arrays for the source signal.
        To be called by update_params.
        """

    @abstractmethod
    def _update_extrinsic(self) -> None:
        """
        Compute the TDI waveform channels specific to the stored extrinsic source parameters.

        Typical extrinsic parameters include the ecliptic longitude, ecliptic colatitude, inclination, and polarization.
        To be called by update_params.
        """
