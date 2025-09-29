"""Objects that store the parametrization for binary sources."""

from __future__ import annotations

from typing import NamedTuple


class ExtrinsicParams(NamedTuple):
    """
    Store the extrinsic parameters common to most detector intrinsic_waveform models.

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
    costh: float
    phi: float
    cosi: float
    psi: float


class SourceParams(NamedTuple):
    """
    Store both the intrinsic and extrinsic parameters for a source.

    Parameters
    ----------
    intrinsic: NamedTuple
        The intrinsic parameters for the particular source class, e.g. LinearFrequencyIntrinsicParams
    extrinsic: LisaWaveformTools.source_params.ExtrinsicParams
        The extrinsic parameters for the source
    """
    intrinsic: NamedTuple
    extrinsic: ExtrinsicParams
