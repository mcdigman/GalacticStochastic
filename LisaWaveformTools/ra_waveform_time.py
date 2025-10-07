"""Time domain TDI waveform generation using the rigid adiabatic approximation."""


import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from LisaWaveformTools.algebra_tools import gradient_uniform_inplace, stabilized_gradient_uniform_inplace
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.spacecraft_objects import AntennaResponseChannels, EdgeRiseModel
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformGeneric, StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange


@njit()
def phase_wrap_helper(
    AET_waveform: StationaryWaveformTime, waveform: StationaryWaveformTime, wrap_thresh: float = 6.0,
) -> None:
    """
    Wrap the tdi perturbations to the phases by appropriate factors of 2 pi.

    Parameters
    ----------
    AET_waveform: StationaryWaveformTime
        Object containing the TDI intrinsic_waveform, with shape (nc_channel, n_t).
    waveform: StationaryWaveformTime
        Object containing the intrinsic intrinsic_waveform, with shape (n_t)
    wrap_thresh: float
        Threshold above which to consider the phase large enough to wrap by 2 pi.

    Returns
    -------
    None
        Results are stored in place in the PT attribute of tdi_waveform.

    """
    AET_PT: NDArray[np.floating] = AET_waveform.PT
    PT: NDArray[np.floating] = waveform.PT

    # validate the shapes of the inputs
    assert len(AET_PT.shape) == 2
    assert len(PT.shape) == 1
    assert PT.shape[0] == AET_PT.shape[1]
    assert PT.shape[0] > 0

    # the number of channels we need to handle phases for
    nc_loc: int = AET_PT.shape[0]
    n_t: int = AET_PT.shape[1]

    for itrc in range(nc_loc):
        # Get the starting perturbation to the phase
        p_old: float = (AET_PT[itrc, 0] - PT[0]) % (2 * np.pi)
        j: float = 0.0
        for n in range(n_t):
            # Isolate just the perturbation to the intrinsic phase
            p: float = (AET_PT[itrc, n] - PT[n]) % (2 * np.pi)

            # If the phase has increased or decreased more than 6 (<~2 pi)
            # try absorbing that change into the reported phase permanently,
            # as it is likely represents wrapping.
            # In testing 6 is a decent choice of the cutoff.
            # It might be possible to detect wrapping by multiple factors of 2 pi
            # using the analytic part of the perturbation in AET_FTs.
            # Note that wrapping due to the linear part of the frequency is assumed
            # to have already been done in the original intrinsic_waveform generation.
            if p - p_old > wrap_thresh:
                j -= 2 * np.pi
            elif p_old - p > wrap_thresh:
                j += 2 * np.pi

            # Add the total amount of previous wrapping we have computed
            AET_PT[itrc, n] += j

            # Store the current phase perturbation
            p_old = p


@njit()
def spacecraft_channel_deriv_helper(spacecraft_channels: AntennaResponseChannels, dx: float) -> None:
    """Get the derivatives of the RR and II components of the TDI and store them in dRR and dII.

    Parameters
    ----------
    spacecraft_channels: SpacecraftChannels
        Object containing the spacecraft channels with RR, II, dRR, and dII attributes,
        with RR and II already computed.
    nt_lim: PixelGenericRange
        Range of time indices to compute the derivatives over and time spacing of uniform samples

    Returns
    -------
    None
        Results are stored in place in dRR and dII attributes of sc_channels

    """
    # get and store the derivatives
    gradient_uniform_inplace(spacecraft_channels.RR, spacecraft_channels.dRR, dx)
    gradient_uniform_inplace(spacecraft_channels.II, spacecraft_channels.dII, dx)


@njit(fastmath=True)
def edge_rise_multiplier_helper(t: float, er: EdgeRiseModel, lc: LISAConstants) -> float:
    """Apply the edge rise model to the amplitude of a waveform."""
    x: float = 1.0
    Tstart: float = er.Tstart
    Tend: float = er.Tend
    t_rise: float = lc.t_rise
    if lc.rise_mode == 0:
        if Tstart < t < Tstart + t_rise:
            x = 0.5 * (1.0 - np.cos(np.pi * (t - Tstart) / t_rise))
        if (Tend - t_rise) < t < Tend:
            x = 0.5 * (1.0 - np.cos(np.pi * (t - Tend) / t_rise))
    elif lc.rise_mode == 0:
        if t > Tend:
            x = 0.0
        if t < Tstart:
            x = 0.0
    elif lc.rise_mode == 2:
        if t > Tend:
            x = 0.0
    elif lc.rise_mode == 3:
        pass
    else:
        msg = 'rise_mode must be 0, 1, or 2'
        raise NotImplementedError(msg)
    return x


@njit(fastmath=True)
def apply_edge_rise_helper(t: NDArray[np.floating], A: NDArray[np.floating], er: EdgeRiseModel, lc: LISAConstants, nx_lim: PixelGenericRange) -> None:
    """Apply edge rise to the amplitude of a waveform in place."""
    if lc.rise_mode == 3:
        pass
    else:
        assert 0 <= nx_lim.nx_min < nx_lim.nx_max <= t.size
        assert len(A.shape) == 2
        assert len(t.shape) == 1
        assert A.shape[1] == t.shape[0]
        for n in range(nx_lim.nx_min, nx_lim.nx_max):
            x = edge_rise_multiplier_helper(t[n], er, lc)
            A[:, n] *= x


@njit(fastmath=True, parallel=True)
def amp_phase_loop_helper(F: NDArray[np.floating], T: NDArray[np.floating], waveform: StationaryWaveformGeneric, AET_waveform: StationaryWaveformGeneric, spacecraft_channels: AntennaResponseChannels, lc: LISAConstants, nx_lim: PixelGenericRange) -> None:
    """Compute TDI (Time Delay Interferometry) modifications for the amplitude, phase, and frequency of a intrinsic_waveform.

    This function perturbs the intrinsic amplitude, phase, and frequency of a generic intrinsic_waveform
    from the stationary wave approximation to compute the TDI intrinsic_waveform in the stationary wave approximation.
    Uses the real (RR) and imaginary (II) components already computed and stored in sc_channels
    using the rigid adiabatic approximation.
    Results are stored in-place in the tdi_waveform parameter. The function handles the general case
    and a subset of special cases where the real (RR) and imaginary (II) components are zero.

    Parameters
    ----------
    F: NDArray[float]
        Array of frequencies corresponding to the waveform samples
    T: NDArray[float]
        Array of times corresponding to the waveform samples
    spacecraft_channels : SpacecraftChannels
        Object containing the real (RR) and imaginary (II) components of the spacecraft
        response, along with their derivatives (dRR, dII)
    AET_waveform : StationaryWaveformGeneric
        Target intrinsic_waveform object where the modified TDI amplitude and phase will be stored.
        Modified in-place with computed values
    waveform : StationaryWaveformGeneric
        Input intrinsic_waveform object containing the original phase (PT) and amplitude (AT) data
    lc : LISAConstants
        LISA constellation constants containing the strain frequency (fstr)

    Notes
    -----
    - The function assumes positive frequencies
    - When RR and II are both near zero, there can be a step function in the phase
      corresponding to a Dirac delta function in frequency, which is not currently
      represented in the frequency or derivative
    - The phase perturbation is computed using arctan2 and is kept within [0, 2π]
    - The frequency derivative calculation ignores potential delta functions that
      can occur from step functions in the phase when RR or II pass through zero
    - All input arrays must be properly sized:
      - sc_channels.RR shape: (nc_channel, n_t)
      - intrinsic_waveform.PT shape: (n_t,)
      - tdi_waveform.PT shape: (nc_channel, n_t)

    Returns
    -------
    None
        Results are stored in-place in tdi_waveform

    """
    # Note that if RR and II are both very near zero, there can be a step function
    # in the phase that would correspond to a dirac delta function in the frequency,
    # that currently is not represented in the frequency or derivative at all,
    # because the formula used for the frequency derivative does not include a delta function.
    # It appears naturally in the phase due to the two argument arctangent.
    # (the limit of the two argument arctangent as one argument approaches zero
    # is one possible representation of a step function)
    # I think this may be the best behavior given the current use of passing to
    # Wavelet domain taylor expansion methods given typically sparse sampling for wavelets
    # but I am not sure if including such delta functions would be better in other use cases.
    # The code implicitly assumes postive frequencies.

    nc_channel: int = spacecraft_channels.RR.shape[0]
    AET_P: NDArray[np.floating] = AET_waveform.P
    AET_A: NDArray[np.floating] = AET_waveform.A
    AET_X: NDArray[np.floating] = AET_waveform.X

    wave_P: NDArray[np.floating] = waveform.P
    wave_A: NDArray[np.floating] = waveform.A
    wave_X: NDArray[np.floating] = waveform.X

    RR_loc: NDArray[np.floating] = spacecraft_channels.RR
    II_loc: NDArray[np.floating] = spacecraft_channels.II
    dRR_loc: NDArray[np.floating] = spacecraft_channels.dRR
    dII_loc: NDArray[np.floating] = spacecraft_channels.dII

    assert len(AET_P.shape) == 2
    assert len(F.shape) == 1
    assert len(RR_loc.shape) == 2
    assert RR_loc.shape == II_loc.shape == dRR_loc.shape == dII_loc.shape
    assert AET_P.shape == AET_A.shape == AET_X.shape
    assert AET_P.shape[0] == nc_channel
    assert RR_loc.shape[0] == nc_channel
    assert wave_X.shape == wave_P.shape == wave_A.shape
    assert F.shape == T.shape == wave_P.shape
    assert RR_loc.shape[-1] == F.shape[0]
    assert AET_P.shape[-1] == F.shape[0]
    assert 0 <= nx_lim.nx_min <= nx_lim.nx_max <= F.shape[0]

    for n in prange(nx_lim.nx_min, nx_lim.nx_max):
        f_on_f: float = F[n] / lc.fstr

        # including TDI + fractional frequency modifiers
        Ampx: float = wave_A[n] * (8 * f_on_f * np.sin(f_on_f))
        for itrc in range(nc_channel):
            RR: float = RR_loc[itrc, n]
            II: float = II_loc[itrc, n]
            dRR: float = dRR_loc[itrc, n]
            dII: float = dII_loc[itrc, n]

            if RR == 0.0 and II == 0.0:
                # Handle zero denominator phase.
                p: float = 0.0

                # Handle zero denominator FT without a delta function.
                # Note that the second derivative could be more complicated here,
                # But we ignore that for now and just take a numerical derivative.
                AET_X[itrc, n] = wave_X[n]
            elif II * dRR == RR * dII:
                # Zero numerator phase is the same as general case.
                p = np.arctan2(II, RR) % (2 * np.pi)

                # Handle zero numerator FT without a delta function
                # May improve numerical stability if denominator is also close to zero
                AET_X[itrc, n] = wave_X[n]
            else:
                # General case of phase.
                p = np.arctan2(II, RR) % (2 * np.pi)

                # Handle general case, with the analytic derivative of the phase.
                # dRR and dII are currently computed through a numerical derivative,
                # But could be constructed analytically.
                # Ignores delta functions in FT that can happen when RR or II pass through 0.
                # The results change slightly if the compiler has the fused multiply and add enabled,
                # e.g. fastmath = {'contract'} for the LLVM compiler
                AET_X[itrc, n] = wave_X[n] - (II * dRR - RR * dII) / (RR**2 + II**2) * 1 / (2 * np.pi)

            # Set the amplitude
            AET_A[itrc, n] = Ampx * np.sqrt(RR**2 + II**2)
            # Set the phase, including the input base phase and the perturbation from this iteration,
            AET_P[itrc, n] = wave_P[n] + p


@njit()
def get_time_tdi_amp_phase_helper(
    spacecraft_channels: AntennaResponseChannels,
    AET_waveform: StationaryWaveformTime,
    waveform: StationaryWaveformTime,
    lc: LISAConstants,
    er: EdgeRiseModel,
    nt_lim: PixelGenericRange,
) -> None:
    """Compute TDI (Time Delay Interferometry) modifications for the amplitude, phase, and frequency of a intrinsic_waveform.

    This function perturbs the intrinsic amplitude, phase, and frequency of a time domain intrinsic_waveform
    from the stationary wave approximation to compute the TDI intrinsic_waveform in the stationary wave approximation.
    Uses the real (RR) and imaginary (II) components already computed and stored in sc_channels
    using the rigid adiabatic approximation.
    Results are stored in-place in the tdi_waveform parameter. The function handles the general case
    and a subset of special cases where the real (RR) and imaginary (II) components are zero.

    Parameters
    ----------
    spacecraft_channels : SpacecraftChannels
        Object containing the real (RR) and imaginary (II) components of the spacecraft
        response, along with their derivatives (dRR, dII)
    AET_waveform : StationaryWaveformTime
        Target intrinsic_waveform object where the modified TDI amplitude and phase will be stored.
        Modified in-place with computed values
    waveform : StationaryWaveformTime
        Input intrinsic_waveform object containing the original phase (PT) and amplitude (AT) data
    lc : LISAConstants
        LISA constellation constants containing the strain frequency (fstr)
    er: EdgeRiseModel
        Edge rise model containing parameters needed to apply edge rise to the amplitude
    nt_lim: PixelGenericRange
        Range of time indices to compute the TDI modifications over

    Notes
    -----
    - The function assumes positive frequencies
    - When RR and II are both near zero, there can be a step function in the phase
      corresponding to a Dirac delta function in frequency, which is not currently
      represented in the frequency or derivative
    - The phase perturbation is computed using arctan2 and is kept within [0, 2π]
    - The frequency derivative calculation ignores potential delta functions that
      can occur from step functions in the phase when RR or II pass through zero
    - All input arrays must be properly sized:
      - sc_channels.RR shape: (nc_channel, n_t)
      - intrinsic_waveform.PT shape: (n_t,)
      - tdi_waveform.PT shape: (nc_channel, n_t)

    Returns
    -------
    None
        Results are stored in-place in tdi_waveform

    """
    # Note that if RR and II are both very near zero, there can be a step function
    # in the phase that would correspond to a dirac delta function in the frequency,
    # that currently is not represented in the frequency or derivative at all,
    # because the formula used for the frequency derivative does not include a delta function.
    # It appears naturally in the phase due to the two argument arctangent.
    # (the limit of the two argument arctangent as one argument approaches zero
    # is one possible representation of a step function)
    # I think this may be the best behavior given the current use of passing to
    # Wavelet domain taylor expansion methods given typically sparse sampling for wavelets
    # but I am not sure if including such delta functions would be better in other use cases.
    # The code implicitly assumes postive frequencies.

    nc_channel = spacecraft_channels.RR.shape[0]

    n_t = AET_waveform.PT.shape[1]

    # some input size validation
    assert spacecraft_channels.RR.shape[1] == n_t
    assert waveform.PT.shape[0] == n_t
    assert AET_waveform.PT.shape[0] == nc_channel

    AET_AT = AET_waveform.AT
    AET_PT = AET_waveform.PT
    AET_FT = AET_waveform.FT

    AET_waveform_generic = StationaryWaveformGeneric(AET_waveform.T, AET_PT, AET_FT, AET_waveform.FTd, AET_AT)
    waveform_generic = StationaryWaveformGeneric(waveform.T, waveform.PT, waveform.FT, waveform.FTd, waveform.AT)
    amp_phase_loop_helper(waveform.FT, waveform.T, waveform_generic, AET_waveform_generic, spacecraft_channels, lc, nt_lim)
    apply_edge_rise_helper(waveform.T, AET_waveform.AT, er, lc, nt_lim)


@njit()
def get_time_tdi_amp_phase(
    spacecraft_channels: AntennaResponseChannels,
    AET_waveform: StationaryWaveformTime,
    waveform: StationaryWaveformTime,
    lc: LISAConstants,
    er: EdgeRiseModel,
    nt_lim: PixelGenericRange,
    phase_wrap_mode: int = 0,
) -> None:
    """Compute TDI modifications for the amplitude, phase, frequency, and frequency derivative of a intrinsic_waveform.

    This function perturbs the intrinsic amplitude, phase, frequency, and frequency derivative of a time domain intrinsic_waveform
    from the stationary wave approximation to compute the TDI intrinsic_waveform in the stationary wave approximation.
    Uses the real (RR) and imaginary (II) components already computed and stored in sc_channels
    using the rigid adiabatic approximation.
    Results are stored in-place in the tdi_waveform parameter. The function handles the general case
    and a subset of special cases where the real (RR) and imaginary (II) components are zero.

    Parameters
    ----------
    spacecraft_channels : SpacecraftChannels
        Object containing the real (RR) and imaginary (II) components of the spacecraft
        response, along with their derivatives (dRR, dII)
    AET_waveform : StationaryWaveformTime
        Target intrinsic_waveform object where the modified TDI amplitude and phase will be stored.
        Modified in-place with computed values
    waveform : StationaryWaveformTime
        Input intrinsic_waveform object containing the original phase (PT) and amplitude (AT) data
    lc : LISAConstants
        LISA constellation constants containing the strain frequency (fstr)
    er: EdgeRiseModel
        Edge rise model containing parameters needed to apply edge rise to the amplitude
    nt_lim: PixelGenericRange
        Range of time indices to compute the TDI modifications over
    phase_wrap_mode: int
        Select which model to use for phase wrapping: current options are 0 (no wrapping)
        and 1 (using phase_wrap_helper).
        Likelihood, snr, and intrinsic_waveform calculations generally should be ok with no wrapping,
        but some applications like semi-analytic fisher matrix calculation might need it.

    Notes
    -----
    - The function assumes positive frequencies
    - When RR and II are both near zero, there can be a step function in the phase
      corresponding to a Dirac delta function in frequency, which is not currently
      represented in the frequency or derivative
    - The phase perturbation is computed using arctan2 and is kept within [0, 2π]
    - The frequency derivative calculation ignores potential delta functions that
      can occur from step functions in the phase when RR or II pass through zero
    - All input arrays must be properly sized:
      - sc_channels.RR shape: (nc_channel, n_t)
      - intrinsic_waveform.PT shape: (n_t,)
      - tdi_waveform.PT shape: (nc_channel, n_t)

    Returns
    -------
    None
        Results are stored in-place in tdi_waveform

    """
    # get and store dRR and dII in the object
    spacecraft_channel_deriv_helper(spacecraft_channels, nt_lim.dx)

    # get the tdi perturbations to the amplitude and phase
    get_time_tdi_amp_phase_helper(spacecraft_channels, AET_waveform, waveform, lc, er, nt_lim)

    if phase_wrap_mode == 1:
        # wrap the phases using phase_wrap_helper
        phase_wrap_helper(AET_waveform, waveform)
    elif phase_wrap_mode == 0:
        # many applications do not need the phases to be wrapped properly
        pass
    else:
        msg = 'Unrecognized option for phase_wrap_mode=' + str(phase_wrap_mode)
        raise NotImplementedError(msg)

    # Get the frequency derivative using a numerical derivative, with offsets for
    # improved numerical accuracy. Because of the behavior near RR or II=0,
    # the numerical derivative may have better practical accuracy than
    # inserting an analytic result, at least without some additional numerical stabilizers.
    stabilized_gradient_uniform_inplace(waveform.FT, waveform.FTd, AET_waveform.FT, AET_waveform.FTd, nt_lim.dx)
