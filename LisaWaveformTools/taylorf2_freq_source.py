"""TaylorF2 waveform model in the frequency domain."""

from typing import override

import numpy as np

from LisaWaveformTools.binary_params_manager import BinaryIntrinsicParams, BinaryIntrinsicParamsManager
from LisaWaveformTools.imrphenomd_waveform import IMRPhenDAmpPhaseFI
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.source_params import AbstractIntrinsicParamsManager, ExtrinsicParams, SourceParams
from LisaWaveformTools.stationary_freq_source import StationarySourceWaveformFreq
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq
from LisaWaveformTools.taylorf2_helpers import TaylorF2_aligned_inplace, TaylorF2_eccentric_inplace, TaylorF2_inplace
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange


def taylorf2_intrinsic_freq(
    waveform: StationaryWaveformFreq, params_intrinsic: BinaryIntrinsicParams, nf_lim: PixelGenericRange, *, model_select: str = 'taylorf2_aligned', amplitude_pn_mode: int = 2, t_offset: float = 0., tc_mode: int = 0, include_pn_ss3: int = 0) -> float:
    """Get the frequency domain intrinsic waveform for a binary inspiral in the TaylorF2 approximation."""
    if model_select == 'taylorf2_basic':
        return TaylorF2_inplace(waveform, params_intrinsic, nf_lim, amplitude_pn_mode=amplitude_pn_mode, t_offset=t_offset, tc_mode=tc_mode)
    if model_select == 'taylorf2_eccentric':
        return TaylorF2_eccentric_inplace(waveform, params_intrinsic, nf_lim, amplitude_pn_mode=amplitude_pn_mode, t_offset=t_offset, tc_mode=tc_mode)
    if model_select == 'taylorf2_aligned':
        return TaylorF2_aligned_inplace(waveform, params_intrinsic, nf_lim, amplitude_pn_mode=amplitude_pn_mode, include_pn_ss3=include_pn_ss3, t_offset=t_offset, tc_mode=tc_mode)
    if model_select == 'imrphenomd':
        # TODO respect tc_mode
        amp0: float = 2. * float(np.sqrt(5. / (64. * np.pi))) * params_intrinsic.mass_total_detector_sec**2 / params_intrinsic.luminosity_distance_m
        MfRef_in = params_intrinsic.frequency_i_hz * params_intrinsic.mass_total_detector_sec
        _, imr_params = IMRPhenDAmpPhaseFI(waveform, params_intrinsic, nf_lim, MfRef_in=MfRef_in, phi0=-params_intrinsic.phase_c + np.pi / 16., amp_mult=amp0, imr_default_t=0)
        return imr_params.TTRef
    msg = f'Unknown model_select value: {model_select}'
    raise ValueError(msg)


class TaylorF2WaveformFreq(StationarySourceWaveformFreq[BinaryIntrinsicParams, ExtrinsicParams]):
    """Store a taylorf2 waveform in the frequency domain and update it."""

    @override
    def __init__(
        self,
        params: SourceParams,
        lc: LISAConstants,
        nf_lim_absolute: PixelGenericRange,
        freeze_limits: int,
        t_obs: float,
        n_pad_F: int = 10,
        *,
        mf_taylor_anchor: float = 1.0e-5,
        model_select: str = 'taylorf2_aligned',
        amplitude_pn_mode: int = 2,
        include_pn_ss3: int = 0,
        tc_mode: int = 0,
    ) -> None:
        """Construct a waveform for a binary using the TaylorF2 model."""
        self.mf_taylor_anchor: float = mf_taylor_anchor
        self.model_select = model_select
        self.amplitude_pn_mode = amplitude_pn_mode
        self.include_pn_ss3 = include_pn_ss3
        self.tc_mode = tc_mode
        super().__init__(params, lc, nf_lim_absolute, freeze_limits, t_obs, n_pad_F=n_pad_F)

    @override
    def _update_intrinsic(self) -> None:
        """Update the waveform to match intrinsic parameters."""
        # TODO check consistency of sign and factor of 2 on phic
        if not isinstance(self.params.intrinsic, BinaryIntrinsicParams):
            msg = 'Intrinsic parameters must be of type BinaryIntrinsic.'
            raise TypeError(msg)

        tc = (
            self.params.intrinsic.time_c_sec
        )  # -self.delta_tm #TODO make sure this is self consistent way to handle shifting merger time between frames

        # TODO proper selectable waveform model everywhere it is used

        self.TTRef: float = taylorf2_intrinsic_freq(self._intrinsic_waveform, self.params.intrinsic, self.nf_lim, model_select=self.model_select, amplitude_pn_mode=self.amplitude_pn_mode, include_pn_ss3=self.include_pn_ss3, tc_mode=self.tc_mode, t_offset=tc)
        itrFCut_new = self.itrFCut
        # TODO set itrFCut_new if we need it

        # TODO check phasing convention
        if itrFCut_new != self.itrFCut:
            # have to force a bounds update if itrfcut changes
            # TODO check to make sure this doesn't break anything
            self.itrFCut: int = itrFCut_new
            self._update_bounds()
        # TODO investigate it blow up difference at f->0 can be mitigated
        self._consistent_intrinsic: bool = True

    def _create_intrinsic_params_manager(self, params_intrinsic: BinaryIntrinsicParams) -> AbstractIntrinsicParamsManager[BinaryIntrinsicParams]:
        return BinaryIntrinsicParamsManager(params_intrinsic)
