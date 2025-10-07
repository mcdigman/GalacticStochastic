"""TaylorF2 waveform model in the frequency domain."""
from typing import override

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from LisaWaveformTools.stationary_freq_source import StationarySourceWaveformFreq
from LisaWaveformTools.taylorf2_helpers import TaylorF2_aligned_inplace, TaylorF2AlignedSpinParams
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange


class StationaryTaylorF2WaveformFreq(StationarySourceWaveformFreq[TaylorF2AlignedSpinParams, ExtrinsicParams]):
    """Store a taylorf2 waveform in the frequency domain and update it."""

    @override
    def __init__(self, params: SourceParams, lc: LISAConstants, nf_lim_absolute: PixelGenericRange, freeze_limits: int, t_obs: float,
                 n_pad_F: int = 10, *, mf_taylor_anchor: float = 1.e-5) -> None:
        """Construct a waveform for a binary using the TaylorF2 model."""
        self.mf_taylor_anchor: float = mf_taylor_anchor
        super().__init__(params, lc, nf_lim_absolute, freeze_limits, t_obs, n_pad_F=n_pad_F)

    @override
    def _update_intrinsic(self) -> None:
        """Update the waveform to match intrinsic parameters."""
        # TODO check consistency of sign and factor of 2 on phic
        if not isinstance(self.params.intrinsic, TaylorF2AlignedSpinParams):
            msg = 'Intrinsic parameters must be of type TaylorF2AlignedSpinParams.'
            raise TypeError(msg)

        tc = self.params.intrinsic.tc  # -self.delta_tm #TODO make sure this is self consistent way to handle shifting merger time between frames

        # TODO proper selectable waveform model everywhere it is used

        self.TTRef: float = TaylorF2_aligned_inplace(self.intrinsic_waveform, self.params.intrinsic, self.nf_lim, include_phenom_amp=True, include_pn_SS3=False, tc_mode=True, t_offset=tc)
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
