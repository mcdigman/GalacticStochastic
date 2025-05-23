"""subroutines for running lisa binary monte carlo search"""

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_time import BinaryTimeWaveformAmpFreqD, StationaryWaveformTime
from WaveletWaveforms.coefficientsWDM_time_helpers import (
    SparseTaylorWaveform,
    WaveletTaylorTimeCoeffs,
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.taylor_wdm_funcs import wavemaket_multi_inplace
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class BinaryWaveletAmpFreqDT:
    """class to store a sparse binary wavelet and update for search"""

    def __init__(self, params, wc: WDMWaveletConstants, lc: LISAConstants, nt_min=0, nt_max=-1) -> None:
        """Construct a binary wavelet object, if NMF_use is not -1 it overrides the default"""
        self.wc = wc
        self.lc = lc
        if nt_max == -1:
            self.nt_max = self.wc.Nt
        else:
            self.nt_max = nt_max
        self.nt_min = nt_min

        self.params = params

        # interpolation for wavelet taylor expansion
        self.taylor_time_table: WaveletTaylorTimeCoeffs = get_taylor_table_time(
            self.wc, cache_mode='skip', output_mode='skip'
        )

        # get the waveform in frequency space
        self.fwt = BinaryTimeWaveformAmpFreqD(
            self.params, self.nt_min, self.nt_max, self.lc, self.wc, self.lc.nc_waveform
        )

        # get a blank waveform in the sparse wavelet domain
        # when consistent is set it will be correct
        self.wavelet_waveform: StationaryWaveformTime = get_empty_sparse_taylor_time_waveform(self.lc.nc_waveform, wc)

        self.consistent = False

        # initialize to input parameters
        self.update_params(params)

    def update_params(self, params_in) -> None:
        """Update the internal wavelet representation to match the input parameters"""
        self.consistent = False
        self.params = params_in

        self.fwt.update_params(params_in)

        wavemaket_multi_inplace(
            self.wavelet_waveform,
            self.fwt.AET_waveform,
            self.nt_min,
            self.nt_max,
            self.wc,
            self.taylor_time_table,
            force_nulls=False,
        )
        self.consistent = True

    def get_unsorted_coeffs(self) -> SparseTaylorWaveform:
        """Get coefficients in the order they are generated"""
        if not self.consistent:
            self.update_params(self.params)
        return self.wavelet_waveform
