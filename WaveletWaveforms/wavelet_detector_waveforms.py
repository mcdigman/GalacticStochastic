"""subroutines for running lisa binary monte carlo search"""

from LisaWaveformTools.linear_frequency_source import LinearFrequencyWaveformTime
from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms.coefficientsWDM_time_helpers import (
    WaveletTaylorTimeCoeffs,
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange, SparseWaveletWaveform
from WaveletWaveforms.taylor_wdm_funcs import wavemaket_multi_inplace
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class BinaryWaveletAmpFreqDT:
    """class to store a sparse binary wavelet and update for search"""

    def __init__(self, params, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelTimeRange) -> None:
        """Construct a binary wavelet object, if NMF_use is not -1 it overrides the default"""
        self.wc = wc
        self.lc = lc
        self.nt_lim_waveform = nt_lim_waveform

        self.params = params

        # interpolation for wavelet taylor expansion
        self.taylor_time_table: WaveletTaylorTimeCoeffs = get_taylor_table_time(
            self.wc, cache_mode='skip', output_mode='skip',
        )

        # get the waveform in frequency space
        self.fwt = LinearFrequencyWaveformTime(
            self.params, self.nt_lim_waveform, self.lc, self.wc,
        )

        # get a blank waveform in the sparse wavelet domain
        # when consistent is set it will be correct
        self.wavelet_waveform: SparseWaveletWaveform = get_empty_sparse_taylor_time_waveform(self.lc.nc_waveform, wc)

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
            self.nt_lim_waveform,
            self.wc,
            self.taylor_time_table,
            force_nulls=False,
        )
        self.consistent = True

    def get_unsorted_coeffs(self) -> SparseWaveletWaveform:
        """Get coefficients in the order they are generated"""
        if not self.consistent:
            self.update_params(self.params)
        return self.wavelet_waveform
