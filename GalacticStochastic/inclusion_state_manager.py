"""Class to store information about the binaries in the galactic background"""
from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, override
from warnings import warn

import h5py
import numpy as np
from numpy.testing import assert_allclose

import GalacticStochastic.global_const as gc
from GalacticStochastic.state_manager import StateManager
from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams, LinearFrequencyWaveletWaveformTime
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from GalacticStochastic.iteration_config import IterationConfig
    from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
    from GalacticStochastic.noise_manager import NoiseModelManager
    from LisaWaveformTools.lisa_config import LISAConstants
    from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
    from WaveletWaveforms.wdm_config import WDMWaveletConstants

N_PAR_GB = 8


def unpack_params_gb(params_in: NDArray[np.floating]) -> SourceParams:
    assert len(params_in.shape) == 1
    assert params_in.size == N_PAR_GB
    # Ecliptic latitude to cosine of ecliptic colatitude
    costh = float(np.cos(np.pi / 2 - params_in[1]))
    # Ecliptic longitude
    phi = float(params_in[2])
    # Cosine of inclination angle
    cosi = float(np.cos(params_in[5]))
    # Polarization angle
    psi = float(params_in[7])
    amp0_t = float(params_in[0])
    F0 = float(params_in[3])
    FTd0 = float(params_in[4])
    phi0 = float(params_in[6] + np.pi)
    params_extrinsic = ExtrinsicParams(costh, phi, cosi, psi)
    params_intrinsic = LinearFrequencyIntrinsicParams(amp0_t, phi0, F0, FTd0)
    return SourceParams(intrinsic=params_intrinsic, extrinsic=params_extrinsic)


class BinaryInclusionState(StateManager):
    """Stores all the binaries under consideration in the galaxy"""

    def __init__(
        self,
        wc: WDMWaveletConstants,
        ic: IterationConfig,
        lc: LISAConstants,
        params_gb_in: NDArray[np.floating],
        noise_manager: NoiseModelManager,
        fit_state: IterativeFitState,
        nt_lim_waveform:  PixelGenericRange,
        snrs_tot_in: NDArray[np.floating] | None = None,
    ) -> None:
        """Class that stores information about which component of the galactic signal binaries belong to."""
        self._wc: WDMWaveletConstants = wc
        self._ic: IterationConfig = ic
        self._lc: LISAConstants = lc
        self._nt_lim_waveform: PixelGenericRange = nt_lim_waveform
        self._noise_manager: NoiseModelManager = noise_manager
        self._fit_state: IterativeFitState = fit_state

        del wc
        del ic
        del lc
        del noise_manager
        del fit_state
        del nt_lim_waveform

        assert len(params_gb_in.shape) == 2
        assert params_gb_in.shape[1] == N_PAR_GB

        self._n_tot: int = params_gb_in.shape[0]
        self._fmin_binary: float = max(0.0, self._ic.fmin_binary)
        self._fmax_binary: float = min((self._wc.Nf - 1) * self._wc.DF, self._ic.fmax_binary)

        if snrs_tot_in is not None:
            faints_in = (
                (snrs_tot_in < self._ic.snr_min_preprocess)
                | (params_gb_in[:, 3] >= self._fmax_binary)
                | (params_gb_in[:, 3] < self._fmin_binary)
            )
        else:
            faints_in = (params_gb_in[:, 3] >= self._fmax_binary) | (params_gb_in[:, 3] < self._fmin_binary)

        self._argbinmap: NDArray[np.integer] = np.argwhere(~faints_in).flatten()
        self._faints_old: NDArray[np.bool_] = faints_in[self._argbinmap]
        assert self._faints_old.sum() == 0.0
        self._n_bin_use: int = self._argbinmap.size

        # ensure we are making a copy of the parameters
        self._params_gb: NDArray[np.floating] = np.zeros((self._n_bin_use, N_PAR_GB))
        self._params_gb[:] = params_gb_in[self._argbinmap]

        self._snrs_old: NDArray[np.floating] = np.zeros(self._n_bin_use)

        # record old snrs if available
        if snrs_tot_in is not None:
            self._snrs_old[:] = snrs_tot_in[self._argbinmap]
        else:
            self._snrs_old[:] = -1.0

        del snrs_tot_in
        del params_gb_in
        del faints_in

        self._snrs_upper: NDArray[np.floating] = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use, self._lc.nc_snr))
        self._snrs_lower: NDArray[np.floating] = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use, self._lc.nc_snr))
        self._snrs_tot_lower: NDArray[np.floating] = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use))
        self._snrs_tot_upper: NDArray[np.floating] = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use))
        self._brights: NDArray[np.bool_] = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use), dtype=np.bool_)
        self._decided: NDArray[np.bool_] = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use), dtype=np.bool_)
        self._faints_cur: NDArray[np.bool_] = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use), dtype=np.bool_)

        params0_sel: NDArray[np.floating] = self._params_gb[0]
        params0: SourceParams = unpack_params_gb(params0_sel)
        self._waveform_manager: LinearFrequencyWaveletWaveformTime = LinearFrequencyWaveletWaveformTime(params0, self._wc, self._lc, self._nt_lim_waveform)

        del params0
        del params0_sel

        self._itrn: int = 0

        self._n_faints_cur: NDArray[np.integer] = np.zeros(self._fit_state.get_n_itr_cut() + 1, dtype=np.int64)
        self._n_brights_cur: NDArray[np.integer] = np.zeros(self._fit_state.get_n_itr_cut() + 1, dtype=np.int64)

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'inclusion_state', group_mode: int = 0, noise_recurse: int = 1) -> h5py.Group:
        # can store different things depending on whether we are in preprocessing mode or not
        if self._fit_state.preprocess_mode == 0:
            storage_mode = self._ic.inclusion_state_storage_mode
        else:
            storage_mode = self._ic.inclusion_state_storage_mode_prelim

        if group_mode == 0:
            hf_include = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_include = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        hf_include.attrs['storage_mode'] = storage_mode
        hf_include.attrs['itrn'] = self._itrn
        hf_include.attrs['n_bin_use'] = self._n_bin_use
        hf_include.attrs['n_tot'] = self._n_tot
        hf_include.attrs['fmin_binary'] = self._fmin_binary
        hf_include.attrs['fmax_binary'] = self._fmax_binary
        hf_include.attrs['noise_manager_name'] = self._noise_manager.__class__.__name__
        hf_include.attrs['fit_state_name'] = self._fit_state.__class__.__name__
        hf_include.attrs['waveform_manager_name'] = self._waveform_manager.__class__.__name__

        _ = hf_include.create_dataset('argbinmap', data=self._argbinmap, compression='gzip')
        _ = hf_include.create_dataset('faints_old', data=self._faints_old, compression='gzip')
        _ = hf_include.create_dataset('snrs_old', data=self._snrs_old, compression='gzip')
        _ = hf_include.create_dataset('n_faints_cur', data=self._n_faints_cur, compression='gzip')
        _ = hf_include.create_dataset('n_brights_cur', data=self._n_brights_cur, compression='gzip')

        _ = hf_include.create_dataset('faints_cur', data=self._faints_cur[:self._itrn], compression='gzip')
        _ = hf_include.create_dataset('decided', data=self._decided[:self._itrn], compression='gzip')
        _ = hf_include.create_dataset('brights', data=self._brights[:self._itrn], compression='gzip')

        if storage_mode in (0, 2):
            # store full snrs
            _ = hf_include.create_dataset('snrs_upper', data=self._snrs_upper[:self._itrn], compression='gzip')
            _ = hf_include.create_dataset('snrs_lower', data=self._snrs_lower[:self._itrn], compression='gzip')
            _ = hf_include.create_dataset('snrs_tot_upper', data=self._snrs_tot_upper[:self._itrn], compression='gzip')
            _ = hf_include.create_dataset('snrs_tot_lower', data=self._snrs_tot_lower[:self._itrn], compression='gzip')

        if storage_mode in (1, 3):
            # store last snrs
            _ = hf_include.create_dataset('snrs_upper', data=self._snrs_upper[self._itrn - 1:self._itrn], compression='gzip')
            _ = hf_include.create_dataset('snrs_lower', data=self._snrs_lower[self._itrn - 1:self._itrn], compression='gzip')
            _ = hf_include.create_dataset('snrs_tot_upper', data=self._snrs_tot_upper[self._itrn - 1:self._itrn], compression='gzip')
            _ = hf_include.create_dataset('snrs_tot_lower', data=self._snrs_tot_lower[self._itrn - 1:self._itrn], compression='gzip')

        if storage_mode in (2, 4):
            # store full params
            _ = hf_include.create_dataset('params_gb', data=self._params_gb, compression='gzip')

        if storage_mode == 5:
            # minimal storage
            _ = hf_include.create_dataset('snrs_tot_upper', data=np.array([self._snrs_tot_upper[self._itrn - 1]]), compression='gzip')

        # option to skip storing the noise manager in case it is redundant
        if noise_recurse == 0:
            pass
        elif noise_recurse == 1:
            _ = self._noise_manager.store_hdf5(hf_include)
        else:
            msg = 'Unrecognized option for noise_recurse'
            raise NotImplementedError(msg)

        _ = self._fit_state.store_hdf5(hf_include)
        _ = self._waveform_manager.store_hdf5(hf_include)

        # the wavelet constants
        hf_include.attrs['wc_name'] = self._wc.__class__.__name__
        hf_wc = hf_include.create_group('wc')
        for key in self._wc._fields:
            hf_wc.attrs[key] = getattr(self._wc, key)

        # lisa related constants
        hf_lc = hf_include.create_group('lc')
        hf_include.attrs['lc_name'] = self._lc.__class__.__name__
        for key in self._lc._fields:
            hf_lc.attrs[key] = getattr(self._lc, key)

        # iterative fit related constants
        hf_include.attrs['ic_name'] = self._ic.__class__.__name__
        hf_ic = hf_include.create_group('ic')
        for key in self._ic._fields:
            hf_ic.attrs[key] = getattr(self._ic, key)

        hf_include.attrs['nt_lim_name'] = self._nt_lim_waveform.__class__.__name__
        hf_nt = hf_include.create_group('nt_lim_waveform')
        for key in self._nt_lim_waveform._fields:
            hf_nt.attrs[key] = getattr(self._nt_lim_waveform, key)

        return hf_include

    def load_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'inclusion_state', group_mode: int = 0) -> None:
        """Load the object from an hdf5 group"""
        if group_mode == 0:
            hf_include = hf_in['inclusion_state']
        elif group_mode == 1:
            hf_include = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)

        if not isinstance(hf_include, h5py.Group):
            msg = 'Could not find group ' + group_name + ' in hdf5 file'
            raise TypeError(msg)

        storage_mode_val = hf_include.attrs['storage_mode']
        assert isinstance(storage_mode_val, (int, np.integer)), 'storage_mode must be int'
        storage_mode = int(storage_mode_val)

        itrn_val = hf_include.attrs['itrn']
        assert isinstance(itrn_val, (int, np.integer)), 'itrn must be int'
        self._itrn = int(itrn_val)

        n_bin_use_val = hf_include.attrs['n_bin_use']
        assert isinstance(n_bin_use_val, (int, np.integer)), 'n_bin_use must be int'
        self._n_bin_use = int(n_bin_use_val)

        n_tot_val = hf_include.attrs['n_tot']
        assert isinstance(n_tot_val, (int, np.integer)), 'n_tot must be int'
        self._n_tot = int(n_tot_val)

        fmin_binary_val = hf_include.attrs['fmin_binary']
        assert isinstance(fmin_binary_val, (float, np.floating, int, np.integer)), 'fmin_binary must be float'
        self._fmin_binary = float(fmin_binary_val)

        fmax_binary_val = hf_include.attrs['fmax_binary']
        assert isinstance(fmax_binary_val, (float, np.floating, int, np.integer)), 'fmax_binary must be float'
        self._fmax_binary = float(fmax_binary_val)

        assert hf_include.attrs['noise_manager_name'] == self._noise_manager.__class__.__name__, 'incorrect noise manager name found in hdf5 file'
        assert hf_include.attrs['fit_state_name'] == self._fit_state.__class__.__name__, 'incorrect fit state name found in hdf5 file'
        assert hf_include.attrs['waveform_manager_name'] == self._waveform_manager.__class__.__name__, 'incorrect waveform manager name found in hdf5 file'
        assert hf_include.attrs['wc_name'] == self._wc.__class__.__name__, 'incorrect wavelet constants name found in hdf5 file'
        assert hf_include.attrs['lc_name'] == self._lc.__class__.__name__, 'incorrect lisa constants name found in hdf5 file'
        assert hf_include.attrs['ic_name'] == self._ic.__class__.__name__, 'incorrect iteration config name found in hdf5 file'
        assert hf_include.attrs['nt_lim_name'] == self._nt_lim_waveform.__class__.__name__, 'incorrect nt_lim_waveform name found in hdf5 file'
        if self._fit_state.preprocess_mode == 0:
            assert storage_mode == self._ic.inclusion_state_storage_mode, 'storage mode in hdf5 file does not match current config'
        else:
            assert storage_mode == self._ic.inclusion_state_storage_mode_prelim, 'storage mode in hdf5 file does not match current config'

        argbbinmap_temp = hf_include['argbinmap']
        assert isinstance(argbbinmap_temp, h5py.Dataset)
        self._argbinmap = argbbinmap_temp[()]

        faints_old_temp = hf_include['faints_old']
        assert isinstance(faints_old_temp, h5py.Dataset)
        self._faints_old = faints_old_temp[()]

        snrs_old_temp = hf_include['snrs_old']
        assert isinstance(snrs_old_temp, h5py.Dataset)
        self._snrs_old = snrs_old_temp[()]

        n_faints_cur_temp = hf_include['n_faints_cur']
        assert isinstance(n_faints_cur_temp, h5py.Dataset)
        self._n_faints_cur = n_faints_cur_temp[()]

        n_brights_cur_temp = hf_include['n_brights_cur']
        assert isinstance(n_brights_cur_temp, h5py.Dataset)
        self._n_brights_cur = n_brights_cur_temp[()]

        assert self._n_bin_use == self._argbinmap.size
        assert self._n_bin_use == self._faints_old.size
        assert self._n_faints_cur.size == self._fit_state.get_n_itr_cut() + 1
        assert self._n_brights_cur.size == self._fit_state.get_n_itr_cut() + 1

        self._faints_cur = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use), dtype=np.bool_)
        self._decided = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use), dtype=np.bool_)
        self._brights = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use), dtype=np.bool_)
        self._snrs_upper = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use, self._lc.nc_snr))
        self._snrs_lower = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use, self._lc.nc_snr))
        self._snrs_tot_lower = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use))
        self._snrs_tot_upper = np.zeros((self._fit_state.get_n_itr_cut(), self._n_bin_use))
        self._params_gb = np.zeros((self._n_bin_use, N_PAR_GB))

        try:
            snrs_upper_temp = hf_include['snrs_upper']
            assert isinstance(snrs_upper_temp, h5py.Dataset)
            snrs_lower_temp = hf_include['snrs_lower']
            assert isinstance(snrs_lower_temp, h5py.Dataset)
            snrs_tot_upper_temp = hf_include['snrs_tot_upper']
            assert isinstance(snrs_tot_upper_temp, h5py.Dataset)
            snrs_tot_lower_temp = hf_include['snrs_tot_lower']
            assert isinstance(snrs_tot_lower_temp, h5py.Dataset)
            if storage_mode in (0, 2):
                # store full snrs
                self._snrs_upper[:self._itrn] = snrs_upper_temp[()]
                self._snrs_lower[:self._itrn] = snrs_lower_temp[()]
                self._snrs_tot_upper[:self._itrn] = snrs_tot_upper_temp[()]
                self._snrs_tot_lower[:self._itrn] = snrs_tot_lower_temp[()]

            if storage_mode in (1, 3):
                # store last snrs
                self._snrs_upper[self._itrn - 1:self._itrn] = snrs_upper_temp[()]
                self._snrs_lower[self._itrn - 1:self._itrn] = snrs_lower_temp[()]
                self._snrs_tot_upper[self._itrn - 1:self._itrn] = snrs_tot_upper_temp[()]
                self._snrs_tot_lower[self._itrn - 1:self._itrn] = snrs_tot_lower_temp[()]

            if storage_mode in (2, 4):
                # store full params
                params_gb_temp = hf_include['params_gb']
                assert isinstance(params_gb_temp, h5py.Dataset)
                self._params_gb[:] = params_gb_temp[()]
                assert len(self._params_gb.shape) == 2
                assert self._params_gb.shape[0] == self._n_bin_use
                assert self._params_gb.shape[1] == N_PAR_GB
            else:
                if self._fit_state.get_preprocess_mode() == 0:
                    msg = 'params_gb not stored in hdf5 file, cannot reconstruct without original file'
                    warn(msg, stacklevel=2)
                # TODO: otherwise we should reconstruct the params from the argbinmap and a file
        except KeyError:
            assert storage_mode == 5, 'snrs_lower, snrs_tot_upper, and snrs_tot_lower datasets not found in hdf5 file'
            # minimal storage
            snrs_tot_upper_temp = hf_include['snrs_tot_upper']
            assert isinstance(snrs_tot_upper_temp, h5py.Dataset)
            snrs_tot_upper_last = snrs_tot_upper_temp[()]
            assert len(snrs_tot_upper_last.shape) == 1
            assert snrs_tot_upper_last.size == 1
            self._snrs_tot_upper[self._itrn - 1, :] = snrs_tot_upper_last[0]

        faints_cur_temp = hf_include['faints_cur']
        assert isinstance(faints_cur_temp, h5py.Dataset)
        self._faints_cur[:self._itrn] = faints_cur_temp[()]

        decided_temp = hf_include['decided']
        assert isinstance(decided_temp, h5py.Dataset)
        self._decided[:self._itrn] = decided_temp[()]

        brights_temp = hf_include['brights']
        assert isinstance(brights_temp, h5py.Dataset)
        self._brights[:self._itrn] = brights_temp[()]

        # shouldn't need to load the waveform manager, because it is reconstructed each time it is used, as long as the type is correct

        for key in self._wc._fields:
            assert getattr(self._wc, key) == hf_include['wc'].attrs[key], f'wavelet constant attribute {key} does not match saved value'

        for key in self._lc._fields:
            assert getattr(self._lc, key) == hf_include['lc'].attrs[key], f'lisa constant attribute {key} does not match saved value'

        for key in self._ic._fields:
            assert np.all(getattr(self._ic, key) == hf_include['ic'].attrs[key]), f'iteration config attribute {key} does not match saved value'

        for key in self._nt_lim_waveform._fields:
            assert getattr(self._nt_lim_waveform, key) == hf_include['nt_lim_waveform'].attrs[key], f'nt_lim_waveform attribute {key} does not match saved value'

    def set_select_params(self, params_gb_in: NDArray[np.floating]) -> None:
        assert params_gb_in.shape == (self._n_tot, N_PAR_GB)
        params_gb_sel = params_gb_in[self._argbinmap]
        assert self._params_gb.shape == params_gb_sel.shape
        assert np.all((params_gb_in[:, 3] < self._fmax_binary) & (params_gb_in[:, 3] >= self._fmin_binary))
        if np.all(self._params_gb == 0.):
            self._params_gb[:] = params_gb_sel
        else:
            assert_allclose(self._params_gb, params_gb_sel)

    def sustain_snr_helper(self) -> None:
        """Helper to carry forward any other snr values we know from a previous iteration"""
        itrn = self._itrn
        if self._fit_state.get_faint_converged():
            assert itrn > 1
            self._snrs_tot_lower[itrn, self._decided[itrn]] = self._snrs_tot_lower[itrn - 1, self._decided[itrn]]
            self._snrs_lower[itrn, self._decided[itrn]] = self._snrs_lower[itrn - 1, self._decided[itrn]]

        if self._fit_state.get_bright_converged():
            assert itrn > 1
            self._snrs_tot_upper[itrn, self._decided[itrn]] = self._snrs_tot_upper[itrn - 1, self._decided[itrn]]
            self._snrs_upper[itrn, self._decided[itrn]] = self._snrs_upper[itrn - 1, self._decided[itrn]]

    def oscillation_check_helper(self) -> tuple[bool, bool, bool]:
        """Helper used by fit_state to decide if the bright binaries are oscillating without converging"""
        osc1 = False
        osc2 = False
        osc3 = False

        if self._itrn > 1:
            osc1 = bool(np.all(self._brights[self._itrn - 1] == self._brights[self._itrn - 2]))
        if self._itrn > 2:
            osc2 = bool(np.all(self._brights[self._itrn - 1] == self._brights[self._itrn - 3]))
        if self._itrn > 3:
            osc3 = bool(np.all(self._brights[self._itrn - 1] == self._brights[self._itrn - 4]))

        converged_or_cycling = osc1 or osc2 or osc3
        old_match = osc2 or osc3
        cycling = old_match and not osc1
        return cycling, converged_or_cycling, old_match

    def delta_faint_check_helper(self) -> int:
        """Get the difference in the number of faint binaries between the last two iterations"""
        if self._itrn - 1 == 0:
            delta_faints = int(self._n_faints_cur[self._itrn - 1])
        else:
            delta_faints = int(self._n_faints_cur[self._itrn - 1] - self._n_faints_cur[self._itrn - 2])
        return delta_faints

    def delta_bright_check_helper(self) -> int:
        """Get the difference in the number of bright binaries between the last two iterations"""
        if self._itrn - 1 == 0:
            delta_brights = int(self._n_brights_cur[self._itrn - 1])
        else:
            delta_brights = int(self._n_brights_cur[self._itrn - 1] - self._n_brights_cur[self._itrn - 2])
        return delta_brights

    def convergence_decision_helper(self) -> tuple[tuple[bool, bool, bool], int, int]:
        return (self.oscillation_check_helper(), self.delta_faint_check_helper(), self.delta_bright_check_helper())

    def run_binary_coadd(self, itrb: int) -> None:
        """Get the intrinsic_waveform for a binary, store its snr, and decide which spectrum to add it to."""
        itrn = self._itrn
        params_loc = unpack_params_gb(self._params_gb[itrb])
        self._waveform_manager.update_params(params_loc)

        self.snr_storage_helper(itrb)
        self._brights[itrn, itrb], self._faints_cur[itrn, itrb] = self.decision_helper(itrb)
        self.decide_coadd_helper(itrb)

    def snr_storage_helper(self, itrb: int) -> None:
        """Helper to store the snrs of the current binary"""
        itrn = self._itrn
        wavelet_waveform = self._waveform_manager.get_unsorted_coeffs()

        if not self._fit_state.get_faint_converged():
            self._snrs_lower[itrn, itrb] = self._noise_manager.noise_lower.get_sparse_snrs(
                wavelet_waveform, self._noise_manager.nt_lim_snr,
            )
            self._snrs_tot_lower[itrn, itrb] = np.linalg.norm(self._snrs_lower[itrn, itrb])
        else:
            assert itrn > 1
            self._snrs_lower[itrn, itrb] = self._snrs_lower[itrn - 1, itrb]
            self._snrs_tot_lower[itrn, itrb] = self._snrs_tot_lower[itrn - 1, itrb]

        if not self._fit_state.get_bright_converged():
            self._snrs_upper[itrn, itrb] = self._noise_manager.noise_upper.get_sparse_snrs(
                wavelet_waveform, self._noise_manager.nt_lim_snr,
            )
            self._snrs_tot_upper[itrn, itrb] = np.linalg.norm(self._snrs_upper[itrn, itrb])
        else:
            assert itrn > 1
            self._snrs_upper[itrn, itrb] = self._snrs_upper[itrn - 1, itrb]
            self._snrs_tot_upper[itrn, itrb] = self._snrs_tot_upper[itrn - 1, itrb]

        if np.isnan(self._snrs_tot_upper[itrn, itrb]) or np.isnan(self._snrs_tot_lower[itrn, itrb]):
            raise ValueError('nan detected in snr at ' + str(itrn) + ', ' + str(itrb))

        if ~np.isfinite(self._snrs_tot_upper[itrn, itrb]) or ~np.isfinite(self._snrs_tot_lower[itrn, itrb]):
            raise ValueError('Non-finite value detected in snr at ' + str(itrn) + ', ' + str(itrb))

    def decision_helper(self, itrb: int) -> tuple[bool, bool]:
        """Helper to decide whether a binary is bright or faint by the current noise spectrum"""
        itrn = self._itrn
        if self._fit_state.get_preprocess_mode() == 1:
            snr_cut_faint_loc = self._ic.snr_min_preprocess
        elif self._fit_state.get_preprocess_mode() == 2:
            snr_cut_faint_loc = self._ic.snr_min_reprocess
        else:
            snr_cut_faint_loc = self._ic.snr_min[itrn]

        if not self._fit_state.get_faint_converged():
            faint_candidate = bool(self._snrs_tot_lower[itrn, itrb] < snr_cut_faint_loc)
        else:
            faint_candidate = False

        if not self._fit_state.get_bright_converged():
            bright_candidate = bool(self._snrs_tot_upper[itrn, itrb] >= self._ic.snr_cut_bright[itrn])
        else:
            bright_candidate = False

        if bright_candidate and faint_candidate:
            # satifisfied conditions to be eliminated in both directions so just keep it
            bright_loc = False
            faint_loc = False
        elif bright_candidate:
            if self._snrs_tot_upper[itrn, itrb] > self._snrs_tot_lower[itrn, itrb]:
                # handle case where snr ordering is wrong to prevent oscillation
                bright_loc = False
            else:
                bright_loc = True
            faint_loc = False
        elif faint_candidate:
            bright_loc = False
            faint_loc = True
        else:
            bright_loc = False
            faint_loc = False

        return bright_loc, faint_loc

    def get_final_snrs_tot_upper(self) -> NDArray[np.floating]:
        return self._snrs_tot_upper[self._itrn - 1, :]

    def decide_coadd_helper(self, itrb: int) -> None:
        """Add each binary to the correct part of the galactic spectrum, depending on whether it is bright or faint"""
        itrn = self._itrn
        # the same binary cannot be decided as both bright and faint
        assert not (self._brights[itrn, itrb] and self._faints_cur[itrn, itrb])

        # don't add to anything if the bright adaptation is already converged and this binary would not be faint
        if self._fit_state.get_bright_converged() and not self._faints_cur[itrn, itrb]:
            return

        wavelet_waveform = self._waveform_manager.get_unsorted_coeffs()

        if not self._faints_cur[itrn, itrb]:
            if self._brights[itrn, itrb]:
                # binary is bright enough to decide
                self._noise_manager.bgd.add_bright(wavelet_waveform)
            else:
                # binary neither faint nor bright enough to decide
                self._noise_manager.bgd.add_undecided(wavelet_waveform)
        # binary is faint enough to decide
        elif itrn == 0:
            self._faints_cur[itrn, itrb] = False
            self._faints_old[itrb] = True
            self._noise_manager.bgd.add_floor(wavelet_waveform)
        else:
            self._noise_manager.bgd.add_faint(wavelet_waveform)

    @override
    def advance_state(self) -> None:
        """Handle any logic necessary to advance the state of the object to the next iteration"""
        if self._itrn == 0:
            self._faints_cur[self._itrn] = False
            self._brights[self._itrn] = False
            self._noise_manager.bgd.clear_undecided()
            self._noise_manager.bgd.clear_above()
        else:
            self._faints_cur[self._itrn] = self._faints_cur[self._itrn - 1]

            if self._fit_state.get_bright_converged():
                self._brights[self._itrn] = self._brights[self._itrn - 1]
            else:
                self._noise_manager.bgd.clear_undecided()
                if self._fit_state.get_do_faint_check():
                    self._noise_manager.bgd.clear_above()
                    self._brights[self._itrn] = False
                else:
                    self._brights[self._itrn] = self._brights[self._itrn - 1]

        self._decided[self._itrn] = self._brights[self._itrn] | self._faints_cur[self._itrn] | self._faints_old

        idxbs: NDArray[np.integer] = np.argwhere(~self._decided[self._itrn]).flatten()

        tib = perf_counter()

        for itrb in idxbs:
            if itrb % 10000 == 0:
                tcb = perf_counter()
                print(
                    'Starting binary # %11d of %11d to consider at t=%9.2f s of iteration %4d'
                    % (itrb, idxbs.size, (tcb - tib), self._itrn),
                )

            self.run_binary_coadd(int(itrb))

        # copy forward prior calculations of snr calculations that were skipped in this loop iteration
        self.sustain_snr_helper()

        self._n_brights_cur[self._itrn] = int(self._brights[self._itrn].sum())
        self._n_faints_cur[self._itrn] = int(self._faints_cur[self._itrn].sum())

        self._itrn += 1

    @override
    def state_check(self) -> None:
        """Do any self consistency checks based on the current state"""
        if self._itrn > 0:
            if self._fit_state.get_bright_converged_old():
                assert self._itrn > 1
                assert np.all(self._brights[self._itrn - 1] == self._brights[self._itrn - 2])

            if self._fit_state.get_faint_converged_old():
                assert self._itrn > 1
                assert np.all(self._faints_cur[self._itrn - 1] == self._faints_cur[self._itrn - 2])

    @override
    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends"""
        Tobs_consider_yr = (self._noise_manager.nt_lim_snr.nx_max - self._noise_manager.nt_lim_snr.nx_min) * self._wc.DT / gc.SECSYEAR
        n_consider = self._n_bin_use
        n_faint = int(self._faints_old.sum())
        n_faint2 = int(self._faints_cur[self._itrn - 1].sum())
        n_bright = int(self._brights[self._itrn - 1].sum())
        n_ambiguous = int((~(self._faints_old | self._brights[self._itrn - 1] | self._faints_cur[self._itrn - 1])).sum())
        print(
            'Out of %10d total binaries, %10d were deemed undetectable by a previous run, %10d were considered here.'
            % (self._n_tot, self._n_tot - n_consider, n_consider),
        )
        print(
            'The iterative procedure deemed (%5.3f yr observation at threshold snr=%5.3f):'
            % (Tobs_consider_yr, self._ic.snr_thresh),
        )
        print('       %10d undetectable due to instrument noise' % n_faint)
        print('       %10d undetectable due to galactic confusion' % n_faint2)
        print('       %10d undecided (presumed undetectable)' % n_ambiguous)
        print('       %10d total undetectable' % (self._n_tot - n_bright))
        print('       %10d total detectable' % n_bright)

        assert n_ambiguous + n_bright + n_faint + n_faint2 == n_consider

    @override
    def log_state(self) -> None:
        """Perform any internal logging that should be done after advance_state is run."""
        return

    @override
    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends"""
        return
