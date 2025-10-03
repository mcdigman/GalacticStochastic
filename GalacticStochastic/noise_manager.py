"""Object to manage the noise models used in the iterative fit."""

from typing import override

import h5py
import numpy as np
from numpy.typing import NDArray

from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.state_manager import StateManager
from GalacticStochastic.testing_tools import unit_normal_battery
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.noise_model import DiagonalNonstationaryDenseNoiseModel, DiagonalStationaryDenseNoiseModel
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class NoiseModelManager(StateManager):
    """Manage the noise models used in the iterative fit."""

    def __init__(
        self,
        ic: IterationConfig,
        wc: WDMWaveletConstants,
        lc: LISAConstants,
        fit_state: IterativeFitState,
        bgd: BGDecomposition,
        cyclo_mode: int,
        nt_lim_snr: PixelGenericRange,
        instrument_random_seed: int,
    ) -> None:
        """
        Initialize a NoiseModelManager to manage noise models for the iterative fit.

        Parameters
        ----------
        ic : IterationConfig
            Configuration object for the iterative fit.
        wc : WDMWaveletConstants
            Wavelet constants describing the time-frequency grid.
        lc : LISAConstants
            LISA instrument configuration constants.
        fit_state : IterativeFitState
            Object tracking the current state of the iterative fit.
        bgd : BGDecomposition
            Galactic background decomposition object.
        cyclo_mode : int
            Cyclostationary mode flag. If nonzero, enables cyclostationary noise modeling.
        nt_lim_snr : PixelGenericRange
            Range of time-frequency pixels used for SNR calculations.
        instrument_random_seed : int
            Random seed for instrument noise realizations.

        Raises
        ------
        ValueError
            If an unrecognized storage mode or noise model mode is provided.
        NotImplementedError
            If a noise model mode other than 0 is requested.
        """
        if ic.noise_model_storage_mode not in (0, 1, 2):
            msg = 'Unrecognized option for storage mode'
            raise ValueError(msg)

        self._ic: IterationConfig = ic
        self._lc: LISAConstants = lc
        self.wc: WDMWaveletConstants = wc
        self.bgd: BGDecomposition = bgd
        self._fit_state: IterativeFitState = fit_state
        self.cyclo_mode: int = cyclo_mode
        self.nt_lim_snr: PixelGenericRange = nt_lim_snr
        self._instrument_random_seed: int = instrument_random_seed

        if ic.noise_model_mode == 0:
            self.S_inst_m: NDArray[np.floating] = instrument_noise_AET_wdm_m(self._lc, self.wc)
        else:
            msg = 'Unrecognized option for noise model mode'
            raise NotImplementedError(msg)

        self._itrn: int = 0

        self._idx_S_save: NDArray[np.integer] = np.hstack(
            [
                np.arange(0, min(10, self._fit_state.get_n_itr_cut())),
                np.arange(min(10, self._fit_state.get_n_itr_cut()), 4),
                self._fit_state.get_n_itr_cut() - 1,
            ],
        )
        self._itr_save: int = 0

        self.S_record_upper: NDArray[np.floating] = np.zeros((self._idx_S_save.size, wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        self.S_record_lower: NDArray[np.floating] = np.zeros((self._idx_S_save.size, wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        self.S_final: NDArray[np.floating] = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))

        S_upper: NDArray[np.floating] = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        S_upper[:] = self.S_inst_m

        S_lower: NDArray[np.floating] = np.zeros((wc.Nt, wc.Nf, self.bgd.nc_galaxy))
        S_lower[:] = self.S_inst_m
        if self._idx_S_save[self._itr_save] == 0:
            self.S_record_upper[0] = S_upper[:, :, :]
            self.S_record_lower[0] = S_lower[:, :, :]
            self._itr_save += 1
        S_lower = np.asarray(np.min([S_lower, S_upper], axis=0), dtype=np.float64)

        # TODO must set seed here
        self.noise_upper: DiagonalNonstationaryDenseNoiseModel = DiagonalNonstationaryDenseNoiseModel(S_upper, wc, prune=1, nc_snr=lc.nc_snr, seed=self._instrument_random_seed)
        self.noise_lower: DiagonalNonstationaryDenseNoiseModel = DiagonalNonstationaryDenseNoiseModel(S_lower, wc, prune=1, nc_snr=lc.nc_snr, seed=self._instrument_random_seed)
        self.noise_instrument: DiagonalStationaryDenseNoiseModel = DiagonalStationaryDenseNoiseModel(self.S_inst_m, wc, prune=1, nc_snr=lc.nc_snr, seed=self._instrument_random_seed)

        del S_upper
        del S_lower

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'noise_model', group_mode: int = 0) -> h5py.Group:
        """
        Store attributes, configuration, and results to an HDF5 file.

        This method saves the current state, including relevant attributes and results,
        to the specified HDF5 group. The data can be organized under a specific group name
        and with a chosen storage mode.

        Parameters
        ----------
        hf_in : h5py.Group
            The HDF5 group where the state will be stored.
        group_name : str
            Name of the group under which to store the state (default is 'state_manager').
        group_mode : int
            If group_mode == 1, do not create a new group, and write directly to hf_in.
            If group_mode == 0, create a new group under hf_in with name group_name (default is 0).

        Returns
        -------
        h5py.Group
            The HDF5 group containing the stored state.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        if group_mode == 0:
            hf_noise = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_noise = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)

        storage_mode = self._ic.noise_model_storage_mode
        hf_noise.attrs['creator_name'] = self.__class__.__name__
        hf_noise.attrs['storage_mode'] = storage_mode
        hf_noise.attrs['itrn'] = self._itrn
        hf_noise.attrs['itr_save'] = self._itr_save
        hf_noise.attrs['cyclo_mode'] = self.cyclo_mode
        hf_noise.attrs['noise_upper_name'] = self.noise_upper.__class__.__name__
        hf_noise.attrs['noise_lower_name'] = self.noise_lower.__class__.__name__
        hf_noise.attrs['noise_instrument_name'] = self.noise_instrument.__class__.__name__
        hf_noise.attrs['bgd_name'] = self.bgd.__class__.__name__
        hf_noise.attrs['fit_state_name'] = self._fit_state.__class__.__name__
        hf_noise.attrs['nt_lim_snr_name'] = self.nt_lim_snr.__class__.__name__
        hf_noise.attrs['instrument_random_seed'] = self._instrument_random_seed
        hf_noise.attrs['wc_name'] = self.wc.__class__.__name__
        hf_noise.attrs['lc_name'] = self._lc.__class__.__name__
        hf_noise.attrs['ic_name'] = self._ic.__class__.__name__

        _ = hf_noise.create_dataset('S_inst_m', data=self.S_inst_m, compression='gzip')
        _ = hf_noise.create_dataset('idx_S_save', data=self._idx_S_save, compression='gzip')

        # at least store the history of the mean spectrum
        _ = hf_noise.create_dataset('S_record_upper_mean', data=np.mean(self.S_record_upper, axis=1), compression='gzip')
        _ = hf_noise.create_dataset('S_record_lower_mean', data=np.mean(self.S_record_lower, axis=1), compression='gzip')

        if storage_mode in (1, 2):
            # can safely be rederived as long as bgd is stored
            _ = hf_noise.create_dataset('S_record_final', data=self.S_record_lower, compression='gzip')

        if storage_mode == 2:
            # full versions of these potentially take a lot of memory, and aren't always needed so don't necessarily want to write them to disk
            _ = hf_noise.create_dataset('S_record_upper', data=self.S_record_upper, compression='gzip')
            _ = hf_noise.create_dataset('S_record_lower', data=self.S_record_lower, compression='gzip')

        _ = self.bgd.store_hdf5(hf_noise)

        # storing the fit state will probably be done more than once, but it shouldn't be very large
        # and it is possibly more self-descriptive that way
        _ = self._fit_state.store_hdf5(hf_noise)

        _ = self.noise_upper.store_hdf5(hf_noise, group_name='noise_upper')
        _ = self.noise_lower.store_hdf5(hf_noise, group_name='noise_lower')
        _ = self.noise_instrument.store_hdf5(hf_noise, group_name='noise_instrument')

        # store all the configuration objects to the file

        hf_nt = hf_noise.create_group('nt_lim_snr')
        for key in self.nt_lim_snr._fields:
            hf_nt.attrs[key] = getattr(self.nt_lim_snr, key)

        # the wavelet constants
        hf_wc = hf_noise.create_group('wc')
        for key in self.wc._fields:
            hf_wc.attrs[key] = getattr(self.wc, key)

        # lisa related constants
        hf_lc = hf_noise.create_group('lc')
        for key in self._lc._fields:
            hf_lc.attrs[key] = getattr(self._lc, key)

        # iterative fit related constants
        hf_ic = hf_noise.create_group('ic')
        for key in self._ic._fields:
            hf_ic.attrs[key] = getattr(self._ic, key)

        return hf_noise

    def load_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'noise_model', group_mode: int = 0) -> None:
        """
        Load attributes, configuration, and results from an HDF5 file.

        This method loads the current state, including relevant attributes and results,
        from the specified HDF5 group, as well as possible. The data can be organized under a specific group name
        and with a chosen storage mode.

        Parameters
        ----------
        hf_in : h5py.Group
            The HDF5 group where the state was stored.
        group_name : str
            Name of the group under which to store the state (default is 'state_manager').
        group_mode : int
            If group_mode == 1, assume no new group was created, and read directly from hf_in.
            If group_mode == 0, assume a new group was created under hf_in with name group_name (default is 0).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        TypeError
            If the format is not as expected.
        ValueError
            If loaded attributes do not match the current object's attributes.
        """
        if group_mode == 0:
            hf_noise = hf_in['noise_model']
        elif group_mode == 1:
            hf_noise = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        if not isinstance(hf_noise, h5py.Group):
            msg = 'Could not find group ' + group_name + ' in hdf5 file'
            raise TypeError(msg)

        assert hf_noise.attrs['creator_name'] == self.__class__.__name__, 'incorrect creator name found in hdf5 file'
        storage_mode_temp = hf_noise.attrs['storage_mode']
        assert isinstance(storage_mode_temp, (int, np.integer))
        storage_mode = int(storage_mode_temp)
        itrn_temp = hf_noise.attrs['itrn']
        assert isinstance(itrn_temp, (int, np.integer))
        self._itrn = int(itrn_temp)
        itr_save_temp = hf_noise.attrs['itr_save']
        assert isinstance(itr_save_temp, (int, np.integer))
        self._itr_save = int(itr_save_temp)
        cyclo_mode_temp = hf_noise.attrs['cyclo_mode']
        assert isinstance(cyclo_mode_temp, (int, np.integer))
        self.cyclo_mode = int(cyclo_mode_temp)
        assert hf_noise.attrs['noise_upper_name'] == self.noise_upper.__class__.__name__, 'incorrect noise upper name found in hdf5 file'
        assert hf_noise.attrs['noise_lower_name'] == self.noise_lower.__class__.__name__, 'incorrect noise lower name found in hdf5 file'
        assert hf_noise.attrs['noise_instrument_name'] == self.noise_instrument.__class__.__name__, 'incorrect noise instrument name found in hdf5 file'
        assert hf_noise.attrs['bgd_name'] == self.bgd.__class__.__name__, 'incorrect bgd name found in hdf5 file'
        assert hf_noise.attrs['fit_state_name'] == self._fit_state.__class__.__name__, 'incorrect fit state name found in hdf5 file'
        assert hf_noise.attrs['nt_lim_snr_name'] == self.nt_lim_snr.__class__.__name__, 'incorrect nt_lim_snr name found in hdf5 file'
        instrument_random_seed_temp = hf_noise.attrs['instrument_random_seed']
        assert isinstance(instrument_random_seed_temp, (int, np.integer))
        self._instrument_random_seed = int(instrument_random_seed_temp)
        assert hf_noise.attrs['wc_name'] == self.wc.__class__.__name__, 'incorrect wc name found in hdf5 file'
        assert hf_noise.attrs['lc_name'] == self._lc.__class__.__name__, 'incorrect lc name found in hdf5 file'
        assert hf_noise.attrs['ic_name'] == self._ic.__class__.__name__, 'incorrect ic name found in hdf5 file'

        self.S_inst_m = np.asarray(hf_noise['S_inst_m'], dtype=np.float64)
        self._idx_S_save = np.asarray(hf_noise['idx_S_save'], dtype=np.int64)
        self.S_record_upper = np.zeros((self._idx_S_save.size, self.wc.Nt, self.wc.Nf, self.bgd.nc_galaxy))
        self.S_record_lower = np.zeros((self._idx_S_save.size, self.wc.Nt, self.wc.Nf, self.bgd.nc_galaxy))
        self.S_final = np.zeros((self.wc.Nt, self.wc.Nf, self.bgd.nc_galaxy))

        if storage_mode in (1, 2):
            self.S_record_final = np.asarray(hf_noise['S_record_final'], dtype=np.float64)

        if storage_mode == 2:
            self.S_record_upper = np.asarray(hf_noise['S_record_upper'], dtype=np.float64)
            self.S_record_lower = np.asarray(hf_noise['S_record_lower'], dtype=np.float64)

        self.bgd.load_hdf5(hf_noise)

        hf_nt = hf_noise['nt_lim_snr']

        if not isinstance(hf_nt, h5py.Group):
            msg = 'Could not find group nt_lim_snr in hdf5 file'
            raise TypeError(msg)

        nx_min_temp = hf_nt.attrs['nx_min']
        assert isinstance(nx_min_temp, (int, np.integer))
        nx_max_temp = hf_nt.attrs['nx_max']
        assert isinstance(nx_max_temp, (int, np.integer))
        dx_temp = hf_nt.attrs['dx']
        assert isinstance(dx_temp, (float, np.floating))
        x_min_temp = hf_nt.attrs['x_min']
        assert isinstance(x_min_temp, (float, np.floating))
        self.nt_lim_snr = PixelGenericRange(int(nx_min_temp), int(nx_max_temp), float(dx_temp), float(x_min_temp))

        hf_wc = hf_noise['wc']
        if not isinstance(hf_wc, h5py.Group):
            msg = 'Could not find group wc in hdf5 file'
            raise TypeError(msg)

        for key in self.wc._fields:
            assert getattr(self.wc, key) == hf_wc.attrs[key], f'wc attribute {key} does not match saved value'

        hf_lc = hf_noise['lc']
        if not isinstance(hf_lc, h5py.Group):
            msg = 'Could not find group lc in hdf5 file'
            raise TypeError(msg)

        for key in self._lc._fields:
            assert getattr(self._lc, key) == hf_lc.attrs[key], f'lc attribute {key} does not match saved value'

        hf_ic = hf_noise['ic']
        if not isinstance(hf_ic, h5py.Group):
            msg = 'Could not find group ic in hdf5 file'
            raise TypeError(msg)

        for key in self._ic._fields:
            assert np.all(getattr(self._ic, key) == hf_ic.attrs[key]), f'ic attribute {key} does not match saved value'

        # just make new noise models, don't try to load them
        # TODO instead add assertions that the loaded models match the expected values stored in the files
        if not self.cyclo_mode:
            period_list = self._ic.period_list
        else:
            period_list = ()

        if self._itrn < self._ic.n_cyclo_switch:
            filter_periods = False
        else:
            filter_periods = not self.cyclo_mode

        S_upper = self.bgd.get_S_below_high(
            self.S_inst_m,
            self._ic.smooth_lengthf[self._itrn],
            filter_periods,
            period_list,
        )
        self.noise_upper = DiagonalNonstationaryDenseNoiseModel(S_upper, self.wc, prune=1, nc_snr=self._lc.nc_snr)

        filter_periods = not self.cyclo_mode
        S_lower = self.bgd.get_S_below_low(self.S_inst_m, self._ic.smooth_lengthf_fix, filter_periods, period_list)
        S_lower = np.asarray(np.min([S_lower, self.noise_upper.get_S()], axis=0), dtype=np.float64)

        self.noise_lower = DiagonalNonstationaryDenseNoiseModel(S_lower, self.wc, prune=1, nc_snr=self._lc.nc_snr)
        self.noise_instrument = DiagonalStationaryDenseNoiseModel(self.S_inst_m, self.wc, prune=1, nc_snr=self._lc.nc_snr, seed=self._instrument_random_seed)

    @override
    def log_state(self) -> None:
        """Perform any internal logging that should be done after advance_state is run."""
        if self._itr_save < self._idx_S_save.size and self._itrn - 1 == self._idx_S_save[self._itr_save]:
            self.S_record_upper[self._itr_save] = self.noise_upper.get_S()[:, :, :]
            self.S_record_lower[self._itr_save] = self.noise_lower.get_S()[:, :, :]
            self._itr_save += 1
        self.bgd.log_state(self.S_inst_m)

    @override
    def loop_finalize(self) -> None:
        """Perform any logic desired after convergence has been achieved and the loop ends."""
        self.S_final[:] = self.noise_upper.get_S()[:, :, :]

    @override
    def state_check(self) -> None:
        """Perform any sanity checks that should be performed at the end of each iteration."""
        self.bgd.state_check()

    @override
    def print_report(self) -> None:
        """Do any printing desired after convergence has been achieved and the loop ends."""
        res_mask = np.asarray(((self.noise_upper.get_S()[:, :, 0] - self.S_inst_m[:, 0]).mean(axis=0) > 0.1 * self.S_inst_m[:, 0]) & (self.S_inst_m[:, 0] > 0.0), dtype=np.bool_)
        galactic_below_high = self.bgd.get_galactic_below_high()
        noise_divide = np.sqrt(
            self.noise_upper.get_S()[self.nt_lim_snr.nx_min : self.nt_lim_snr.nx_max, res_mask, :2] - self.S_inst_m[res_mask, :2],
        )
        points_res = (
            galactic_below_high.reshape(self.wc.Nt, self.wc.Nf, self.bgd.nc_galaxy)[
                self.nt_lim_snr.nx_min : self.nt_lim_snr.nx_max,
                res_mask,
                :2,
            ]
            / noise_divide
        )
        n_points = points_res.size

        del noise_divide
        del galactic_below_high
        del res_mask
        unit_normal_res, a2score, mean_rat, std_rat = unit_normal_battery(
            points_res.flatten(),
            a2_cut=2.28,
            sig_thresh=5.0,
            do_assert=False,
        )
        del points_res
        if unit_normal_res:
            print(
                'Background PASSES normality: points=%12d A2=%3.5f, mean ratio=%3.5f, std ratio=%3.5f' % (n_points, a2score, mean_rat, std_rat),
            )
        else:
            print(
                'Background FAILS  normality: points=%12d A2=%3.5f, mean ratio=%3.5f, std ratio=%3.5f' % (n_points, a2score, mean_rat, std_rat),
            )

    @override
    def advance_state(self) -> None:
        """Handle any logic necessary to advance the state of the object to the next iteration."""
        noise_safe_upper = self._fit_state.get_noise_safe_upper()
        noise_safe_lower = self._fit_state.get_noise_safe_lower()

        if not self.cyclo_mode:
            period_list = self._ic.period_list
        else:
            period_list = ()

        if not noise_safe_upper:
            # assert not noise_safe_lower

            # don't use cyclostationary model until specified iteration
            if self._itrn < self._ic.n_cyclo_switch:
                filter_periods = False
            else:
                filter_periods = not self.cyclo_mode

            # use higher estimate of galactic bg
            S_upper = self.bgd.get_S_below_high(
                self.S_inst_m,
                self._ic.smooth_lengthf[self._itrn],
                filter_periods,
                period_list,
            )
            self.noise_upper = DiagonalNonstationaryDenseNoiseModel(S_upper, self.wc, prune=1, nc_snr=self._lc.nc_snr)

            del S_upper

        if not noise_safe_lower:
            # make sure this will always predict >= snrs to the actual spectrum in use
            # use lower estimate of galactic bg
            filter_periods = not self.cyclo_mode
            S_lower = self.bgd.get_S_below_low(self.S_inst_m, self._ic.smooth_lengthf_fix, filter_periods, period_list)
            S_lower = np.asarray(np.min([S_lower, self.noise_upper.get_S()], axis=0), dtype=np.float64)
            self.noise_lower = DiagonalNonstationaryDenseNoiseModel(S_lower, self.wc, prune=1, nc_snr=self._lc.nc_snr)
            del S_lower
        self._itrn += 1

    def get_instrument_realization(self, white_mode: int = 1) -> NDArray[np.floating]:
        """Get the realization of the instrument noise model based on the current random seed."""
        return self.noise_instrument.generate_dense_noise(white_mode=white_mode)
