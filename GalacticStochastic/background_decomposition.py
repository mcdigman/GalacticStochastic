"""Classes and functions to handle the decomposition of the galactic background."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.testing import assert_allclose

from GalacticStochastic.galactic_fit_helpers import get_S_cyclo
from WaveletWaveforms.sparse_waveform_functions import SparseWaveletWaveform, sparse_addition_helper

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from WaveletWaveforms.wdm_config import WDMWaveletConstants


class BGDecomposition:
    """Handle the internal decomposition of the galactic background."""

    def __init__(
        self,
        wc: WDMWaveletConstants,
        nc_galaxy: int,
        *,
        galactic_floor: NDArray[np.floating] | None = None,
        galactic_below: NDArray[np.floating] | None = None,
        galactic_undecided: NDArray[np.floating] | None = None,
        galactic_above: NDArray[np.floating] | None = None,
        track_mode: int = 1,
        storage_mode: int = 0,
    ) -> None:
        """
        Initialize a BGDecomposition object for handling the galactic background decomposition.

        Parameters
        ----------
        wc : WDMWaveletConstants
            Wavelet constants describing the time-frequency grid.
        nc_galaxy : int
            Number of tdi channels in the galactic background.
        galactic_floor : ndarray of float, optional
            Array containing the faintest (floor) component of the galactic background.
            If None, initialized to zeros.
        galactic_below : ndarray of float, optional
            Array containing the faint component of the galactic background.
            If None, initialized to zeros.
        galactic_undecided : ndarray of float, optional
            Array containing the undecided component of the galactic background.
            If None, initialized to zeros.
        galactic_above : ndarray of float, optional
            Array containing the bright (above threshold) component of the galactic background.
            If None, initialized to zeros.
        track_mode : int
            If nonzero, enables internal consistency checks and diagnostics. Default is 1.
        storage_mode : int
            Storage mode for the background. Only 0 is supported. Default is 0.

        Raises
        ------
        ValueError
            If an unrecognized storage mode is provided.
        AssertionError
            If provided arrays do not match the expected shapes.
        """
        if storage_mode != 0:
            msg = 'Unrecognized option for storage mode'
            raise ValueError(msg)

        self._wc: WDMWaveletConstants = wc
        self._storage_mode: int = storage_mode
        self._nc_galaxy: int = nc_galaxy
        self._shape1: tuple[int, int] = (wc.Nt * wc.Nf, self.nc_galaxy)
        self._shape2: tuple[int, int, int] = (wc.Nt, wc.Nf, self.nc_galaxy)

        if galactic_floor is None:
            self._galactic_floor: NDArray[np.floating] = np.zeros(self._shape1, dtype=np.float64)
        else:
            assert galactic_floor.shape == self._shape1
            self._galactic_floor = galactic_floor

        if galactic_below is None:
            self._galactic_below: NDArray[np.floating] = np.zeros(self._shape1)
        else:
            assert galactic_below.shape == self._shape1
            self._galactic_below = galactic_below

        if galactic_undecided is None:
            self._galactic_undecided: NDArray[np.floating] = np.zeros(self._shape1, dtype=np.float64)
        else:
            assert galactic_undecided.shape == self._shape1
            self._galactic_undecided = galactic_undecided

        if galactic_above is None:
            self._galactic_above: NDArray[np.floating] = np.zeros(self._shape1, dtype=np.float64)
        else:
            assert galactic_above.shape == self._shape1
            self._galactic_above = galactic_above

        self._galactic_total_cache: NDArray[np.floating] | None = None

        self._track_mode: int = track_mode

        self._power_galactic_undecided: list[NDArray[np.floating]] = []
        self._power_galactic_above: list[NDArray[np.floating]] = []
        self._power_galactic_below_low: list[NDArray[np.floating]] = []
        self._power_galactic_below_high: list[NDArray[np.floating]] = []
        self._power_galactic_total: list[NDArray[np.floating]] = []

    @property
    def nc_galaxy(self) -> int:
        """Number of tdi channels in the galactic background.

        Returns
        -------
        int
            The number of channels being tracked in the galactic background
        """
        return self._nc_galaxy

    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'background', group_mode: int = 0) -> h5py.Group:
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
            hf_background = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_background = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        hf_background.attrs['creator_name'] = self.__class__.__name__
        hf_background.attrs['storage_mode'] = self._storage_mode
        hf_background.attrs['track_mode'] = self._track_mode
        hf_background.attrs['nc_galaxy'] = self.nc_galaxy
        hf_background.attrs['shape1'] = self._shape1
        hf_background.attrs['shape2'] = self._shape2

        if self._storage_mode == 0:
            _ = hf_background.create_dataset(
                'galactic_below_low', data=self.get_galactic_below_low(), compression='gzip',
            )
            _ = hf_background.create_dataset(
                'galactic_above', data=self.get_galactic_coadd_resolvable(), compression='gzip',
            )
            _ = hf_background.create_dataset(
                'galactic_undecided', data=self.get_galactic_coadd_undecided(), compression='gzip',
            )
        return hf_background

    def load_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'background', group_mode: int = 0) -> None:
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
            hf_background = hf_in[group_name]
        elif group_mode == 1:
            hf_background = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        if not isinstance(hf_background, h5py.Group):
            msg = 'Could not find group ' + group_name + ' in hdf5 file'
            raise TypeError(msg)

        assert hf_background.attrs['creator_name'] == self.__class__.__name__, (
            'incorrect creator name found in hdf5 file'
        )

        storage_mode_temp = hf_background.attrs['storage_mode']
        assert isinstance(storage_mode_temp, (int, np.integer))
        self._storage_mode = int(storage_mode_temp)
        track_mode_temp = hf_background.attrs['track_mode']
        assert isinstance(track_mode_temp, (int, np.integer))
        self._track_mode = int(track_mode_temp)
        nc_galaxy_temp = hf_background.attrs['nc_galaxy']
        assert isinstance(nc_galaxy_temp, (int, np.integer))

        nc_galaxy_loaded = int(nc_galaxy_temp)
        if nc_galaxy_loaded != self.nc_galaxy:
            msg = f'nc_galaxy does not match: {nc_galaxy_loaded} != {self.nc_galaxy}'
            raise ValueError(msg)

        shape1_temp = hf_background.attrs['shape1']
        assert isinstance(shape1_temp, (tuple, list, np.ndarray))
        shape1_loaded = tuple(int(x) for x in shape1_temp)

        if shape1_loaded != self._shape1:
            msg = f'shape1 does not match: {shape1_loaded} != {self._shape1}'
            raise ValueError(msg)

        shape2_temp = hf_background.attrs['shape2']
        assert isinstance(shape2_temp, (tuple, list, np.ndarray))
        shape2_loaded = tuple(int(x) for x in shape2_temp)

        if shape2_loaded != self._shape2:
            msg = f'shape2 does not match: {shape2_loaded} != {self._shape2}'
            raise ValueError(msg)

        if self._storage_mode == 0:
            self._galactic_below[:] = 0.0  # reset to zero, since we cannot separate the two components

            galactic_below_low_temp = hf_background['galactic_below_low']
            assert isinstance(galactic_below_low_temp, h5py.Dataset)
            galactic_below_low = np.asarray(galactic_below_low_temp)
            assert galactic_below_low.shape == self._shape1, 'Incorrect shape for galactic_below_low in hdf5 file'
            self._galactic_floor[:] = galactic_below_low

            galactic_above_temp = hf_background['galactic_above']
            assert isinstance(galactic_above_temp, h5py.Dataset)
            galactic_above = np.asarray(galactic_above_temp)
            assert galactic_above.shape == self._shape1, 'Incorrect shape for galactic_above in hdf5 file'
            self._galactic_above[:] = galactic_above

            galactic_undecided_temp = hf_background['galactic_undecided']
            assert isinstance(galactic_undecided_temp, h5py.Dataset)
            galactic_undecided = np.asarray(galactic_undecided_temp)
            assert galactic_undecided.shape == self._shape1, 'Incorrect shape for galactic_undecided in hdf5 file'
            self._galactic_undecided[:] = galactic_undecided

            self._galactic_total_cache = None  # reset cache since we have new data
            # reset diagnostics
            self._power_galactic_above = []
            self._power_galactic_undecided = []
            self._power_galactic_below_low = []
            self._power_galactic_below_high = []
            self._power_galactic_total = []

    def _output_shape_select(
        self, representation: NDArray[np.floating], *, shape_mode: int = 0,
    ) -> NDArray[np.floating]:
        r"""
        Select and reshape the output array to the desired shape for galactic background components.

        This function reshapes the input array representing a galactic background component
        to one of the supported output shapes, depending on the specified `shape_mode`.

        Parameters
        ----------
        representation : NDArray[np.floating]
            Input array containing the galactic background component data.
            Must have a size compatible with the expected shapes.
        shape_mode : int
            Output shape mode:
            - 0: Reshape to (Nt \* Nf, nc_galaxy)
            - 1: Reshape to (Nt, Nf, nc_galaxy)
            Any other value raises NotImplementedError.
            Default is 0.

        Returns
        -------
        NDArray[np.floating]
            The reshaped array in the selected output format.

        Raises
        ------
        NotImplementedError
            If an unsupported `shape_mode` is provided.
        """
        if shape_mode == 0:
            return representation.reshape(self._shape1)
        if shape_mode == 1:
            return representation.reshape(self._shape2)
        msg = 'No implementation for given shape mode'
        raise NotImplementedError(msg)

    def get_galactic_total(self, *, bypass_check: bool = False, shape_mode: int = 0) -> NDArray[np.floating]:
        r"""
        Get the sum of the entire galactic signal, including detectable binaries.

        Parameters
        ----------
        bypass_check : bool
            If True, skip the internal state consistency check. Default is False.
        shape_mode : int
            Output shape mode:
            - 0: Reshape to (Nt \* Nf, nc_galaxy)
            - 1: Reshape to (Nt, Nf, nc_galaxy)
            Default is 0.

        Returns
        -------
        NDArray[np.floating]
            Array containing the total galactic signal in the selected output shape.
        """
        if not bypass_check:
            self.state_check()
        res = self.get_galactic_below_high(bypass_check=True) + self._galactic_above
        return self._output_shape_select(res, shape_mode=shape_mode)

    def get_galactic_below_high(self, *, bypass_check: bool = False, shape_mode: int = 0) -> NDArray[np.floating]:
        r"""
        Get the upper estimate of the unresolvable signal from the galactic background.

        Assume that the undecided part of the signal *is* part of the unresolvable background

        Parameters
        ----------
        bypass_check : bool
            If True, skip the internal state consistency check. Default is False.
        shape_mode : int
            Output shape mode:
            - 0: Reshape to (Nt \* Nf, nc_galaxy)
            - 1: Reshape to (Nt, Nf, nc_galaxy)
            Default is 0.

        Returns
        -------
        NDArray[np.floating]
            Array containing the total galactic signal in the selected output shape.
        """
        if not bypass_check:
            self.state_check()
        res = self.get_galactic_below_low(bypass_check=True) + self._galactic_undecided
        return self._output_shape_select(res, shape_mode=shape_mode)

    def get_galactic_below_low(self, *, bypass_check: bool = False, shape_mode: int = 0) -> NDArray[np.floating]:
        r"""
        Get the lower estimate of the unresolvable signal from the galactic background.

        Assume that the undecided part of the signal is *not* part of the unresolvable background

        Parameters
        ----------
        bypass_check : bool
            If True, skip the internal state consistency check. Default is False.
        shape_mode : int
            Output shape mode:
            - 0: Reshape to (Nt \* Nf, nc_galaxy)
            - 1: Reshape to (Nt, Nf, nc_galaxy)
            Default is 0.

        Returns
        -------
        NDArray[np.floating]
            Array containing the total galactic signal in the selected output shape.
        """
        if not bypass_check:
            self.state_check()
        res = self._galactic_floor + self._galactic_below
        return self._output_shape_select(res, shape_mode=shape_mode)

    def get_galactic_coadd_resolvable(self, *, bypass_check: bool = False, shape_mode: int = 0) -> NDArray[np.floating]:
        r"""
        Get coadded signal from the resolvable (bright) galactic binaries.

        Parameters
        ----------
        bypass_check : bool
            If True, skip the internal state consistency check. Default is False.
        shape_mode : int
            Output shape mode:
            - 0: Reshape to (Nt \* Nf, nc_galaxy)
            - 1: Reshape to (Nt, Nf, nc_galaxy)
            Default is 0.

        Returns
        -------
        NDArray[np.floating]
            Array containing the total galactic signal in the selected output shape.
        """
        if not bypass_check:
            self.state_check()
        res = self._galactic_above
        return self._output_shape_select(res, shape_mode=shape_mode)

    def get_galactic_coadd_undecided(self, *, bypass_check: bool = False, shape_mode: int = 0) -> NDArray[np.floating]:
        r"""
        Get coadded signal from binaries whose status as bright or faint has not yet been decided.

        Parameters
        ----------
        bypass_check : bool
            If True, skip the internal state consistency check. Default is False.
        shape_mode : int
            Output shape mode:
            - 0: Reshape to (Nt \* Nf, nc_galaxy)
            - 1: Reshape to (Nt, Nf, nc_galaxy)
            Default is 0.

        Returns
        -------
        NDArray[np.floating]
            Array containing the total galactic signal in the selected output shape.
        """
        if not bypass_check:
            self.state_check()
        res = self._galactic_undecided
        return self._output_shape_select(res, shape_mode=shape_mode)

    def get_galactic_coadd_floor(self, *, bypass_check: bool = False, shape_mode: int = 0) -> NDArray[np.floating]:
        r"""
        Get coadded signal from the faintest set of galactic binaries.

        Parameters
        ----------
        bypass_check : bool
            If True, skip the internal state consistency check. Default is False.
        shape_mode : int
            Output shape mode:
            - 0: Reshape to (Nt \* Nf, nc_galaxy)
            - 1: Reshape to (Nt, Nf, nc_galaxy)
            Default is 0.

        Returns
        -------
        NDArray[np.floating]
            Array containing the total galactic signal in the selected output shape.
        """
        if not bypass_check:
            self.state_check()
        res = self._galactic_floor
        return self._output_shape_select(res, shape_mode=shape_mode)

    def state_check(self) -> None:
        """If the total recorded galactic signal is cached, check that the total not changed much.

        Otherwise, cache the current total so future runs can check if it has changed.
        """
        if self._track_mode:
            if self._galactic_total_cache is None:
                assert np.all(self._galactic_below == 0.0)
                self._galactic_total_cache = self.get_galactic_total(bypass_check=True)
            else:
                # check all contributions to the total signal are tracked accurately
                assert_allclose(
                    self._galactic_total_cache,
                    self.get_galactic_total(bypass_check=True),
                    atol=1.0e-300,
                    rtol=1.0e-6,
                )

    def log_state(self, S_mean: NDArray[np.floating]) -> None:
        """Record any diagnostics we want to track about this iteration.

        Parameters
        ----------
        S_mean : NDArray[np.floating]
            Mean power spectrum used for normalization.
        """
        power_undecided = np.sum(
            np.sum((self.get_galactic_coadd_undecided(bypass_check=True, shape_mode=1) ** 2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        )
        power_above = np.sum(
            np.sum((self.get_galactic_coadd_resolvable(bypass_check=True, shape_mode=1) ** 2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        )

        power_total = np.sum(
            np.sum((self.get_galactic_total(bypass_check=True, shape_mode=1) ** 2)[:, 1:, :], axis=0) / S_mean[1:, :],
            axis=0,
        )
        power_below_high = np.sum(
            np.sum((self.get_galactic_below_high(bypass_check=True, shape_mode=1) ** 2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        )
        power_below_low = np.sum(
            np.sum((self.get_galactic_below_low(bypass_check=True, shape_mode=1) ** 2)[:, 1:, :], axis=0)
            / S_mean[1:, :],
            axis=0,
        )

        self._power_galactic_undecided.append(np.asarray(power_undecided, dtype=np.float64))
        self._power_galactic_above.append(np.asarray(power_above, dtype=np.float64))

        self._power_galactic_total.append(np.asarray(power_total, dtype=np.float64))
        self._power_galactic_below_high.append(np.asarray(power_below_high, dtype=np.float64))
        self._power_galactic_below_low.append(np.asarray(power_below_low, dtype=np.float64))

    def clear_undecided(self) -> None:
        """Clear the undecided part of the galactic spectrum."""
        self._galactic_undecided[:] = 0.0

    def clear_above(self) -> None:
        """Clear the bright part of the galactic spectrum."""
        self._galactic_above[:] = 0.0

    def clear_below(self) -> None:
        """Clear the faint part of the galactic spectrum."""
        self._galactic_below[:] = 0.0

    def get_S_below_high(
        self,
        S_mean: NDArray[np.floating],
        smooth_lengthf: float,
        filter_periods: int,
        period_list: tuple[int, ...] | tuple[np.floating, ...],
    ) -> NDArray[np.floating]:
        """
        Get the upper estimate of the galactic power spectrum.

        This method computes the power spectral density (PSD) for the upper estimate of the unresolvable
        galactic background, assuming the undecided component is included as part of the unresolvable signal.
        The PSD is computed using the provided mean spectrum, smoothing length, and filter parameters.

        Parameters
        ----------
        S_mean : NDArray[np.floating]
            Mean power spectrum used for normalization.
        smooth_lengthf : float
            Smoothing length in the frequency domain.
        filter_periods : int
            Number of periods to use for filtering.
        period_list : tuple of int or float
            List of periods to consider in the cyclostationary analysis.

        Returns
        -------
        NDArray[np.floating]
            The estimated power spectral density for the upper bound of the unresolvable galactic background.
        """
        galactic_loc = self.get_galactic_below_high(bypass_check=True)
        S, _, _, _, _ = get_S_cyclo(
            galactic_loc.reshape(self._shape2),
            S_mean,
            self._wc.DT,
            smooth_lengthf,
            filter_periods,
            period_list=period_list,
        )
        return S

    def get_S_below_low(
        self,
        S_mean: NDArray[np.floating],
        smooth_lengthf: float,
        filter_periods: int,
        period_list: tuple[int, ...] | tuple[np.floating, ...],
    ) -> NDArray[np.floating]:
        """
        Get the lower estimate of the galactic power spectrum.

        This method computes the power spectral density (PSD) for the lower estimate of the unresolvable
        galactic background, assuming the undecided component is excluded from the unresolvable signal.
        The PSD is computed using the provided mean spectrum, smoothing length, and filter parameters.

        Parameters
        ----------
        S_mean : NDArray[np.floating]
            Mean power spectrum used for normalization.
        smooth_lengthf : float
            Smoothing length in the frequency domain.
        filter_periods : int
            Number of periods to use for filtering.
        period_list : tuple of int or float
            List of periods to consider in the cyclostationary analysis.

        Returns
        -------
        NDArray[np.floating]
            The estimated power spectral density for the lower bound of the unresolvable galactic background.
        """
        galactic_loc = self.get_galactic_below_low(bypass_check=True)
        S, _, _, _, _ = get_S_cyclo(
            galactic_loc.reshape(self._shape2),
            S_mean,
            self._wc.DT,
            smooth_lengthf,
            filter_periods,
            period_list=period_list,
        )
        return S

    def add_undecided(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """
        Add a binary to the undecided component of the galactic background.

        Parameters
        ----------
        wavelet_waveform : SparseWaveletWaveform
            The sparse wavelet waveform representing the binary to be added.
        """
        sparse_addition_helper(wavelet_waveform, self._galactic_undecided)

    def add_floor(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the floor component of the galactic background.

        Parameters
        ----------
        wavelet_waveform : SparseWaveletWaveform
            The sparse wavelet waveform representing the binary to be added.
        """
        sparse_addition_helper(wavelet_waveform, self._galactic_floor)

    def add_faint(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the faint component of the galactic background.

        Parameters
        ----------
        wavelet_waveform : SparseWaveletWaveform
            The sparse wavelet waveform representing the binary to be added.
        """
        sparse_addition_helper(wavelet_waveform, self._galactic_below)

    def add_bright(self, wavelet_waveform: SparseWaveletWaveform) -> None:
        """Add a binary to the bright component of the galactic background.

        Parameters
        ----------
        wavelet_waveform : SparseWaveletWaveform
            The sparse wavelet waveform representing the binary to be added.
        """
        sparse_addition_helper(wavelet_waveform, self._galactic_above)


def _check_correct_component_shape(
    nc: int, wc: WDMWaveletConstants, galactic_component: NDArray[np.floating], *, shape_mode: int = 0,
) -> NDArray[np.floating]:
    """
    Check and reshape a galactic background component to the correct shape.

    This function verifies that the input galactic component array has a size compatible
    with the expected shapes based on the wavelet constants and number of channels.
    If necessary, it reshapes the array to the desired output shape specified by `shape_mode`.

    Parameters
    ----------
    nc : int
        Number of galactic background channels.
    wc : WDMWaveletConstants
        Wavelet constants describing the time-frequency grid.
    galactic_component : NDArray[np.floating]
        Input array representing a galactic background component.
        Must have a size of `wc.Nt * wc.Nf * nc`.
    shape_mode : int
        Output shape mode:
        - 0: Reshape to (Nt * Nf, nc)
        - 1: Reshape to (Nt, Nf, nc)
        - 2: Reshape to (Nt * Nf * nc,)
        Default is 0.

    Returns
    -------
    NDArray[np.floating]
        The galactic component array reshaped to the selected output format.

    Raises
    ------
    AssertionError
        If the input array does not have the expected size or an allowed shape.
    ValueError
        If an invalid `shape_mode` is provided.
    """
    assert galactic_component.size == wc.Nt * wc.Nf * nc, 'Incorrectly sized galaxy component'

    shape1 = (wc.Nt * wc.Nf, nc)
    shape2 = (wc.Nt, wc.Nf, nc)
    shape3 = (wc.Nt * wc.Nf * nc,)
    shapes_allowed = (shape1, shape2, shape3)

    if shape_mode >= len(shapes_allowed):
        msg = f'Invalid shape mode {shape_mode}'
        raise ValueError(msg)

    shape_got = galactic_component.shape
    assert shape_got in shapes_allowed, 'Unrecognized shape for galactic background component'

    if shape_got != shapes_allowed[shape_mode]:
        galactic_component = galactic_component.reshape(shapes_allowed[shape_mode])

    return galactic_component
