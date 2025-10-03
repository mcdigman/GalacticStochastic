"""Functions for reading and writing galactic binary parameter files and iterative fit results."""

import hashlib
from pathlib import Path
from typing import Any
from warnings import warn

import h5py
import numpy as np
from numpy.typing import NDArray

import GalacticStochastic.global_const as gc
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from WaveletWaveforms.wdm_config import WDMWaveletConstants

n_par_gb = 8
labels_gb = [
    'Amplitude',
    'EclipticLatitude',
    'EclipticLongitude',
    'Frequency',
    'FrequencyDerivative',
    'Inclination',
    'InitialPhase',
    'Polarization',
]


def get_galaxy_filename(config: dict[str, Any]) -> str:
    """Get the filename where the binaries in the galaxy are stored.

    Parameters
    ----------
    config : dict of str to Any
        Configuration dictionary containing file paths.

    Returns
    -------
    str
        Full path to the galaxy file.
    """
    config_files: dict[str, str] = config['files']
    return str(config_files['galaxy_dir']) + str(config_files['galaxy_file'])


def get_processed_galactic_filename(config: dict[str, Any], wc: WDMWaveletConstants, *, preprocess_mode: int = 2) -> str:
    """Get the filename where the iterative fit results are stored.

    Parameters
    ----------
    config : dict of str to Any
        Configuration dictionary containing file paths.
    wc : WDMWaveletConstants
        Wavelet constants describing the time-frequency grid.
    preprocess_mode : int
        Preprocessing mode used to determine the filename:
        - 0: Processed file (default).
        - 1: Pre-processed file.
        - 2: Re-processed file.

    Returns
    -------
    str
        Full path to the processed galactic binary file.
    """
    config_files: dict[str, str] = config['files']
    galaxy_dir = str(config_files['galaxy_dir'])
    if preprocess_mode == 0:
        file_prefix = str(config_files.get('processed_prefix', 'processed_iterations'))
    else:
        file_prefix = str(config_files.get('preprocessed_prefix', 'preprocessed_background'))
    return galaxy_dir + file_prefix + '_Nf=' + str(wc.Nf) + '_Nt=' + str(wc.Nt) + ('_dt=%.2f' % (wc.dt)) + '.hdf5'


def _source_mask_read_helper(hf_sky: h5py.Group, key: str, fmin: float, fmax: float) -> tuple[int, NDArray[np.floating]]:
    """
    Read and filter galactic binary parameters from an HDF5 group by frequency range.

    This helper function accesses a specific binary category within the HDF5 file,
    applies a frequency mask, and extracts the parameters for binaries whose frequencies
    fall within the specified range.

    Parameters
    ----------
    hf_sky : h5py.Group
        The HDF5 group containing binary categories and their parameter datasets.
    key : str
        The name of the binary category group to read (e.g., 'dgb', 'igb', 'vgb').
    fmin : float
        Minimum frequency (inclusive) for selecting binaries.
    fmax : float
        Maximum frequency (exclusive) for selecting binaries.

    Returns
    -------
    n_loc : int
        Number of binaries in the selected category within the specified frequency range.
    params : NDArray[np.floating]
        Array of shape (n_loc, n_par_gb) containing the parameters for the selected binaries.

    Raises
    ------
    TypeError
        If the HDF5 file structure does not match the expected format or data types.
    """
    hf_loc = hf_sky[key]

    if not isinstance(hf_loc, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    hf_cat = hf_loc['cat']

    if not isinstance(hf_cat, (h5py.Group, h5py.Dataset)):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    freqs = np.asarray(hf_cat['Frequency'])  # pyright: ignore[reportArgumentType]

    if not np.issubdtype(freqs.dtype, np.floating):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    mask = np.zeros(freqs.shape, dtype=np.bool_)
    mask[:] = (freqs > fmin) & (freqs < fmax)
    n_loc = int(np.sum(1 * mask))
    params = np.zeros((n_loc, n_par_gb), dtype=np.float64)
    for itrl in range(n_par_gb):
        hf_param = hf_cat[labels_gb[itrl]]  # pyright: ignore[reportArgumentType]
        if not isinstance(hf_param, (h5py.Dataset, np.ndarray)):
            msg = 'Unrecognized hdf5 file format'
            raise TypeError(msg)
        params[:, itrl] = hf_param[mask]

    return n_loc, params


def get_full_galactic_params(config: dict[str, Any]) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """
    Retrieve the full set of galactic binary parameters from the galaxy file.

    This function reads the galactic binary parameters from an HDF5 file, applying frequency range
    filtering and combining multiple binary categories (e.g., detached, interacting, verification)
    into a single parameter array. It performs validation checks on the parameter values and
    issues warnings for unexpected or suspicious data.

    The parameter array in the input file is assumed to be ordered as follows:
        (Amplitude, EclipticLatitude, EclipticLongitude, Frequency,
         FrequencyDerivative, Inclination, InitialPhase, Polarization)

    Parameters
    ----------
    config : dict of str to Any
        Configuration dictionary containing file paths, frequency limits, and component list.

    Returns
    -------
    params_gb : NDArray[np.floating]
        Array of shape (n_binaries, n_par_gb) containing the parameters for all selected galactic binaries.
    ns_got : NDArray[np.integer]
        Array containing the number of binaries found in each category.

    Raises
    ------
    AssertionError
        If parameter values are out of expected bounds or contain non-finite values.
    TypeError
        If the HDF5 file structure does not match the expected format.
    UserWarning
        If some parameters are always zero or have suspicious values (issued as warnings).

    Notes
    -----
    The function expects the HDF5 file to contain groups for each binary category under the 'sky' group,
    with datasets for each parameter. Frequency filtering is applied based on the configuration.
    """
    # dgb is detached galactic binaries, igb is interacting galactic binaries, vgb is verification
    config_ic: dict[str, Any] = config['iterative_fit_constants']
    categories: list[str] = config_ic.get('component_list', ['dgb', 'igb', 'vgb'])
    fmin: float = float(config_ic.get('fmin_binary', 1.0e-8))
    fmax: float = float(config_ic.get('fmax_binary', 1.0e0))
    assert fmin <= fmax
    full_galactic_params_filename = get_galaxy_filename(config)
    filename = full_galactic_params_filename

    hf_in = h5py.File(filename, 'r')

    hf_sky = hf_in['sky']

    if not isinstance(hf_sky, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    ns_got = np.zeros(len(categories), dtype=np.int64)
    params_got: list[NDArray[np.floating]] = []
    for itr, label in enumerate(categories):
        ns_got[itr], params_loc = _source_mask_read_helper(hf_sky, label, fmin, fmax)
        if label == 'dgb' and np.any(params_loc[:, 4] < 0.0):
            warn('Some binaries reported as detached have negative frequency derivatives', stacklevel=2)
        params_got.append(params_loc)
        print(str(label), str(ns_got[itr]))

    n_tot = int(ns_got.sum())

    print('total', n_tot)

    params_gb = np.zeros((n_tot, n_par_gb))

    n_old = 0
    for itr in range(len(categories)):
        n_cur = n_old + int(ns_got[itr])
        params_gb[n_old:n_cur, :] = params_got[itr]
        n_old = n_cur

    hf_in.close()
    assert np.all(np.isfinite(params_gb)), 'Some binaries have non-finite parameters'
    assert not np.any(np.all(params_gb == 0.0, axis=1)), 'Some binaries have zero for all parameters'
    if np.any(np.all(params_gb == 0.0, axis=0)):
        warn('Some parameters are always zero', stacklevel=2)
    assert np.all(params_gb[:, 0] > 0.0), 'Some binaries have non-positive amplitudes'
    assert np.all((-np.pi / 2 <= params_gb[:, 1]) & (params_gb[:, 1] <= np.pi / 2)), 'Ecliptic latitude not bounded in expected range'
    assert np.all((params_gb[:, 2] >= 0.0) & (params_gb[:, 2] <= 2 * np.pi)), 'Ecliptic longitude not bounded in expected range'
    assert np.all(params_gb[:, 3] > 0.0), 'Some binaries have non-positive frequencies'
    if np.any(np.abs(params_gb[:, 4]) * gc.SECSYEAR * 10 > 0.001):
        warn('Some binaries have large frequency derivatives', stacklevel=2)
    print('Largest frequency derivative', np.max(np.abs(params_gb[:, 4]) * gc.SECSYEAR * 10))
    assert np.all((params_gb[:, 5] >= 0.0) & (params_gb[:, 5] <= np.pi)), 'Inclination not bounded in expected range'
    assert np.all((params_gb[:, 6] >= 0.0) & (params_gb[:, 6] <= 2 * np.pi)), 'Initial phase not bounded in expected range'
    assert np.all((params_gb[:, 7] >= 0.0) & (params_gb[:, 7] <= 2 * np.pi)), 'Polarization phase not bounded in expected range'
    return params_gb, ns_got


def load_processed_galactic_file(
    ifm: IterativeFitManager,
    config: dict[str, Any],
    ic: IterationConfig,
    wc: WDMWaveletConstants,
    nt_lim_snr: tuple[int, int] = (0, -1),
    *,
    cyclo_mode: int = 1,
    preprocess_mode: int = 0,
) -> None:
    """
    Load the results of a previous iterative fit procedure from an HDF5 file.

    This function locates and opens the HDF5 file containing the results of a previous iterative fit,
    navigates to the appropriate group based on the SNR threshold, time-frequency range, and cyclostationary mode,
    and loads the results into the provided IterativeFitManager instance.

    Parameters
    ----------
    ifm : IterativeFitManager
        The manager object into which the loaded results will be stored.
    config : dict of str to Any
        Configuration dictionary containing file paths and iterative fit settings.
    ic : IterationConfig
        Configuration object specifying the parameters for the iterative fit.
    wc : WDMWaveletConstants
        Wavelet constants describing the time-frequency grid.
    nt_lim_snr : tuple of int
        Tuple specifying the time-frequency pixel range to use. Defaults to (0, -1), which uses the full range.
    cyclo_mode : int
        Cyclostationary mode key used to select the correct HDF5 group (default is 1).
    preprocess_mode : int
        Preprocessing mode used to determine the input filename (default is 0).

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file or group cannot be found.
    TypeError
        If the HDF5 file structure does not match the expected format.

    Notes
    -----
    The function expects the HDF5 file to be organized by SNR threshold, time-frequency range, and cyclostationary mode.
    The loaded data is passed to the `load_hdf5` method of the provided IterativeFitManager.
    """
    if preprocess_mode == 1:
        snr_thresh = ic.snr_min_preprocess
    elif preprocess_mode == 2:
        snr_thresh = ic.snr_min_reprocess
    else:
        snr_thresh = ic.snr_thresh
    filename_in = get_processed_galactic_filename(config, wc, preprocess_mode=preprocess_mode)
    if nt_lim_snr == (0, -1):
        nt_range: tuple[int, int] = (0, wc.Nt)
    else:
        nt_range = nt_lim_snr

    cyclo_key = str(cyclo_mode)

    try:
        hf_in = h5py.File(filename_in, 'r')
        hf_itr = hf_in['iteration_results']
        if not isinstance(hf_itr, h5py.Group):
            msg = 'Unrecognized hdf5 file format'
            raise TypeError(msg)
        hf_snr = hf_itr[str(snr_thresh)]
        if not isinstance(hf_snr, h5py.Group):
            msg = 'Unrecognized hdf5 file format'
            raise TypeError(msg)
        print(filename_in, preprocess_mode, nt_range, cyclo_key)
        hf_nt = hf_snr[str(nt_range)]
        if not isinstance(hf_nt, h5py.Group):
            msg = 'Unrecognized hdf5 file format'
            raise TypeError(msg)
        hf_run = hf_nt[cyclo_key]
        if not isinstance(hf_run, h5py.Group):
            msg = 'Unrecognized hdf5 file format'
            raise TypeError(msg)
    except (OSError, KeyError) as e:
        msg = f'Could not find processed galactic binary file {filename_in} with snr_thresh={snr_thresh}, nt_range={nt_range}, cyclo_mode={cyclo_mode}'
        raise FileNotFoundError(msg) from e

    ifm.load_hdf5(hf_run)


def store_processed_gb_file(
    config: dict[str, Any],
    wc: WDMWaveletConstants,
    ifm: IterativeFitManager,
    *,
    write_mode: int = 0,
    preprocess_mode: int = 0,
    hash_mode: int = 1,
) -> None:
    """
    Store the results of an iterative fit to an HDF5 file.

    This function writes the results of the iterative fit, including configuration and provenance information,
    to an HDF5 file. It manages file versioning, verifies file integrity using SHA256 checksums, and archives
    relevant metadata for reproducibility.

    Parameters
    ----------
    config : dict of str to Any
        Configuration dictionary containing file paths and iterative fit settings.
    wc : WDMWaveletConstants
        Wavelet constants describing the time-frequency grid.
    ifm : IterativeFitManager
        Manager object containing the results of the iterative fit to be stored.
    write_mode : int
        File writing mode:
        - 0: Overwrite existing group if present (default).
        - 1: Abort if group exists to avoid overwriting.
        - 2: Create a new file, entirely overwriting any existing file.
    preprocess_mode : int
        Preprocessing mode used to determine the output filename (default is 0).
    hash_mode : int
        Hash verification mode:
        - 1: Perform SHA256 checksum verification and recording (default).
        - 0: Skip hash verification, but still record checksums.

    Raises
    ------
    NotImplementedError
        If an unrecognized write_mode is provided.
    ValueError
        If an unexpected state is encountered when writing the HDF5 file.
    AssertionError
        If file integrity checks fail (e.g., mismatched SHA256 checksums).
    TypeError
        If the HDF5 file structure does not match the expected format.
    """
    ic = ifm.ic

    filename_gb_init = get_processed_galactic_filename(config, wc, preprocess_mode=1)
    filename_out = get_processed_galactic_filename(config, wc, preprocess_mode=preprocess_mode)

    if preprocess_mode == 1:
        assert filename_gb_init == filename_out
        snr_thresh: float = ic.snr_min_preprocess
        if write_mode != 2:
            msg = 'Modifying pre-processed file without overwriting will break checksums'
            warn(msg, stacklevel=2)
        skip_init = True
    elif preprocess_mode == 2:
        snr_thresh = ic.snr_min_reprocess
        skip_init = True
    else:
        snr_thresh = ic.snr_thresh
        skip_init = False

    if preprocess_mode == 2:
        assert write_mode != 2, 'Cannot overwrite pre-processed file when re-processing it'

    sha256_hex_orig = None
    if write_mode != 2:
        try:
            # if we are re-processing a pre-processed file, record the old hash before we modified it
            with Path(filename_out).open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256')
                sha256_hex_orig = digest.hexdigest()
        except FileNotFoundError:
            pass

    if write_mode in (0, 1):
        hf_out = h5py.File(filename_out, 'a')
    elif write_mode == 2:
        hf_out = h5py.File(filename_out, 'w')
    else:
        msg = 'Unrecognized option for write_mode'
        raise NotImplementedError(msg)

    noise_manager = ifm.noise_manager
    nt_lim_snr = noise_manager.nt_lim_snr
    cyclo_mode = noise_manager.cyclo_mode

    filename_source_gb = get_galaxy_filename(config)
    filename_config: str = config.get('toml_filename', 'not_recorded')

    # save the configuration filenames to the object and raise an error if they are already there but do not match
    if (not skip_init) and (filename_gb_init in hf_out.attrs):
        assert hf_out.attrs['filename_gb_init'] == filename_gb_init
    if filename_config in hf_out.attrs:
        assert hf_out.attrs['filename_config'] == filename_config
    if filename_source_gb in hf_out.attrs:
        assert hf_out.attrs['filename_source_gb'] == filename_source_gb

    # get the hdf5 group that corresponds to this snr threshold, nt range, and value for cyclo_mode
    # creating sub groups as necessary if they do not already exist

    hf_itr = hf_out.require_group('iteration_results')

    hf_snr = hf_itr.require_group(str(snr_thresh))
    del hf_itr

    nt_range: tuple[int, int] = (nt_lim_snr.nx_min, nt_lim_snr.nx_max)

    hf_nt = hf_snr.require_group(str(nt_range))
    del hf_snr

    cyclo_key = str(cyclo_mode)
    if cyclo_key in hf_nt:
        # this exact group already exists
        if write_mode == 0:
            # delete the existing group and overwrite
            del hf_nt[cyclo_key]
            print('Overwriting exsting hdf5 object')
        elif write_mode == 1:
            warn('Requested hdf5 object already exists, aborting write to avoid overwriting', stacklevel=2)
            hf_out.close()
            return
        else:
            msg = 'Unexpected state when writing hdf5 file'
            raise ValueError(msg)

    # if we make it here, we are changing something about the file, so record the old hash if we had it
    if sha256_hex_orig is not None:
        old_hash_log: Any = hf_out.attrs.get('filename_out_sha256_history', [])
        if not isinstance(old_hash_log, (list, np.ndarray)):
            msg = 'filename_out_sha256_history must be a list or numpy array'
            raise TypeError(msg)
        old_hash_list = list(old_hash_log)
        old_hash_list.append(sha256_hex_orig)
        hf_out.attrs['filename_out_sha256_history'] = old_hash_list
        del old_hash_log
        del old_hash_list
        del sha256_hex_orig

    hf_run = hf_nt.require_group(cyclo_key)

    # Compute the sha256 checksum of the source galactic binary file and the pre-processed file.
    # If they have been previously recorded in the hdf5 file, check if they match.
    # Otherwise, record them.

    with Path(filename_source_gb).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256')
        sha256_hex_gb = digest.hexdigest()
        print('Computed sha256 checksum of source galactic binary file', sha256_hex_gb)

    if hash_mode == 1 and 'filename_source_gb_sha256' in hf_out.attrs:
        assert hf_out.attrs['filename_source_gb_sha256'] == sha256_hex_gb, 'Processed file was generated from a different source galactic binary file than the one currently specified'
        print('Processed file source galactic binary sha256 checksum matches current source galactic binary file')

    hf_out.attrs['filename_source_gb_sha256'] = sha256_hex_gb

    prelim_hash_list = []

    if (not skip_init) and hash_mode == 1:
        with h5py.File(filename_gb_init, 'r') as hf_prelim:
            try:
                sha256_hex_gb_prelim = hf_prelim.attrs['filename_source_gb_sha256']
                assert sha256_hex_gb_prelim == sha256_hex_gb, 'Pre-processed file was generated from a different source galactic binary file than the one currently specified'
                print('Pre-processed file source galactic binary sha256 checksum matches current source galactic binary file')
                del sha256_hex_gb_prelim
            except KeyError:
                warn('Pre-processed file did not record a sha256 checksum, cannot verify it matches source galactic binary file', stacklevel=2)

            # get old recorded hashes from the pre-processed file if they exist
            prelim_hash_log: Any = hf_prelim.attrs.get('filename_out_sha256_history', [])
            if not isinstance(prelim_hash_log, (list, np.ndarray)):
                msg = 'filename_out_sha256_history must be a list or numpy array'
                raise TypeError(msg)
            prelim_hash_list = list(prelim_hash_log)

    del sha256_hex_gb

    if not skip_init:

        with Path(filename_gb_init).open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256')
            sha256_hex_gb_init = digest.hexdigest()
            print('Computed sha256 checksum of pre-processed file', sha256_hex_gb_init)

        if hash_mode == 1 and 'filename_gb_init_sha256' in hf_out.attrs:
            sha256_hex_gb_prelim_expect = hf_out.attrs['filename_gb_init_sha256']
            prelim_hash_list.append(sha256_hex_gb_prelim_expect)
            if sha256_hex_gb_prelim_expect != sha256_hex_gb_init:
                print('Previous pre-processed file hash', sha256_hex_gb_prelim_expect)
                print('Current pre-processed file hash', sha256_hex_gb_init)
                warn('Pre-processed file has changed since last recorded in processed file', stacklevel=2)
            if sha256_hex_gb_prelim_expect not in prelim_hash_list:
                msg = 'Pre-processed file does not match any previously recorded hash'
                raise ValueError(msg)

            print('Pre-processed file sha256 checksum matches a previously recorded value')

        hf_out.attrs['filename_gb_init_sha256'] = sha256_hex_gb_init
        del sha256_hex_gb_init

        hf_out.attrs['filename_gb_init'] = filename_gb_init
        hf_run.attrs['filename_gb_init'] = filename_gb_init

    hf_out.attrs['filename_config'] = filename_config
    hf_out.attrs['filename_source_gb'] = filename_source_gb

    # store the filenames again to this object

    hf_run.attrs['filename_config'] = filename_config
    hf_run.attrs['filename_source_gb'] = filename_source_gb

    # archive the entire raw text of the configuration file to the hdf5 file as well
    with Path(filename_config).open('rb') as file:
        file_content = file.read()

    hf_run.attrs['config_content'] = file_content

    _ = ifm.store_hdf5(hf_run)

    hf_out.close()

    with Path(filename_out).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256')
        sha256_hex_out = digest.hexdigest()
    print(f'Wrote {filename_out} with sha256 checksum {sha256_hex_out}')
