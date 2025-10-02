"""index for loading the current versions of files"""

import hashlib
from pathlib import Path
from typing import Any
from warnings import warn

import h5py
import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

import GalacticStochastic.global_const as gc
from GalacticStochastic.background_decomposition import load_bgd_from_hdf5
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import LISAConstants
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


def get_preliminary_filename(config: dict[str, Any], snr_thresh: float, Nf: int, Nt: int, dt: float) -> str:
    config_files: dict[str, str] = config['files']
    galaxy_dir = str(config_files['galaxy_dir'])
    preprocessed_prefix = str(config_files.get('preprocessed_prefix', 'preprocessed_background'))
    return (
        galaxy_dir
        + preprocessed_prefix
        + '=%.2f' % snr_thresh
        + '_Nf='
        + str(Nf)
        + '_Nt='
        + str(Nt)
        + '_dt=%.2f.hdf5' % dt
    )


def get_preliminary_filename_alt(config: dict[str, Any], snr_thresh: float, Nf: int, Nt: int, dt: float) -> str:
    config_files: dict[str, str] = config['files']
    galaxy_dir = str(config_files['galaxy_dir'])
    preprocessed_prefix = str(config_files.get('preprocessed_prefix', 'preprocessed_background'))
    return (
        galaxy_dir
        + preprocessed_prefix
        + '_alt'
        + '=%.2f' % snr_thresh
        + '_Nf='
        + str(Nf)
        + '_Nt='
        + str(Nt)
        + '_dt=%.2f.hdf5' % dt
    )


def get_galaxy_filename(config: dict[str, Any]) -> str:
    return str(config['files']['galaxy_dir']) + str(config['files']['galaxy_file'])


def get_processed_gb_filename(config: dict[str, Any], wc: WDMWaveletConstants) -> str:
    config_files: dict[str, str] = config['files']
    galaxy_dir = str(config_files['galaxy_dir'])
    processed_prefix = str(config_files.get('processed_prefix', 'processed_iterations'))
    return (
        galaxy_dir
        + processed_prefix
        + '_Nf='
        + str(wc.Nf)
        + '_Nt='
        + str(wc.Nt)
        + ('_dt=%.2f' % (wc.dt))
        + '.hdf5'
    )


def _source_mask_read_helper(hf_sky: h5py.Group, key: str, fmin: float, fmax: float) -> tuple[int, NDArray[np.floating]]:
    hf_loc = hf_sky[key]

    if not isinstance(hf_loc, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    hf_cat = hf_loc['cat']

    if not isinstance(hf_cat, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    freqs = np.asarray(hf_cat['Frequency'])

    if not np.issubdtype(freqs.dtype, np.floating):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    mask = np.zeros(freqs.shape, dtype=np.bool_)
    mask[:] = (freqs > fmin) & (freqs < fmax)
    n_loc = int(np.sum(1 * mask))
    params = np.zeros((n_loc, n_par_gb), dtype=np.float64)
    for itrl in range(n_par_gb):
        hf_param = hf_cat[labels_gb[itrl]]
        if not isinstance(hf_param, h5py.Dataset):
            msg = 'Unrecognized hdf5 file format'
            raise TypeError(msg)
        params[:, itrl] = hf_param[mask]

    return n_loc, params


def get_full_galactic_params(config: dict[str, Any]):
    """Get the galaxy dataset binaries"""
    # dgb is detached galactic binaries, igb is interacting galactic binaries, vgb is verification
    categories = config['iterative_fit_constants'].get('component_list', ['dgb', 'igb', 'vgb'])
    fmin: float = float(config['iterative_fit_constants'].get('fmin_binary', 1.e-8))
    fmax: float = float(config['iterative_fit_constants'].get('fmax_binary', 1.e0))
    assert fmin <= fmax
    full_galactic_params_filename = get_galaxy_filename(config)
    filename = full_galactic_params_filename

    hf_in = h5py.File(filename, 'r')

    hf_sky = hf_in['sky']

    if not isinstance(hf_sky, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    ns_got = np.zeros(len(categories), dtype=np.int64)
    params_got = []
    for itr, label in enumerate(categories):
        ns_got[itr], params_loc = _source_mask_read_helper(hf_sky, label, fmin, fmax)
        if label == 'dgb' and np.any(params_loc[:, 4] < 0.):
            warn('Some binaries reported as detached have negative frequency derivatives', stacklevel=2)
        params_got.append(params_loc)
        print(label, ns_got[itr])

    n_tot = int(ns_got.sum())

    print('total', n_tot)

    params_gb = np.zeros((n_tot, n_par_gb))

    n_old = 0
    for itr in range(len(categories)):
        n_cur = n_old + int(ns_got[itr])
        params_gb[n_old: n_cur, :] = params_got[itr]
        n_old = n_cur

    hf_in.close()
    assert np.all(np.isfinite(params_gb)), 'Some binaries have non-finite parameters'
    assert not np.any(np.all(params_gb == 0., axis=1)), 'Some binaries have zero for all parameters'
    if np.any(np.all(params_gb == 0., axis=0)):
        warn('Some parameters are always zero', stacklevel=2)
    assert np.all(params_gb[:, 0] > 0.), 'Some binaries have non-positive amplitudes'
    assert np.all((-np.pi / 2 <= params_gb[:, 1]) & (params_gb[:, 1] <= np.pi / 2)), 'Ecliptic latitude not bounded in expected range'
    assert np.all((params_gb[:, 2] >= 0.0) & (params_gb[:, 2] <= 2 * np.pi)), 'Ecliptic longitude not bounded in expected range'
    assert np.all(params_gb[:, 3] > 0.), 'Some binaries have non-positive frequencies'
    if np.any(np.abs(params_gb[:, 4]) * gc.SECSYEAR * 10 > 0.001):
        warn('Some binaries have large frequency derivatives', stacklevel=2)
    print('Largest frequency derivative', np.max(np.abs(params_gb[:, 4]) * gc.SECSYEAR * 10))
    assert np.all((params_gb[:, 5] >= 0.0) & (params_gb[:, 5] <= np.pi)), 'Inclination not bounded in expected range'
    assert np.all((params_gb[:, 6] >= 0.0) & (params_gb[:, 6] <= 2 * np.pi)), 'Initial phase not bounded in expected range'
    assert np.all((params_gb[:, 7] >= 0.0) & (params_gb[:, 7] <= 2 * np.pi)), 'Polarization phase not bounded in expected range'
    return params_gb, ns_got


def load_processed_galactic_file_alt(
    ifm: IterativeFitManager,
    config: dict[str, Any],
    ic: IterationConfig,
    wc: WDMWaveletConstants,
    nt_lim_snr: tuple[int, int] = (0, -1),
    cyclo_mode: int = 1,
):
    snr_thresh = ic.snr_thresh
    filename_in = get_processed_gb_filename(config, wc)
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


def load_preliminary_galactic_file_alt(
    config: dict[str, Any],
    ic: IterationConfig,
    wc: WDMWaveletConstants,
):
    snr_thresh = ic.snr_thresh
    preliminary_gb_filename = get_preliminary_filename_alt(config, snr_thresh, wc.Nf, wc.Nt, wc.dt)

    nt_range: tuple[int, int] = (0, wc.Nt)

    cyclo_mode = 1
    cyclo_key = str(cyclo_mode)

    hf_in = h5py.File(preliminary_gb_filename, 'r')
    hf_itr = hf_in['iteration_results']
    if not isinstance(hf_itr, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)
    hf_snr = hf_itr[str(snr_thresh)]
    if not isinstance(hf_snr, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)
    hf_nt = hf_snr[str(nt_range)]
    if not isinstance(hf_nt, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)
    hf_run = hf_nt[cyclo_key]
    if not isinstance(hf_run, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)
    hf_ifm = hf_run['iterative_manager']
    if not isinstance(hf_ifm, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)
    hf_bis = hf_ifm['inclusion_state']
    if not isinstance(hf_bis, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    hf_snrs_data = hf_bis['snrs_tot_upper']
    if not isinstance(hf_snrs_data, h5py.Dataset):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    snrs_tot_upper_in = np.asarray(hf_snrs_data)[-1]

    hf_noise = hf_ifm['noise_model']
    if not isinstance(hf_noise, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)
    S_inst_m = np.asarray(hf_noise['S_inst_m'])

    hf_bgd = hf_noise['background']
    if not isinstance(hf_bgd, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)
    galactic_below_in = np.asarray(hf_bgd['galactic_below_low'])

    return galactic_below_in, snrs_tot_upper_in, S_inst_m


def load_preliminary_galactic_file(config: dict[str, Any], ic: IterationConfig, wc: WDMWaveletConstants, lc: LISAConstants):
    snr_thresh = ic.snr_thresh
    preliminary_gb_filename = get_preliminary_filename(config, snr_thresh, wc.Nf, wc.Nt, wc.dt)

    hf_in = h5py.File(preliminary_gb_filename, 'r')

    hf_galaxy = hf_in['galaxy']

    if not isinstance(hf_galaxy, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    hf_binary = hf_galaxy['binaries']

    if not isinstance(hf_binary, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    hf_signal = hf_galaxy['signal']

    if not isinstance(hf_signal, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    hf_noise = hf_in['noise_model']

    if not isinstance(hf_noise, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    # check the galaxy filename matches
    source_str_raw = hf_binary['galaxy_file']
    if not isinstance(source_str_raw, h5py.Dataset):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)
    gb_file_source: str = source_str_raw[()].decode()
    full_galactic_params_filename = get_galaxy_filename(config)
    assert gb_file_source == full_galactic_params_filename

    galactic_below_in = np.asarray(hf_signal['galactic_below'])

    snrs_tot_upper_in = np.asarray(hf_binary['snrs_tot_upper'])

    S_inst_m = np.asarray(hf_noise['S_instrument'])

    # check input S makes sense, first value not checked as it may not be consistent
    S_inst_m_alt = instrument_noise_AET_wdm_m(lc, wc)
    assert_allclose(S_inst_m[1:], S_inst_m_alt[1:], atol=1.0e-80, rtol=1.0e-13)

    hf_in.close()

    return galactic_below_in, snrs_tot_upper_in, S_inst_m


def load_processed_gb_file(
    config: dict[str, Any], wc: WDMWaveletConstants):
    # TODO loading should produce a galactic background decomposition object
    filename_in = get_processed_gb_filename(config, wc)
    hf_in = h5py.File(filename_in, 'r')

    hf_S = hf_in['S']

    if not isinstance(hf_S, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    hf_signal = hf_in['signal']

    if not isinstance(hf_signal, h5py.Group):
        msg = 'Unrecognized hdf5 file format'
        raise TypeError(msg)

    bgd = load_bgd_from_hdf5(wc, hf_signal)

    argbinmap = np.asarray(hf_S['argbinmap'])

    hf_in.close()

    return argbinmap, bgd.get_galactic_below_high().reshape((wc.Nt, wc.Nf, bgd.nc_galaxy))


def store_preliminary_gb_file(
    config_filename: str,
    config: dict[str, Any],
    wc: WDMWaveletConstants,
    lc: LISAConstants,
    ic: IterationConfig,
    galactic_below,
    S_inst_m,
    snrs_tot_upper,
) -> None:
    filename_out = get_preliminary_filename(config, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)
    hf_out = h5py.File(filename_out, 'w')

    # store results related to the input galaxy
    galaxy_hf = hf_out.create_group('galaxy')

    signal_hf = galaxy_hf.create_group('signal')
    binary_hf = galaxy_hf.create_group('binaries')

    # the faint part of the galactic signal
    signal_hf.create_dataset('galactic_below', data=galactic_below, compression='gzip')

    # the filename of the file with the galaxy in it
    binary_hf.create_dataset('galaxy_file', data=get_galaxy_filename(config))

    # the initial computed snr
    binary_hf.create_dataset('snrs_tot_upper', data=snrs_tot_upper[0], compression='gzip')

    # store parameters related to the noise model
    noise_hf = hf_out.create_group('noise_model')

    noise_hf.create_dataset('S_instrument', data=S_inst_m)

    # store configuration parameters
    config_hf = hf_out.create_group('configuration')
    config_wc = config_hf.create_group('_wc')
    config_ic = config_hf.create_group('ic')
    config_lc = config_hf.create_group('_lc')
    config_tx = config_hf.create_group('config_text')

    # store all the configuration objects to the file

    # the wavelet constants
    for key in wc._fields:
        config_wc.attrs[key] = getattr(wc, key)

    # lisa related constants
    for key in lc._fields:
        config_lc.attrs[key] = getattr(lc, key)

    # iterative fit related constants
    for key in ic._fields:
        config_ic.attrs[key] = getattr(ic, key)

    # archive the entire raw text of the configuration file to the hdf5 file as well
    with Path(config_filename).open('rb') as file:
        file_content = file.read()

    config_tx.create_dataset(config_filename, data=file_content)

    hf_out.close()


def store_preliminary_gb_file_alt(
    config: dict[str, Any],
    wc: WDMWaveletConstants,
    ifm: IterativeFitManager,
    *,
    write_mode: int = 0,
) -> None:
    ic = ifm.ic
    filename_out = get_preliminary_filename_alt(config, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)

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
    filename_config = config.get('toml_filename', 'not_recorded')

    # save the configuration filenames to the object and raise an error if they are already there but do not match
    if filename_config in hf_out.attrs:
        assert hf_out.attrs['filename_config'] == filename_config
    if filename_source_gb in hf_out.attrs:
        assert hf_out.attrs['filename_source_gb'] == filename_source_gb

    # get the hdf5 group that corresponds to this snr threshold, nt range, and value for cyclo_mode
    # creating sub groups as necessary if they do not already exist

    hf_itr = hf_out.require_group('iteration_results')

    hf_snr = hf_itr.require_group(str(ic.snr_thresh))
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

    hf_run = hf_nt.require_group(cyclo_key)

    # Compute the sha256 checksum of the source galactic binary file and the pre-processed file.
    # If they have been previously recorded in the hdf5 file, check they match.
    # Otherwise, record them.

    with Path(filename_source_gb).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256')
        sha256_hex_gb = digest.hexdigest()
        print('Computed sha256 checksum of source galactic binary file', sha256_hex_gb)

    if 'filename_source_gb_sha256' in hf_out.attrs:
        assert hf_out.attrs[
                   'filename_source_gb_sha256'] == sha256_hex_gb, 'Pre-processed output file was generated from a different source galactic binary file than the one currently specified'
        print(
            'Pre-processed output file source galactic binary sha256 checksum matches current source galactic binary file')

    hf_out.attrs['filename_source_gb_sha256'] = sha256_hex_gb

    del sha256_hex_gb

    # store filenames at top level if they were not already there
    hf_out.attrs['filename_config'] = filename_config
    hf_out.attrs['filename_source_gb'] = filename_source_gb

    # store the filenames again to this object
    hf_run.attrs['filename_config'] = filename_config
    hf_run.attrs['filename_source_gb'] = filename_source_gb

    # archive the entire raw text of the configuration file to the hdf5 file as well
    with Path(filename_config).open('rb') as file:
        file_content = file.read()

    hf_run.create_dataset('config_content', data=file_content)

    ifm.store_hdf5(hf_run)

    hf_out.close()

    with Path(filename_out).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256')
        sha256_hex_out = digest.hexdigest()
    print(f'Wrote {filename_out} with sha256 checksum {sha256_hex_out}')


def store_processed_gb_file(
    config: dict[str, Any],
    wc: WDMWaveletConstants,
    ifm: IterativeFitManager,
    *,
    write_mode: int = 0,
) -> None:
    ic = ifm.ic
    filename_gb_init = get_preliminary_filename(config, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)
    filename_out = get_processed_gb_filename(config, wc)

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
    filename_config = config.get('toml_filename', 'not_recorded')

    # save the configuration filenames to the object and raise an error if they are already there but do not match
    if filename_gb_init in hf_out.attrs:
        assert hf_out.attrs['filename_gb_init'] == filename_gb_init
    if filename_config in hf_out.attrs:
        assert hf_out.attrs['filename_config'] == filename_config
    if filename_source_gb in hf_out.attrs:
        assert hf_out.attrs['filename_source_gb'] == filename_source_gb

    # get the hdf5 group that corresponds to this snr threshold, nt range, and value for cyclo_mode
    # creating sub groups as necessary if they do not already exist

    hf_itr = hf_out.require_group('iteration_results')

    hf_snr = hf_itr.require_group(str(ic.snr_thresh))
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

    hf_run = hf_nt.require_group(cyclo_key)

    # Compute the sha256 checksum of the source galactic binary file and the pre-processed file.
    # If they have been previously recorded in the hdf5 file, check if they match.
    # Otherwise, record them.

    with Path(filename_source_gb).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256')
        sha256_hex_gb = digest.hexdigest()
        print('Computed sha256 checksum of source galactic binary file', sha256_hex_gb)

    if 'filename_source_gb_sha256' in hf_out.attrs:
        assert hf_out.attrs['filename_source_gb_sha256'] == sha256_hex_gb, 'Processed file was generated from a different source galactic binary file than the one currently specified'
        print('Processed file source galactic binary sha256 checksum matches current source galactic binary file')

    hf_out.attrs['filename_source_gb_sha256'] = sha256_hex_gb

    with h5py.File(filename_gb_init, 'r') as hf_prelim:
        try:
            sha256_hex_gb_prelim = hf_prelim.attrs['filename_source_gb_sha256']
            assert sha256_hex_gb_prelim == sha256_hex_gb, 'Pre-processed file was generated from a different source galactic binary file than the one currently specified'
            print('Pre-processed file source galactic binary sha256 checksum matches current source galactic binary file')
            del sha256_hex_gb_prelim
        except KeyError:
            warn('Pre-processed file did not record a sha256 checksum, cannot verify it matches source galactic binary file', stacklevel=2)

    del sha256_hex_gb

    with Path(filename_gb_init).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256')
        sha256_hex_gb_init = digest.hexdigest()
        print('Computed sha256 checksum of pre-processed file', sha256_hex_gb_init)

    if 'filename_gb_init_sha256' in hf_out.attrs:
        assert hf_out.attrs['filename_gb_init_sha256'] == sha256_hex_gb_init, 'Pre-processed file has changed since last recorded in processed file'
        print('Pre-processed file sha256 checksum matches previously recorded value')

    hf_out.attrs['filename_gb_init_sha256'] = sha256_hex_gb_init
    del sha256_hex_gb_init

    hf_out.attrs['filename_gb_init'] = filename_gb_init
    hf_out.attrs['filename_config'] = filename_config
    hf_out.attrs['filename_source_gb'] = filename_source_gb

    # store the filenames again to this object
    hf_run.attrs['filename_gb_init'] = filename_gb_init
    hf_run.attrs['filename_config'] = filename_config
    hf_run.attrs['filename_source_gb'] = filename_source_gb

    # archive the entire raw text of the configuration file to the hdf5 file as well
    with Path(filename_config).open('rb') as file:
        file_content = file.read()

    hf_run.create_dataset('config_content', data=file_content)

    ifm.store_hdf5(hf_run)

    hf_out.close()

    with Path(filename_out).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256')
        sha256_hex_out = digest.hexdigest()
    print(f'Wrote {filename_out} with sha256 checksum {sha256_hex_out}')
