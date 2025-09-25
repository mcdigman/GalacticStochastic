"""index for loading the current versions of files"""

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from GalacticStochastic.background_decomposition import load_bgd_from_hdf5
from GalacticStochastic.iteration_config import IterationConfig
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from LisaWaveformTools.instrument_noise import instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
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


def get_common_noise_filename(config: dict, snr_thresh, wc: WDMWaveletConstants) -> str:
    galaxy_dir = config['files']['galaxy_dir']
    return (
        galaxy_dir
        + ('preprocessed_background=%.2f' % snr_thresh)
        + '_Nf='
        + str(wc.Nf)
        + '_Nt='
        + str(wc.Nt)
        + '_dt=%.2f.hdf5' % wc.dt
    )


def get_preliminary_filename(config: dict, snr_thresh: float, Nf: int, Nt: int, dt: float) -> str:
    galaxy_dir = config['files']['galaxy_dir']
    return (
        galaxy_dir
        + ('preprocessed_background=%.2f' % snr_thresh)
        + '_Nf='
        + str(Nf)
        + '_Nt='
        + str(Nt)
        + '_dt=%.2f.hdf5' % dt
    )


def get_galaxy_filename(config: dict) -> str:
    return config['files']['galaxy_dir'] + config['files']['galaxy_file']


def get_processed_gb_filename(config: dict, stat_only, snr_thresh, wc: WDMWaveletConstants, nt_lim_snr: PixelGenericRange) -> str:
    galaxy_dir = config['files']['galaxy_dir']
    return (
        galaxy_dir
        + ('gb8_processed_snr=%.2f' % snr_thresh)
        + '_Nf='
        + str(wc.Nf)
        + '_Nt='
        + str(wc.Nt)
        + ('_dt=%.2f' % (wc.dt))
        + '_const='
        + str(stat_only)
        + '_nt_min='
        + str(nt_lim_snr.nx_min)
        + '_nt_max='
        + str(nt_lim_snr.nx_max)
        + '.hdf5'
    )


def get_noise_common(config: dict, snr_thresh, wc: WDMWaveletConstants, lc):
    try:
        filename_gb_common = get_common_noise_filename(config, snr_thresh, wc)
        hf_in = h5py.File(filename_gb_common, 'r')

        hf_S = hf_in['S']

        if not isinstance(hf_S, h5py.Group):
            msg = 'Unrecognized hdf5 file format'
            raise TypeError(msg)

        noise_realization_common = np.asarray(hf_S['noise_realization'])

        hf_in.close()
        return noise_realization_common
    except KeyError:
        return np.random.normal(0., 1., (wc.Nt, wc.Nf, lc.nc_waveform))


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


def get_full_galactic_params(
    config: dict):
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
        params_got.append(params_loc)
        print(label, ns_got[itr])

    n_tot = int(ns_got.sum())

    print('total', n_tot)

    params_gb = np.zeros((n_tot, n_par_gb))

    n_old = 0
    for itr in range(len(categories)):
        n_cur = n_old + int(ns_got[itr])
        params_gb[n_old: n_cur, :] = params_got[itr]

    hf_in.close()
    return params_gb, ns_got


def load_preliminary_galactic_file(config: dict, ic: IterationConfig, wc: WDMWaveletConstants, lc: LISAConstants):
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
    gb_file_source = source_str_raw[()].decode()
    full_galactic_params_filename = get_galaxy_filename(config)
    assert gb_file_source == full_galactic_params_filename

    galactic_below_in = np.asarray(hf_signal['galactic_below'])

    snrs_tot_upper_in = np.asarray(hf_binary['snrs_tot_upper'])

    S_inst_m = np.asarray(hf_noise['S_instrument'])

    # check input S makes sense, first value not checked as it may not be consistent
    S_inst_m_alt = instrument_noise_AET_wdm_m(lc, wc)
    assert np.allclose(S_inst_m[1:], S_inst_m_alt[1:], atol=1.0e-80, rtol=1.0e-13)

    hf_in.close()

    return galactic_below_in, snrs_tot_upper_in, S_inst_m


def load_processed_gb_file(
    config: dict, snr_thresh, wc: WDMWaveletConstants, nt_lim_snr: PixelGenericRange, *, stat_only,
):
    # TODO loading should produce a galactic background decomposition object
    filename_in = get_processed_gb_filename(config, stat_only, snr_thresh, wc, nt_lim_snr)
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
    config: dict,
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


def store_processed_gb_file(
    config: dict,
    wc: WDMWaveletConstants,
    lc: LISAConstants,
    ic: IterationConfig,
    ifm: IterativeFitManager,
) -> None:
    noise_manager = ifm.noise_manager
    bgd = noise_manager.bgd
    n_full_converged = ifm.n_full_converged
    bis = ifm.bis
    nt_lim_snr = noise_manager.nt_lim_snr
    S_inst_m = noise_manager.S_inst_m
    S_final = noise_manager.S_final
    stat_only = noise_manager.stat_only

    filename_gb_init = get_preliminary_filename(config, ic.snr_thresh, wc.Nf, wc.Nt, wc.dt)
    filename_gb_common = get_common_noise_filename(config, ic.snr_thresh, wc)
    filename_out = get_processed_gb_filename(config, stat_only, ic.snr_thresh, wc, nt_lim_snr)

    period_list = ic.period_list

    n_bin_use = bis.n_bin_use
    snrs_tot_upper = bis.snrs_tot_upper
    argbinmap = bis.argbinmap
    faints_old = bis.faints_old
    faints_cur = bis.faints_cur
    brights = bis.brights

    hf_out = h5py.File(filename_out, 'w')
    hf_S = hf_out.create_group('S')
    hf_signal = hf_out.create_group('signal')
    hf_signal.create_dataset('galactic_below', data=bgd.get_galactic_below_low(), compression='gzip')
    hf_signal.create_dataset('galactic_above', data=bgd.get_galactic_coadd_resolvable(), compression='gzip')
    hf_signal.create_dataset('galactic_undecided', data=bgd.get_galactic_coadd_undecided(), compression='gzip')

    hf_S.create_dataset('period_list', data=period_list)
    hf_S.attrs['n_bin_use'] = n_bin_use
    hf_S.create_dataset('S_stat_m', data=S_inst_m)
    hf_S.create_dataset('snrs_tot_upper', data=snrs_tot_upper[n_full_converged], compression='gzip')
    hf_S.create_dataset('argbinmap', data=argbinmap, compression='gzip')

    hf_S.create_dataset('faints_old', data=faints_old, compression='gzip')
    hf_S.create_dataset('faints_cur', data=faints_cur[n_full_converged], compression='gzip')

    hf_S.create_dataset('brights', data=brights[n_full_converged], compression='gzip')
    hf_S.create_dataset('S_final', data=S_final, compression='gzip')

    hf_S.create_dataset('source_gb_file', data=get_galaxy_filename(config))
    hf_S.create_dataset('preliminary_gb_file', data=filename_gb_init)  # TODO these are redundant as constructed
    hf_S.create_dataset('init_gb_file', data=filename_gb_init)
    hf_S.create_dataset('common_gb_noise_file', data=filename_gb_common)

    hf_wc = hf_out.create_group('_wc')
    for key in wc._fields:
        hf_wc.attrs[key] = getattr(wc, key)

    hf_lc = hf_out.create_group('_lc')
    for key in lc._fields:
        hf_lc.attrs[key] = getattr(lc, key)

    hf_ic = hf_out.create_group('ic')
    for key in ic._fields:
        hf_ic.attrs[key] = getattr(ic, key)

    hf_out.close()
