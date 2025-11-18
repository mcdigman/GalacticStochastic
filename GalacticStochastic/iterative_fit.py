"""Run or fetch the iterative fit loop for the binaries in the galactic background."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from GalacticStochastic import global_file_index as gfi
from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.config_helper import get_config_objects_from_dict
from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from GalacticStochastic.iteration_config import IterationConfig
    from LisaWaveformTools.lisa_config import LISAConstants
    from WaveletWaveforms.wdm_config import WDMWaveletConstants


def fetch_or_run_iterative_loop(
    config: dict[str, Any],
    cyclo_mode: int | str,
    *,
    nt_range_snr: tuple[int, int] = (0, -1),
    fetch_mode: int | str = 0,
    output_mode: int | str = 1,
    wc_in: WDMWaveletConstants | None = None,
    lc_in: LISAConstants | None = None,
    ic_in: IterationConfig | None = None,
    instrument_random_seed_in: int | None = None,
    preprocess_mode: int | str = 0,
    params_gb_in: NDArray[np.floating] | None = None,
    custom_params: int = -1,
) -> IterativeFitManager:
    """
    Run or fetch the iterative fit loop for galactic binary background analysis.

    This function either runs a new iterative fit loop or fetches results from a previously
    stored file, depending on the specified fetch and preprocess modes. It sets up all
    necessary configuration objects, background decomposition, noise models, and binary
    inclusion states, and manages the logic for pre-processing and file I/O.

    Parameters
    ----------
    config : dict of str to Any
        Configuration dictionary containing file paths and fit settings.
    cyclo_mode : int | str
        Cyclostationary mode key for the fit.
    nt_range_snr : tuple of int
        Time-frequency pixel range as (min, max) indices. Defaults to (0, -1) for full range.
    fetch_mode : int | str
        Mode for fetching or running the fit:
        0 or 'run_reprocess_only': run from scratch, stop if preliminary file missing;
        1 or 'run_or_fetch_reprocess_only': try to fetch, else run, stop if preliminary missing;
        2 or 'fetch_or_fail_reprocess_only': try to fetch, else abort, stop if preliminary missing;
        3 or 'fetch_or_run_all': try to fetch, else run, create preliminary if missing;
        4 or 'run_all': run from scratch, do not check for preliminary file.
        Default is 0.
    output_mode : int | str
        Output mode for storing results:
        0 or 'no_store': do not store;
        1 or 'store_if_new': store results unless they are fetched (default).
        2 or 'store_always': store results even they were fetched
    wc_in : WDMWaveletConstants, optional
        Optional override for wavelet constants.
    lc_in : LISAConstants, optional
        Optional override for LISA instrument constants.
    ic_in : IterationConfig, optional
        Optional override for iteration configuration.
    instrument_random_seed_in : int, optional
        Optional override for instrument random seed.
    preprocess_mode : int | str
        Preprocessing mode:
        0 or 'final': do final processing;
        1 or 'initial': do first step of pre-processing
        2 or 'repeat_initial': re-process an existing pre-processing result
        Default is 0.

    Returns
    -------
    IterativeFitManager
        The manager object containing the results of the iterative fit.

    Raises
    ------
    ValueError
        If an unrecognized fetch_mode, output_mode, or preprocess_mode is provided.
    FileNotFoundError
        If required files or entries are missing and fetch_mode is set to abort.
    NotImplementedError
        If an unsupported preprocess_mode is specified.
    """
    # input validations
    assert output_mode in (0, 1, 2, 'no_store', 'store_if_new', 'store_always'), 'Unrecognized option for output mode'
    assert preprocess_mode in (0, 1, 2, 'final', 'initial', 'repeat_initial'), 'Unrecognized option for processing mode'
    assert fetch_mode in (0, 1, 2, 3, 4, 'run_reprocess_only', 'run_or_fetch_reprocess_only', 'fetch_or_fail_reprocess_only', 'fetch_or_run_all', 'run_all'), 'Unrecognized option for fetch_mode'
    assert cyclo_mode in (0, 1, 'cyclostationary', 'stationary')

    if isinstance(fetch_mode, int):
        fetch_mode_int: int = fetch_mode
    elif isinstance(fetch_mode, str):
        if fetch_mode == 'run_reprocess_only':
            fetch_mode_int = 0
        elif fetch_mode == 'run_or_fetch_reprocess_only':
            fetch_mode_int = 1
        elif fetch_mode == 'fetch_or_fail_reprocess_only':
            fetch_mode_int = 2
        elif fetch_mode == 'fetch_or_run_all':
            fetch_mode_int = 3
        elif fetch_mode == 'run_all':
            fetch_mode_int = 4
        else:
            msg = f'Unrecognized fetch_mode {fetch_mode}'
            raise ValueError(msg)
    else:
        msg = f'Unrecognized type of fetch_mode {fetch_mode}'
        raise TypeError(msg)

    if isinstance(cyclo_mode, int):
        cyclo_mode_int: int = cyclo_mode
    elif isinstance(cyclo_mode, str):
        if cyclo_mode == 'cyclostationary':
            cyclo_mode_int = 0
        elif cyclo_mode == 'stationary':
            cyclo_mode_int = 1
        else:
            msg = f'Unrecognized cyclo_mode {cyclo_mode}'
            raise ValueError(msg)
    else:
        msg = f'Unrecognized type of cyclo_mode {cyclo_mode}'
        raise TypeError(msg)

    if isinstance(output_mode, int):
        output_mode_int: int = output_mode
    elif isinstance(output_mode, str):
        if output_mode == 'no_store':
            output_mode_int = 0
        elif output_mode == 'store_if_new':
            output_mode_int = 1
        elif output_mode == 'store_always':
            output_mode_int = 2
        else:
            msg = f'Unrecognized output_mode {output_mode}'
            raise ValueError(msg)
    else:
        msg = f'Unrecognized type of output_mode {output_mode}'
        raise TypeError(msg)

    if isinstance(preprocess_mode, int):
        preprocess_mode_int: int = preprocess_mode
    elif isinstance(preprocess_mode, str):
        if preprocess_mode == 'final':
            preprocess_mode_int = 0
        elif preprocess_mode == 'initial':
            preprocess_mode_int = 1
        elif preprocess_mode == 'repeat_initial':
            preprocess_mode_int = 2
        else:
            msg = f'Unrecognized preprocess_mode {preprocess_mode}'
            raise ValueError(msg)
    else:
        msg = f'Unrecognized type of preprocess_mode {preprocess_mode}'
        raise TypeError(msg)

    config, wc, lc, ic, instrument_random_seed = get_config_objects_from_dict(config)
    if wc_in is not None:
        wc = wc_in

    if lc_in is not None:
        lc = lc_in

    if ic_in is not None:
        ic = ic_in

    if nt_range_snr == (0, -1):
        nt_min_snr = 0
        nt_max_snr = wc.Nt
    else:
        nt_min_snr = nt_range_snr[0]
        nt_max_snr = nt_range_snr[1]

    assert 0 <= nt_min_snr < nt_max_snr <= wc.Nt

    if instrument_random_seed_in is not None:
        instrument_random_seed = instrument_random_seed_in

    # TODO handle storage with custom params
    if params_gb_in is not None:
        params_gb = params_gb_in
        if custom_params == -1:
            custom_params = 1
    else:
        params_gb, _ = gfi.get_full_galactic_params(config)

    del params_gb_in

    nt_lim_snr = PixelGenericRange(nt_min_snr, nt_max_snr, wc.DT, 0.0)
    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)

    print(
        nt_min_snr,
        nt_max_snr,
        nt_lim_waveform.nx_min,
        nt_lim_waveform.nx_max,
        wc.Nt,
        wc.Nf,
        cyclo_mode,
    )

    if preprocess_mode_int in (1, 2):
        assert cyclo_mode_int == 1, "Pre-processing is only compatible with cyclo_mode = 1 or 'stationary'"

    if preprocess_mode_int in (0, 2):
        # fetch mode options if we are in a state where the preliminary file is needed
        # 0: run the loop from scratch, stop if the preliminary file does not exist
        # 1: try to fetch the final file, else run the loop, stop if the preliminary file does not exist
        # 2: try to fetch the final file, else abort, stop if the preliminary file does not exist
        # 3: try to fetch the final file, else run the loop, if the preliminary file does not exist create it
        # 4: run the loop from scratch, do not use the preliminary file
        if fetch_mode_int in (0, 1, 2):
            fetch_mode_prelim = 2
        elif fetch_mode_int == 3:
            fetch_mode_prelim = 1
        elif fetch_mode_int == 4:
            fetch_mode_prelim = 0
        else:
            msg = f'Unrecognized option for fetch_mode_int {fetch_mode_int}'
            raise ValueError(msg)

        nt_range_prelim = (0, wc.Nt)
        ifm_prelim = fetch_or_run_iterative_loop(
            config,
            cyclo_mode=1,
            nt_range_snr=nt_range_prelim,
            fetch_mode=fetch_mode_prelim,
            preprocess_mode=1,
            output_mode=output_mode_int,
            wc_in=wc_in,
            lc_in=lc,
            ic_in=ic_in,
            instrument_random_seed_in=instrument_random_seed_in,
            params_gb_in=params_gb,
            custom_params=custom_params,
        )

        galactic_below_in = ifm_prelim.noise_manager.bgd.get_galactic_below_low()
        snrs_tot_in = ifm_prelim.bis.get_final_snrs_tot_upper()
        galactic_total_in = ifm_prelim.noise_manager.bgd.get_galactic_total()
        print('INITIAL TOTAL SUM', np.sum(galactic_total_in**2))

        del ifm_prelim

        bgd = BGDecomposition(
            wc,
            ic.nc_galaxy,
            galactic_floor=galactic_below_in.copy(),
            storage_mode=ic.background_storage_mode,
        )
        # set the expected galactic total, for later consistency checks
        # bgd.set_expected_total(galactic_total_in)
        del galactic_below_in
        del galactic_total_in
    elif preprocess_mode_int == 1:
        # fetch mode options if we are in a state where the preliminary file is not needed
        # (0, 4): run the loop from scratch
        # (1, 3): try to fetch the final file, if it does not exist run the loop
        # 2: try to fetch the final file, if it does not exist abort

        bgd = BGDecomposition(
            wc,
            ic.nc_galaxy,
            storage_mode=ic.background_storage_mode,
        )
        snrs_tot_in = None

    else:
        msg = f'Unrecognized option for preprocess_mode_int {preprocess_mode_int}'
        raise NotImplementedError(msg)

    fit_state = IterativeFitState(ic, preprocess_mode=preprocess_mode_int)

    noise_manager = NoiseModelManager(
        ic,
        wc,
        lc,
        fit_state,
        bgd,
        cyclo_mode_int,
        nt_lim_snr,
        instrument_random_seed=instrument_random_seed,
    )

    bis = BinaryInclusionState(
        wc, ic, lc, params_gb, noise_manager, fit_state, nt_lim_waveform, snrs_tot_in=snrs_tot_in
    )

    del snrs_tot_in

    ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

    fetched = False
    if fetch_mode_int in (1, 2, 3):
        try:
            gfi.load_processed_galactic_file(
                ifm,
                config,
                ic,
                wc,
                (nt_lim_snr.nx_min, nt_lim_snr.nx_max),
                cyclo_mode=cyclo_mode_int,
                preprocess_mode=preprocess_mode_int,
            )
            ifm.bis.set_select_params(params_gb)
            fetched = True
        except FileNotFoundError as e:
            # loading did not work because either the file does not exist or the key does not exist in the file
            if fetch_mode_int == 2:
                msg = 'File or entry not found: aborting'
                raise FileNotFoundError(msg) from e
            print(e)
            print('Running loop')
            ifm.do_loop()
    elif fetch_mode_int in (0, 4):
        ifm.do_loop()
    else:
        msg = f'Unrecognized option for fetch_mode_int {fetch_mode}'
        raise ValueError(msg)

    if fetched and output_mode_int != 2:
        pass
    elif output_mode_int in (1, 2):
        if preprocess_mode_int == 0:
            # if pre-processing in mode 2 has happened we will have an inconsistent hash for any re-pre-processed files
            write_mode = 0
        else:
            if preprocess_mode_int == 1:
                write_mode = 2
            else:
                write_mode = 0
        gfi.store_processed_gb_file(
            config,
            wc,
            ifm,
            write_mode=write_mode,
            preprocess_mode=preprocess_mode_int,
            params_gb_in=params_gb,
        )

    elif output_mode_int == 0:
        pass
    else:
        msg = f'Unrecognized option for output_mode_int {output_mode_int}'
        raise ValueError(msg)
    return ifm
