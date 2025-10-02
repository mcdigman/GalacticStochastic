from __future__ import annotations

from typing import TYPE_CHECKING, Any

from GalacticStochastic import global_file_index as gfi
from GalacticStochastic.background_decomposition import BGDecomposition
from GalacticStochastic.config_helper import get_config_objects_from_dict
from GalacticStochastic.inclusion_state_manager import BinaryInclusionState
from GalacticStochastic.iterative_fit_manager import IterativeFitManager
from GalacticStochastic.iterative_fit_state_machine import IterativeFitState
from GalacticStochastic.noise_manager import NoiseModelManager
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

if TYPE_CHECKING:
    from GalacticStochastic.iteration_config import IterationConfig
    from LisaWaveformTools.lisa_config import LISAConstants
    from WaveletWaveforms.wdm_config import WDMWaveletConstants


def fetch_or_run_iterative_loop(
    config: dict[str, Any],
    cyclo_mode: int,
    *,
    nt_range: tuple[int, int] = (0, -1),
    fetch_mode: int = 0,
    output_mode: int = 1,
    wc_in: WDMWaveletConstants | None = None,
    lc_in: LISAConstants | None = None,
    ic_in: IterationConfig | None = None,
    instrument_random_seed_in: int | None = None,
    preprocess_mode: int = 0,
) -> IterativeFitManager:
    config, wc, lc, ic, instrument_random_seed = get_config_objects_from_dict(config)
    if wc_in is not None:
        wc = wc_in
    if lc_in is not None:
        lc = lc_in
    if ic_in is not None:
        ic = ic_in
    if instrument_random_seed_in is not None:
        instrument_random_seed = instrument_random_seed_in

    if nt_range == (0, -1):
        nt_min = 0
        nt_max = wc.Nt
    else:
        nt_min = nt_range[0]
        nt_max = nt_range[1]

    assert 0 <= nt_min < nt_max <= wc.Nt

    assert fetch_mode in (0, 1, 2, 3, 4), 'Unrecognized option for fetch_mode'

    nt_lim_snr = PixelGenericRange(nt_min, nt_max, wc.DT, 0.0)
    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)

    print(
        nt_lim_snr.nx_min, nt_lim_snr.nx_max, nt_lim_waveform.nx_min, nt_lim_waveform.nx_max, wc.Nt, wc.Nf, cyclo_mode,
    )

    if preprocess_mode in (1, 2):
        assert cyclo_mode == 1, 'Pre-processing is only compatible with cyclo_mode = 1'

    if preprocess_mode in (0, 2):

        # galactic_below_in, snrs_tot_in = gfi.load_preliminary_galactic_file(
        #    config,
        #    ic,
        #    wc,
        # )
        # fetch mode options if we are in a state where the preliminary file is needed
        # 0: run the loop from scratch, stop if the preliminary file does not exist
        # 1: try to fetch the final file, if it does not exist run the loop, stop if the preliminary file does not exist
        # 2: try to fetch the final file, if it does not exist abort, stop if the preliminary file does not exist
        # 3: try to fetch the final file, if it does not exist run the loop, if the preliminary file does not exist run the loop to create it
        # 4: run the loop from scratch, do not use the preliminary file
        if fetch_mode in (0, 1, 2):
            fetch_mode_prelim = 2
        elif fetch_mode == 3:
            fetch_mode_prelim = 1
        elif fetch_mode == 4:
            fetch_mode_prelim = 0
        else:
            msg = 'Unrecognized option for fetch_mode'
            raise ValueError(msg)

        nt_range_prelim = (0, wc.Nt)
        ifm_prelim = fetch_or_run_iterative_loop(config, cyclo_mode=1, nt_range=nt_range_prelim, fetch_mode=fetch_mode_prelim,
                                                 preprocess_mode=1, output_mode=output_mode, wc_in=wc_in, lc_in=lc,
                                                 ic_in=ic_in, instrument_random_seed_in=instrument_random_seed_in)

        # assert_allclose(ifm_prelim.noise_manager.bgd.get_galactic_below_low(), galactic_below_in)
        # assert_allclose(ifm_prelim.bis._snrs_tot_upper[-1], snrs_tot_in)
        galactic_below_in = ifm_prelim.noise_manager.bgd.get_galactic_below_low()
        snrs_tot_in = ifm_prelim.bis.get_final_snrs_tot_upper()

        bgd = BGDecomposition(
            wc, ic.nc_galaxy, galactic_floor=galactic_below_in.copy(), storage_mode=ic.background_storage_mode,
        )
        del galactic_below_in
    elif preprocess_mode == 1:
        # fetch mode options if we are in a state where the preliminary file is not needed
        # (0, 4): run the loop from scratch
        # (1, 3): try to fetch the final file, if it does not exist run the loop
        # 2: try to fetch the final file, if it does not exist abort

        bgd = BGDecomposition(
            wc, ic.nc_galaxy, storage_mode=ic.background_storage_mode,
        )
        snrs_tot_in = None
    else:
        msg = 'Unrecognized option for preprocess_mode'
        raise NotImplementedError(msg)

    params_gb, _ = gfi.get_full_galactic_params(config)

    fit_state = IterativeFitState(ic, preprocess_mode=preprocess_mode)

    noise_manager = NoiseModelManager(
        ic, wc, lc, fit_state, bgd, cyclo_mode, nt_lim_snr, instrument_random_seed=instrument_random_seed,
    )

    bis = BinaryInclusionState(wc, ic, lc, params_gb, noise_manager, fit_state, nt_lim_waveform, snrs_tot_in=snrs_tot_in)

    del snrs_tot_in

    ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

    # TODO preprocessed calls might need different snrs cutoffs based on threshold entered
    fetched = False
    if fetch_mode in (1, 2, 3):
        try:
            gfi.load_processed_galactic_file(
                ifm, config, ic, wc, (nt_lim_snr.nx_min, nt_lim_snr.nx_max), cyclo_mode=cyclo_mode, preprocess_mode=preprocess_mode,
            )
            ifm.bis.set_select_params(params_gb)
            fetched = True
        except FileNotFoundError as e:
            # loading did not work because either the file does not exist or the key does not exist in the file
            if fetch_mode == 2:
                msg = 'File or entry not found: aborting'
                raise FileNotFoundError(msg) from e
            print(e)
            print('Running loop')
            ifm.do_loop()
    elif fetch_mode in (0, 4):
        ifm.do_loop()
    else:
        msg = 'Unrecognized option for fetch_mode'
        raise ValueError(msg)

    del params_gb

    if fetched:
        pass
    elif output_mode == 1:
        if preprocess_mode == 0:
            # TODO if fetch_mode==4, we will have an inconsistent hash in the pre-processed file
            # if pre-processing in mode 2 has happened we will have an inconsistent hash for any re-pre-processed files
            gfi.store_processed_gb_file(
                config,
                wc,
                ifm,
            )
        else:
            if preprocess_mode == 1:
                write_mode = 2
            else:
                write_mode = 0

            gfi.store_preliminary_gb_file(
                config,
                wc,
                ifm,
                write_mode=write_mode,
            )

    elif output_mode == 0:
        pass
    else:
        msg = 'Unrecognized option for output_mode'
        raise ValueError(msg)
    return ifm
