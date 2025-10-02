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


def fetch_or_run_iterative_loop(config: dict[str, Any], cyclo_mode: int, *, nt_range: tuple[int, int] = (0, -1), fetch_mode: int = 0, output_mode: int = 1, wc_in: WDMWaveletConstants | None = None, lc_in: LISAConstants | None = None, ic_in: IterationConfig | None = None, instrument_random_seed_in: int | None = None):
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

    assert fetch_mode in (0, 1)
    nt_lim_snr = PixelGenericRange(nt_min, nt_max, wc.DT, 0.)
    nt_lim_waveform = PixelGenericRange(0, wc.Nt, wc.DT, 0.)

    print(nt_lim_snr.nx_min, nt_lim_snr.nx_max, nt_lim_waveform.nx_min, nt_lim_waveform.nx_max, wc.Nt, wc.Nf, cyclo_mode)

    galactic_below_in, snrs_tot_in, _ = gfi.load_preliminary_galactic_file(
        config, ic, wc,
    )

    params_gb, _ = gfi.get_full_galactic_params(config)

    fit_state = IterativeFitState(ic)

    bgd = BGDecomposition(wc, ic.nc_galaxy, galactic_floor=galactic_below_in.copy(), storage_mode=ic.background_storage_mode)
    del galactic_below_in

    noise_manager = NoiseModelManager(ic, wc, lc, fit_state, bgd, cyclo_mode, nt_lim_snr, instrument_random_seed=instrument_random_seed)

    bis = BinaryInclusionState(wc, ic, lc, params_gb, noise_manager, fit_state, nt_lim_waveform, snrs_tot_in)

    del snrs_tot_in

    ifm = IterativeFitManager(ic, fit_state, noise_manager, bis)

    fetched = False
    if fetch_mode == 1:
        try:
            gfi.load_processed_galactic_file(ifm, config, ic, wc, (nt_lim_snr.nx_min, nt_lim_snr.nx_max), cyclo_mode=cyclo_mode)
            ifm.bis.set_select_params(params_gb)
            fetched = True
        except FileNotFoundError as e:
            # loading did not work because either the file does not exist or the key does not exist in the file
            print(e)
            print('Running loop')
            ifm.do_loop()
    else:
        ifm.do_loop()

    del params_gb

    if fetched:
        pass
    elif output_mode == 1:
        gfi.store_processed_gb_file(
            config,
            wc,
            ifm,
        )
    elif output_mode == 0:
        pass
    else:
        msg = 'Unrecognized option for output_mode'
        raise ValueError(msg)
    return ifm
