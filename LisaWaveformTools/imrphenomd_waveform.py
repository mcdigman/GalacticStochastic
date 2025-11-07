"""Python implementation of IMRPhenomD behavior by Matthew Digman copyright 2021"""
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import PyIMRPhenomD.IMRPhenomD_const as imrc

# nb.parfors.parfor.sequential_parfor_lowering = True
from numba import njit, prange
from PyIMRPhenomD.IMRPhenomD_internals import ComputeDeltasFromCollocation, ComputeIMRPhenDPhaseConnectionCoefficients, DPhiInsAnsatzInt, DPhiIntAnsatz, DPhiMRD, FinalSpin0815, PhiInsAnsatzInt, PhiIntAnsatz, PhiMRDAnsatzInt, PNPhasingSeriesTaylorF2, alphaFits, amp0Func, betaFits, fmaxCalc, fringdown, gamma_funs, rho_funs, sigmaFits

from LisaWaveformTools.binary_params_manager import BinaryIntrinsicParams
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

if TYPE_CHECKING:
    from numpy.typing import NDArray

# TODO remove implicit dependence on global constants in PyIMRPhenomD.IMRPhenomD_const
# TODO make PixelGenericRange work to ignore values


# from IMRPhenomD_internals import AmpIn
class PrefactorPhasingSeries(NamedTuple):
    minus_five_thirds: float
    minus_four_thirds: float
    minus_one: float
    minus_two_thirds: float
    minus_third: float
    initial_phasing: float
    third: float
    two_thirds: float
    one: float
    four_thirds: float
    five_thirds: float
    two: float
    seven_thirds: float
    eight_thirds: float
    three: float
    zero_with_logv: float
    third_with_logv: float


class IMRPhenomDParams(NamedTuple):
    Mf_ringdown: float
    Mf_damp: float
    final_spin: float
    MfMRDJoinPhi: float
    MfMRDJoinAmp: float
    MfRef: float
    C1Int: float
    C2Int: float
    C1MRD: float
    C2MRD: float
    dPhifRef: float
    TTRef: float
    TTRefIns: float
    TTRefInt: float
    TTRefMRD: float
    phi0: float
    phifRef_base: float
    phifRef: float
    phifRefIns: float
    phifRefInt: float
    phifRefMRD: float
    amp_mult: float
    amp_mult0: float
    amp0: float
    params: BinaryIntrinsicParams


@njit()
def get_imr_phenomd_params(params: BinaryIntrinsicParams, MfRef_in: float, MfRef_max: float, imr_default_t: int, TTRef_in: float, amp_mult: float, phi0: float, t_offset: float) -> IMRPhenomDParams:
    final_spin: float = FinalSpin0815(params.symmetric_mass_ratio, params.chi_s, params.chi_a)
    Mf_ringdown, Mf_damp = fringdown(params.symmetric_mass_ratio, params.chi_s, params.chi_a, final_spin)

    #   Transition frequencies
    #   Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    MfMRDJoinPhi: float = Mf_ringdown / 2.0
    MfMRDJoinAmp: float = fmaxCalc(Mf_ringdown, Mf_damp, params.symmetric_mass_ratio, params.chi_postnewtonian)

    # Compute coefficients to make phase C^1 continuous (phase and first derivative)
    C1Int, C2Int, C1MRD, C2MRD = ComputeIMRPhenDPhaseConnectionCoefficients(Mf_ringdown, Mf_damp, params.symmetric_mass_ratio, params.chi_s, params.chi_a, params.chi_postnewtonian, MfMRDJoinPhi)

    # NOTE: previously MfRef=0 was by default MfRef=fmin, now MfRef defaults to MfmaxCalc (fpeak in the paper)
    # If fpeak is outside of the frequency range, take the last frequency
    if MfRef_in == 0.:
        MfRef: float = min(MfMRDJoinAmp, MfRef_max)
    else:
        MfRef = MfRef_in

    if imr_default_t == 1:
        dPhifRef: float = -DPhiMRD(MfMRDJoinAmp, Mf_ringdown, Mf_damp, params.symmetric_mass_ratio, params.chi_postnewtonian)
    else:
        if MfRef < imrc.PHI_fJoin_INS:
            dPhifRef = -DPhiInsAnsatzInt(MfRef, params.symmetric_mass_ratio, params.chi_s, params.chi_a, params.chi_postnewtonian)
        elif MfRef < MfMRDJoinPhi:
            dPhifRef = -DPhiIntAnsatz(MfRef, params.symmetric_mass_ratio, params.chi_postnewtonian) - C2Int
        else:
            dPhifRef = -DPhiMRD(MfRef, Mf_ringdown, Mf_damp, params.symmetric_mass_ratio, params.chi_postnewtonian) - C2MRD

    dm: float = params.mass_total_detector_sec / (2 * np.pi)

    if ~np.isnan(TTRef_in):
        TTRef: float = TTRef_in
    else:
        TTRef = dPhifRef * dm

    TTRefIns: float = TTRef
    TTRefInt: float = TTRefIns + C2Int * dm
    TTRefMRD: float = TTRefIns + C2MRD * dm

    if MfRef < imrc.PHI_fJoin_INS:
        phifRef_base: float = PhiInsAnsatzInt(MfRef, params.symmetric_mass_ratio, params.chi_s, params.chi_a, params.chi_postnewtonian)
    elif MfRef < MfMRDJoinPhi:
        phifRef_base = PhiIntAnsatz(MfRef, params.symmetric_mass_ratio, params.chi_postnewtonian) + C1Int + C2Int * MfRef
    else:
        phifRef_base = PhiMRDAnsatzInt(MfRef, Mf_ringdown, Mf_damp, params.symmetric_mass_ratio, params.chi_postnewtonian) + C1MRD + C2MRD * MfRef  # MRD range

    # TODO check factors of pi/4 in phifref
    phifRef: float = phifRef_base + 2 * np.pi * TTRef * MfRef / params.mass_total_detector_sec + t_offset / dm * MfRef + 2 * phi0

    # NOTE: opposite Fourier convention with respect to PhenomD - to ensure 22 mode has power for positive f

    phifRefIns: float = phifRef
    phifRefInt: float = phifRef - C1Int
    phifRefMRD: float = phifRef - C1MRD

    amp_mult0: float = amp_mult * amp0Func(params.symmetric_mass_ratio)

    amp0: float = float(amp_mult0 / params.mass_total_detector_sec ** (7 / 6))

    return IMRPhenomDParams(
        Mf_ringdown=Mf_ringdown,
        Mf_damp=Mf_damp,
        final_spin=final_spin,
        MfMRDJoinPhi=MfMRDJoinPhi,
        MfMRDJoinAmp=MfMRDJoinAmp,
        MfRef=MfRef,
        C1Int=C1Int,
        C2Int=C2Int,
        C1MRD=C1MRD,
        C2MRD=C2MRD,
        dPhifRef=dPhifRef,
        TTRef=TTRef,
        TTRefIns=TTRefIns,
        TTRefInt=TTRefInt,
        TTRefMRD=TTRefMRD,
        phi0=phi0,
        phifRef_base=phifRef_base,
        phifRef=phifRef,
        phifRefIns=phifRefIns,
        phifRefInt=phifRefInt,
        phifRefMRD=phifRefMRD,
        amp_mult=amp_mult,
        amp_mult0=amp_mult0,
        amp0=amp0,
        params=params
    )


@njit()
def PhiInsPrefactorsMt(params: BinaryIntrinsicParams) -> PrefactorPhasingSeries:
    v, vlogv = PNPhasingSeriesTaylorF2(params.symmetric_mass_ratio, params.chi_s, params.chi_a)
    # PN phasing series
    cbrt_mass: float = np.cbrt(params.mass_total_detector_sec)
    cbrt_scale: float = cbrt_mass * np.cbrt(np.pi)
    minus_five_thirds: float = v[0] / cbrt_scale ** 5
    minus_one: float = v[2] / cbrt_scale ** 3
    minus_two_thirds: float = v[3] / cbrt_scale ** 2
    minus_third: float = v[4] / cbrt_scale ** 1
    initial_phasing: float = v[5] - np.pi / 4
    third: float = v[6] * cbrt_scale ** 1
    two_thirds: float = v[7] * cbrt_scale ** 2

    zero_with_logv: float = vlogv[5]
    third_with_logv: float = vlogv[6] * cbrt_scale ** 1

    # higher order terms that were calibrated for PhenomD
    # TODO check pi on these terms
    sigmas = sigmaFits(params.symmetric_mass_ratio, params.chi_postnewtonian)
    one: float = sigmas[0] / params.symmetric_mass_ratio * cbrt_mass ** 3
    four_thirds: float = 3 / 4 * sigmas[1] / params.symmetric_mass_ratio * cbrt_mass ** 4
    five_thirds: float = 3 / 5 * sigmas[2] / params.symmetric_mass_ratio * cbrt_mass ** 5
    two: float = 1 / 2 * sigmas[3] / params.symmetric_mass_ratio * cbrt_mass ** 6
    seven_thirds: float = 0.
    eight_thirds: float = 0.
    three: float = 0.

    prefactor_series = PrefactorPhasingSeries(
        minus_five_thirds,
        0.,
        minus_one,
        minus_two_thirds,
        minus_third,
        initial_phasing,
        third,
        two_thirds,
        one,
        four_thirds,
        five_thirds,
        two,
        seven_thirds,
        eight_thirds,
        three,
        zero_with_logv,
        third_with_logv
    )
    return prefactor_series


@njit()
def AmpInsPrefactorsMt(params: BinaryIntrinsicParams) -> PrefactorPhasingSeries:
    rhos = rho_funs(params.symmetric_mass_ratio, params.chi_postnewtonian)
    rho1 = rhos[0]
    rho2 = rhos[1]
    rho3 = rhos[2]

    cbrt_mass: float = np.cbrt(params.mass_total_detector_sec)
    cbrt_scale: float = cbrt_mass * np.cbrt(np.pi)

    two_thirds: float = 1 / 672 * cbrt_scale ** 2 * (-969 + 1804 * params.symmetric_mass_ratio)
    one: float = 1 / 24 * np.pi * params.mass_total_detector_sec * (81 * (params.chi_s + params.chi_a * params.mass_delta) - 44 * params.chi_s * params.symmetric_mass_ratio)
    four_thirds: float = 1 / 8128512 * cbrt_scale ** 4 \
                  * (-27312085 - 41150592 * params.chi_a * params.chi_s * params.mass_delta
                     + 254016 * params.chi_s ** 2 * (-81 + 68 * params.symmetric_mass_ratio) + 254016 * params.chi_a ** 2 * (-81 + 256 * params.symmetric_mass_ratio)
                     + 24 * params.symmetric_mass_ratio * (-1975055 + 1473794 * params.symmetric_mass_ratio))
    five_thirds: float = 1 / 16128 * cbrt_scale ** 5 \
                  * (params.chi_a * params.mass_delta * (285197 - 6316 * params.symmetric_mass_ratio) + params.chi_s * (285197 - 136 * params.symmetric_mass_ratio * (2703 + 262 * params.symmetric_mass_ratio))
                     + 21420 * np.pi * (-1 + 4 * params.symmetric_mass_ratio))
    two: float = 1 / 60085960704 * cbrt_scale ** 6 \
          * (-1242641879927 + 6544617945468 * params.symmetric_mass_ratio
             + 931392 * params.chi_a ** 2 * (1614569 + 4 * params.symmetric_mass_ratio * (-1873643 + 832128 * params.symmetric_mass_ratio))
             + 1862784 * params.chi_a * params.mass_delta * (params.chi_s * (1614569 - 1991532 * params.symmetric_mass_ratio) + 83328 * np.pi)
             + 336 * (2772 * params.chi_s ** 2 * (1614569 + 16 * params.symmetric_mass_ratio * (-184173 + 57451 * params.symmetric_mass_ratio)) - 14902272 * params.chi_s * (-31 + 28 * params.symmetric_mass_ratio) * np.pi
                      + params.symmetric_mass_ratio * (params.symmetric_mass_ratio * (-3248849057 + 965246212 * params.symmetric_mass_ratio) - 763741440 * np.pi ** 2)))
    seven_thirds: float = rho1 * cbrt_mass ** 7
    eight_thirds: float = rho2 * cbrt_mass ** 8
    three: float = rho3 * cbrt_mass ** 9

    prefactor_series = PrefactorPhasingSeries(
        0., 0., 0., 0., 0., 0.,
        0., two_thirds, one, four_thirds, five_thirds,
        two, seven_thirds, eight_thirds, three,
        0., 0.
    )
    return prefactor_series


@njit()
def AmpInsAnsatzInplace(waveform: StationaryWaveformFreq, imr_params: IMRPhenomDParams, nf_lim: PixelGenericRange) -> None:
    """The Newtonian term in LAL is fine and we should use exactly the same (either hardcoded or call).
    We just use the Mathematica expression for convenience.
    Inspiral amplitude plus rho phenom coefficents. rho coefficients computed in rho_funs function.
    Amplitude is a re-expansion. See 1508.07253 and Equation 29, 30 and Appendix B arXiv:1508.07253 for details
    """
    params = imr_params.params
    amp_prefactors = AmpInsPrefactorsMt(params)

    floc = waveform.F[nf_lim.nx_min:nf_lim.nx_max]
    # fv = floc**(1/3)
    fv = floc**(1 / 3)

    waveform.AF[nf_lim.nx_min:nf_lim.nx_max] = 1 / np.sqrt(fv**7) * (
              imr_params.amp0
            + imr_params.amp0 * amp_prefactors.two_thirds * fv**2
            + imr_params.amp0 * amp_prefactors.one * floc
            + imr_params.amp0 * amp_prefactors.four_thirds * fv**4
            + imr_params.amp0 * amp_prefactors.five_thirds * fv**5
            + imr_params.amp0 * amp_prefactors.two * fv**6
            + imr_params.amp0 * amp_prefactors.seven_thirds * fv**7
            + imr_params.amp0 * amp_prefactors.eight_thirds * fv**8
            + imr_params.amp0 * amp_prefactors.three * fv**9
            )


@njit()
def AmpIntAnsatzInplace(waveform: StationaryWaveformFreq, imr_params: IMRPhenomDParams, nf_lim: PixelGenericRange) -> None:
    """Ansatz for the intermediate amplitude. Equation 21 arXiv:1508.07253"""
    params = imr_params.params
    deltas = ComputeDeltasFromCollocation(params.symmetric_mass_ratio, params.chi_s, params.chi_a, params.chi_postnewtonian, imr_params.Mf_ringdown, imr_params.Mf_damp)

    floc = waveform.F[nf_lim.nx_min:nf_lim.nx_max]

    waveform.AF[nf_lim.nx_min:nf_lim.nx_max] = imr_params.amp0 * 1 / floc**(7 / 6) * (deltas[0] + deltas[1] * params.mass_total_detector_sec * floc + deltas[2] * params.mass_total_detector_sec ** 2 * floc ** 2 + deltas[3] * params.mass_total_detector_sec ** 3 * floc ** 3 + deltas[4] * params.mass_total_detector_sec ** 4 * floc ** 4)


@njit()
def AmpMRDAnsatzInplace(waveform: StationaryWaveformFreq, imr_params: IMRPhenomDParams, nf_lim: PixelGenericRange) -> None:
    """Ansatz for the merger-ringdown amplitude. Equation 19 arXiv:1508.07253"""
    params = imr_params.params
    gammas = gamma_funs(params.symmetric_mass_ratio, params.chi_postnewtonian)
    gamma1 = gammas[0]
    gamma2 = gammas[1]
    gamma3 = gammas[2]

    fDMgamma3 = imr_params.Mf_damp * gamma3 / params.mass_total_detector_sec

    floc = waveform.F[nf_lim.nx_min:nf_lim.nx_max]

    fminfRD = floc - imr_params.Mf_ringdown / params.mass_total_detector_sec

    waveform.AF[nf_lim.nx_min:nf_lim.nx_max] = imr_params.amp0 * fDMgamma3 / params.mass_total_detector_sec * gamma1 * 1 / (floc ** (7 / 6) * (fminfRD ** 2 + fDMgamma3 ** 2)) * np.exp(-(gamma2 / fDMgamma3) * fminfRD)


@njit()
def AmpPhaseSeriesInsAnsatz(waveform: StationaryWaveformFreq, imr_params: IMRPhenomDParams, nf_lim: PixelGenericRange) -> None:
    """Ansatz for the inspiral phase. and amplitude
    We call the LAL TF2 coefficients here.
    The exact values of the coefficients used are given
    as comments in the top of this file
    Defined by Equation 27 and 28 arXiv:1508.07253
    """
    params = imr_params.params
    # Assemble PN phasing series
    phi_prefactors = PhiInsPrefactorsMt(params)

    amp_prefactors = AmpInsPrefactorsMt(params)

    dm = 1 / (2 * np.pi)

    for itrf in prange(nf_lim.nx_min, nf_lim.nx_max):
        f: float = waveform.F[itrf]
        fv: float = np.cbrt(f)
        logfv: float = 1 / 3 * np.log(params.mass_total_detector_sec) + 1 / 3 * np.log(f) + 1 / 3 * np.log(np.pi)

        waveform.AF[itrf] = 1 / np.sqrt(fv**7) * (
                  imr_params.amp0
                + imr_params.amp0 * amp_prefactors.two_thirds * fv**2
                + imr_params.amp0 * amp_prefactors.one * f
                + imr_params.amp0 * amp_prefactors.four_thirds * fv**4
                + imr_params.amp0 * amp_prefactors.five_thirds * fv**5
                + imr_params.amp0 * amp_prefactors.two * fv**6
                + imr_params.amp0 * amp_prefactors.seven_thirds * fv**7
                + imr_params.amp0 * amp_prefactors.eight_thirds * fv**8
                + imr_params.amp0 * amp_prefactors.three * fv**9
            )

        waveform.PF[itrf] = 1 / fv ** 5 * (phi_prefactors.minus_five_thirds
            + phi_prefactors.minus_one * fv**2
            + phi_prefactors.minus_two_thirds * f
            + phi_prefactors.minus_third * fv**4
            ) \
                            + phi_prefactors.third * fv \
                            + phi_prefactors.two_thirds * fv ** 2 \
                            + (phi_prefactors.one + imr_params.TTRefIns / dm) * f \
                            + phi_prefactors.four_thirds * fv ** 4 \
                            + phi_prefactors.five_thirds * fv ** 5 \
                            + phi_prefactors.two * f ** 2 \
                            + phi_prefactors.zero_with_logv * logfv \
                            + phi_prefactors.third_with_logv * logfv * fv \
                            + (phi_prefactors.initial_phasing - imr_params.phifRefIns)\

        if imrc.findT:
            waveform.TF[itrf] = 1 / fv ** 8 * (
                - 5 / 3 * dm * phi_prefactors.minus_five_thirds
                - 3 / 3 * dm * phi_prefactors.minus_one * fv**2
                - 2 / 3 * dm * phi_prefactors.minus_two_thirds * f
                - 1 / 3 * dm * phi_prefactors.minus_third * fv**4
                + 1 / 3 * dm * phi_prefactors.zero_with_logv * fv**5
                + 1 / 3 * dm * (phi_prefactors.third_with_logv + phi_prefactors.third) * f**2
                + 2 / 3 * dm * phi_prefactors.two_thirds * fv**7
                ) \
                                + 4 / 3 * dm * phi_prefactors.four_thirds * fv \
                                + 5 / 3 * dm * phi_prefactors.five_thirds * fv ** 2 \
                                + 6 / 3 * dm * phi_prefactors.two * f \
                                + 1 / 3 * dm * phi_prefactors.third_with_logv * 1 / fv ** 2 * logfv \
                                + 3 / 3 * dm * phi_prefactors.one + imr_params.TTRefIns\

            waveform.TFp[itrf] = 1 / fv**11 * (
                + 40 / 9 * dm * phi_prefactors.minus_five_thirds
                + 18 / 9 * dm * phi_prefactors.minus_one * fv**2
                + 10 / 9 * dm * phi_prefactors.minus_two_thirds * f
                + 4 / 9 * dm * phi_prefactors.minus_third * fv**4
                - 3 / 9 * dm * phi_prefactors.zero_with_logv * fv**5
                - 1 / 9 * dm * (phi_prefactors.third_with_logv + 2 * phi_prefactors.third) * f**2
                - 2 / 9 * dm * phi_prefactors.two_thirds * fv**7
                + 4 / 9 * dm * phi_prefactors.four_thirds * f**3
                + 10 / 9 * dm * phi_prefactors.five_thirds * fv**10
                )\
                - 2 / 9 * dm * phi_prefactors.third_with_logv * 1 / fv**5 * logfv \
                + 18 / 9 * dm * phi_prefactors.two\



@njit()
def PhiSeriesInsAnsatz(waveform: StationaryWaveformFreq, imr_params: IMRPhenomDParams, nf_lim: PixelGenericRange) -> None:
    """Ansatz for the inspiral phase.
    We call the LAL TF2 coefficients here.
    The exact values of the coefficients used are given
    as comments in the top of this file
    Defined by Equation 27 and 28 arXiv:1508.07253
    """
    # Assemble PN phasing series
    params = imr_params.params
    phi_prefactors = PhiInsPrefactorsMt(params)
    dm: float = 1 / (2 * np.pi)
    for itrf in prange(nf_lim.nx_min, nf_lim.nx_max):
        floc: float = waveform.F[itrf]

        fv: float = np.cbrt(floc)
        logfv: float = 1 / 3 * np.log(np.pi * params.mass_total_detector_sec) + 1 / 3 * np.log(floc)

        waveform.PF[itrf] = 1 / fv ** 5 * (phi_prefactors.minus_five_thirds
            + phi_prefactors.minus_one * fv**2
            + phi_prefactors.minus_two_thirds * floc
            + phi_prefactors.minus_third * fv**4
            ) \
                            + phi_prefactors.third * fv \
                            + phi_prefactors.two_thirds * fv ** 2 \
                            + (phi_prefactors.one + imr_params.TTRefIns / dm) * floc \
                            + phi_prefactors.four_thirds * fv ** 4 \
                            + phi_prefactors.five_thirds * fv ** 5 \
                            + phi_prefactors.two * fv ** 6 \
                            + phi_prefactors.zero_with_logv * logfv \
                            + phi_prefactors.third_with_logv * logfv * fv \
                            + phi_prefactors.initial_phasing - imr_params.phifRefIns\

        if imrc.findT:
            waveform.TF[itrf] = 1 / fv ** 8 * (
                - 5 / 3 * dm * phi_prefactors.minus_five_thirds
                - 3 / 3 * dm * phi_prefactors.minus_one * fv**2
                - 2 / 3 * dm * phi_prefactors.minus_two_thirds * floc
                - 1 / 3 * dm * phi_prefactors.minus_third * fv**4
                + 1 / 3 * dm * phi_prefactors.zero_with_logv * fv**5
                + 1 / 3 * dm * phi_prefactors.third_with_logv * fv**6 * logfv
                + 1 / 3 * dm * (phi_prefactors.third_with_logv + phi_prefactors.third) * fv**6
                + 2 / 3 * dm * phi_prefactors.two_thirds * fv**7
                ) \
                                + 4 / 3 * dm * phi_prefactors.four_thirds * fv \
                                + 5 / 3 * dm * phi_prefactors.five_thirds * fv ** 2 \
                                + 6 / 3 * dm * phi_prefactors.two * floc \
                                + imr_params.TTRefIns + 3 / 3 * dm * phi_prefactors.one\


            waveform.TFp[itrf] = 1 / fv**11 * (
                + 40 / 9 * dm * phi_prefactors.minus_five_thirds
                + 18 / 9 * dm * phi_prefactors.minus_one * fv**2
                + 10 / 9 * dm * phi_prefactors.minus_two_thirds * floc
                + 4 / 9 * dm * phi_prefactors.minus_third * fv**4
                - 3 / 9 * dm * phi_prefactors.zero_with_logv * fv**5
                - 2 / 9 * dm * phi_prefactors.third_with_logv * fv**6 * logfv
                - 1 / 9 * dm * (phi_prefactors.third_with_logv + 2 * phi_prefactors.third) * fv**6
                - 2 / 9 * dm * phi_prefactors.two_thirds * fv**7
                + 4 / 9 * dm * phi_prefactors.four_thirds * fv**9
                + 10 / 9 * dm * phi_prefactors.five_thirds * fv**10
                ) \
                + 18 / 9 * dm * phi_prefactors.two\



@njit()
def PhiSeriesIntAnsatz(waveform: StationaryWaveformFreq, imr_params: IMRPhenomDParams, nf_lim: PixelGenericRange) -> None:
    """Ansatz for the intermediate phase defined by Equation 16 arXiv:1508.07253"""
    params = imr_params.params
    betas = betaFits(params.symmetric_mass_ratio, params.chi_postnewtonian)
    coeff0 = betas[0] / params.symmetric_mass_ratio * params.mass_total_detector_sec
    coeff1 = betas[1] / params.symmetric_mass_ratio
    coeff2 = -1 / 3 * betas[2] / params.symmetric_mass_ratio / params.mass_total_detector_sec**3

    dm = 1 / (2 * np.pi)

    if imrc.findT:
        floc = waveform.F[nf_lim.nx_min:nf_lim.nx_max]

        waveform.PF[nf_lim.nx_min:nf_lim.nx_max] = coeff1 * np.log(params.mass_total_detector_sec) - imr_params.phifRefInt \
                                 + (2 * np.pi * imr_params.TTRefInt + coeff0) * floc \
                                 + coeff2 / floc ** 3 \
                                 + coeff1 * np.log(floc)

        waveform.TF[nf_lim.nx_min:nf_lim.nx_max] = imr_params.TTRefInt + dm * coeff0 \
                            + 1 / floc**4 * (- 3 * dm * coeff2 + dm * coeff1 * floc**3)

        waveform.TFp[nf_lim.nx_min:nf_lim.nx_max] = 1 / floc**5 * (+12 * dm * coeff2 - dm * coeff1 * floc**3)
    else:
        floc = waveform.F[nf_lim.nx_min:nf_lim.nx_max]

        waveform.PF[nf_lim.nx_min:nf_lim.nx_max] = coeff1 * np.log(params.mass_total_detector_sec) - imr_params.phifRefInt \
                                 + (2 * np.pi * imr_params.TTRefInt + coeff0) * floc \
                                 + coeff2 / floc ** 3 \
                                 + coeff1 * np.log(floc)


@njit()
def PhiSeriesMRDAnsatz(waveform: StationaryWaveformFreq, imr_params: IMRPhenomDParams, nf_lim: PixelGenericRange) -> None:
    """Ansatz for the merger-ringdown phase Equation 14 arXiv:1508.07253"""
    params = imr_params.params
    alphas = alphaFits(params.symmetric_mass_ratio, params.chi_postnewtonian)
    coeff0 = alphas[0] / params.symmetric_mass_ratio * params.mass_total_detector_sec
    coeff1 = -alphas[1] / params.symmetric_mass_ratio / params.mass_total_detector_sec
    coeff2 = 4 / 3 * alphas[2] / params.symmetric_mass_ratio * float(params.mass_total_detector_sec**(3 / 4))
    coeff3 = alphas[3] / params.symmetric_mass_ratio

    dm = 1 / (2 * np.pi)
    fDM = imr_params.Mf_damp / params.mass_total_detector_sec
    fRD = imr_params.Mf_ringdown / params.mass_total_detector_sec

    # numba cannot fuse loops across the conditional so need to write everything that needs to be fused twice
    if imrc.findT:
        floc: NDArray[np.floating] = waveform.F[nf_lim.nx_min:nf_lim.nx_max]

        fq: NDArray[np.floating] = np.sqrt(np.sqrt(floc**3)) * floc
        fadj: NDArray[np.floating] = (floc - alphas[4] * fRD) / fDM
        waveform.PF[nf_lim.nx_min:nf_lim.nx_max] = -imr_params.phifRefMRD \
                                 + (2 * np.pi * imr_params.TTRefMRD + coeff0) * floc \
                                 + 1 / floc * (coeff1 + coeff2 * fq) \
                                 + coeff3 * np.arctan(fadj)

        waveform.TF[nf_lim.nx_min:nf_lim.nx_max] = imr_params.TTRefMRD + dm * coeff0 \
                                 + 1 / floc ** 2 * (- dm * coeff1 + 3 / 4 * dm * coeff2 * fq) \
                                 + dm * coeff3 / fDM * (1 / (1 + fadj**2))
        waveform.TFp[nf_lim.nx_min:nf_lim.nx_max] = + 1 / floc**3 * (2 * dm * coeff1 - 3 / 16 * dm * coeff2 * fq) \
                            - 2 * dm * coeff3 / fDM**2 * fadj * (1 / (1 + fadj**2))**2
    else:
        floc = waveform.F[nf_lim.nx_min:nf_lim.nx_max]

        fq = np.sqrt(np.sqrt(floc**3)) * floc
        fadj = (floc - alphas[4] * fRD) / fDM
        waveform.PF[nf_lim.nx_min:nf_lim.nx_max] = -imr_params.phifRefMRD \
                                 + (2 * np.pi * imr_params.TTRefMRD + coeff0) * floc \
                                 + 1 / floc * (coeff1 + coeff2 * fq) \
                                 + coeff3 * np.arctan(fadj)


# Phase: glueing function ################
@njit()
def IMRPhenDPhaseFI(waveform: StationaryWaveformFreq, params: BinaryIntrinsicParams, nf_lim: PixelGenericRange, MfRef_in: float, phi0: float, MfRef_max: float = np.inf, imr_default_t: int = 0, t_offset: float = 0., TTRef_in: float = np.nan, amp_mult: float = 1.) -> tuple[int, IMRPhenomDParams]:
    """This function computes the IMR phase given phenom coefficients.
    Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    The inspiral, intermediate and merger-ringdown phase parts
    split the calculation to just 1 of 3 possible mutually exclusive ranges
    Mfs must be sorted
    modified to anchor frequencies to FI at t=0
    """
    if waveform.F[-1] > imrc.f_CUT / params.mass_total_detector_sec:
        itrFCut: int = int(np.searchsorted(waveform.F, imrc.f_CUT / params.mass_total_detector_sec, side='right'))
    else:
        itrFCut = nf_lim.nx_max

    MfRef_max = min(MfRef_max, float(params.mass_total_detector_sec * waveform.F[itrFCut - 1]))
    imr_params: IMRPhenomDParams = get_imr_phenomd_params(params=params, MfRef_in=MfRef_in, MfRef_max=MfRef_max, imr_default_t=imr_default_t, TTRef_in=TTRef_in, phi0=phi0, t_offset=t_offset, amp_mult=amp_mult)

    if waveform.F[itrFCut - 1] < imrc.PHI_fJoin_INS / params.mass_total_detector_sec:
        itrfMRDPhi: int = itrFCut
        itrfIntPhi: int = itrFCut
    elif waveform.F[itrFCut - 1] < imr_params.MfMRDJoinPhi / params.mass_total_detector_sec:
        itrfMRDPhi = itrFCut
        itrfIntPhi = int(np.searchsorted(waveform.F, imrc.PHI_fJoin_INS / params.mass_total_detector_sec))
    else:
        itrfMRDPhi = int(np.searchsorted(waveform.F, imr_params.MfMRDJoinPhi / params.mass_total_detector_sec))
        itrfIntPhi = int(np.searchsorted(waveform.F, imrc.PHI_fJoin_INS / params.mass_total_detector_sec))

    if itrfIntPhi > 0:
        PhiSeriesInsAnsatz(waveform, imr_params, PixelGenericRange(0, itrfIntPhi, 1., 0.))  # Ins range
    if itrfIntPhi < itrfMRDPhi:
        PhiSeriesIntAnsatz(waveform, imr_params, PixelGenericRange(itrfIntPhi, itrfMRDPhi, 1., 0.))  # intermediate range
    if itrfMRDPhi < itrFCut:
        PhiSeriesMRDAnsatz(waveform, imr_params, PixelGenericRange(itrfMRDPhi, itrFCut, 1., 0.))  # MRD range

    waveform.PF[itrFCut:] = 0.
    waveform.TF[itrFCut:] = 0.
    waveform.TFp[itrFCut:] = 0.

    return itrFCut, imr_params


@njit()
def IMRPhenDAmplitudeFI(waveform: StationaryWaveformFreq, params: BinaryIntrinsicParams, nf_lim: PixelGenericRange, amp_mult: float = 1., MfRef_in: float = 0., MfRef_max: float = np.inf, imr_default_t: int = 0, phi0: float = 0., TTRef_in: float = np.nan, t_offset: float = 0.) -> IMRPhenomDParams:
    """This function computes the IMR amplitude given phenom coefficients.
    Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    The inspiral, intermediate and merger-ringdown amplitude parts
    """
    imr_params = get_imr_phenomd_params(params=params, MfRef_in=MfRef_in, MfRef_max=MfRef_max, imr_default_t=imr_default_t, amp_mult=amp_mult, phi0=phi0, TTRef_in=TTRef_in, t_offset=t_offset)

    if waveform.F[-1] > imrc.f_CUT / params.mass_total_detector_sec:
        itrFCut: int = int(np.searchsorted(waveform.F, imrc.f_CUT / params.mass_total_detector_sec, side='right'))
    else:
        itrFCut = nf_lim.nx_max

    if waveform.F[itrFCut - 1] < imrc.AMP_fJoin_INS / params.mass_total_detector_sec:
        itrfMRDAmp: int = itrFCut
        itrfIntAmp: int = itrFCut
    elif waveform.F[itrFCut - 1] < imr_params.MfMRDJoinAmp / params.mass_total_detector_sec:
        itrfMRDAmp = itrFCut
        itrfIntAmp = int(np.searchsorted(waveform.F, imrc.AMP_fJoin_INS / params.mass_total_detector_sec))
    else:
        itrfMRDAmp = int(np.searchsorted(waveform.F, imr_params.MfMRDJoinAmp / params.mass_total_detector_sec))
        itrfIntAmp = int(np.searchsorted(waveform.F, imrc.AMP_fJoin_INS / params.mass_total_detector_sec))

    # split the calculation to just 1 of 3 possible mutually exclusive ranges
    if itrfIntAmp > 0:
        AmpInsAnsatzInplace(waveform, imr_params, PixelGenericRange(0, itrfIntAmp, 1., 0.))  # Inspiral range
    if itrfIntAmp < itrfMRDAmp:
        AmpIntAnsatzInplace(waveform, imr_params, PixelGenericRange(itrfIntAmp, itrfMRDAmp, 1., 0.))  # Intermediate range
    if itrfMRDAmp < itrFCut:
        AmpMRDAnsatzInplace(waveform, imr_params, PixelGenericRange(itrfMRDAmp, itrFCut, 1., 0.))  # MRD range

    waveform.AF[itrFCut:] = 0.

    return imr_params


@njit()
def IMRPhenDAmpPhaseFI_get_TTRef(params: BinaryIntrinsicParams, MfRef_in: float, imr_default_t: int = 0, t_offset: float = 0., MfRef_max: float = np.inf, TTRef_in: float = np.nan, phi0: float = 0., amp_mult: float = 1.) -> float:
    """Get only TTRef given input FI at MfRef_in if imr_default_t is true, use the phasing convention from IMRPhenomD,
    otherwise try to set MfRef_in=Mf at t=0
    """
    imr_params = get_imr_phenomd_params(params=params, MfRef_in=MfRef_in, MfRef_max=MfRef_max, imr_default_t=imr_default_t, TTRef_in=TTRef_in, phi0=phi0, amp_mult=amp_mult, t_offset=t_offset)

    return imr_params.TTRef


@njit()
def IMRPhenDAmpPhase_tc(waveform: StationaryWaveformFreq, params: BinaryIntrinsicParams, nf_lim: PixelGenericRange, TTRef_in: float, phi0: float, amp_mult: float, MfRef_in: float = 0., MfRef_max: float = np.inf, imr_default_t: int = 0, t_offset: float = 0.) -> tuple[int, IMRPhenomDParams]:
    """Get both amplitude and phase in place at the same time given input TTRef_in"""
    # TODO reabsorb this now redundant function
    imr_params: IMRPhenomDParams = get_imr_phenomd_params(params=params, MfRef_in=MfRef_in, MfRef_max=MfRef_max, imr_default_t=imr_default_t, TTRef_in=TTRef_in, phi0=phi0, amp_mult=amp_mult, t_offset=t_offset)

    MfLast: float = float(waveform.F[nf_lim.nx_max - 1] * params.mass_total_detector_sec)
    if MfLast > imrc.f_CUT:
        itrFCut: int = int(np.searchsorted(waveform.F, imrc.f_CUT / params.mass_total_detector_sec, side='right'))
        MfLast = float(waveform.F[itrFCut - 1] * params.mass_total_detector_sec)
    else:
        itrFCut = nf_lim.nx_max

    # TODO duplicate logic
    if MfLast < imrc.AMP_fJoin_INS:
        itrfMRDAmp: int = itrFCut
        itrfIntAmp: int = itrFCut
    elif MfLast < imr_params.MfMRDJoinAmp:
        itrfMRDAmp = itrFCut
        itrfIntAmp = int(np.searchsorted(waveform.F, imrc.AMP_fJoin_INS / params.mass_total_detector_sec))
    else:
        itrfIntAmp = int(np.searchsorted(waveform.F, imrc.AMP_fJoin_INS / params.mass_total_detector_sec))
        itrfMRDAmp = int(np.searchsorted(waveform.F, imr_params.MfMRDJoinAmp / params.mass_total_detector_sec))

    if MfLast < imrc.PHI_fJoin_INS:
        itrfMRDPhi: int = itrFCut
        itrfIntPhi: int = itrFCut
    elif MfLast < imr_params.MfMRDJoinPhi:
        itrfMRDPhi = itrFCut
        itrfIntPhi = int(np.searchsorted(waveform.F, imrc.PHI_fJoin_INS / params.mass_total_detector_sec))
    else:
        itrfMRDPhi = int(np.searchsorted(waveform.F, imr_params.MfMRDJoinPhi / params.mass_total_detector_sec))
        itrfIntPhi = int(np.searchsorted(waveform.F, imrc.PHI_fJoin_INS / params.mass_total_detector_sec))

    itrfIntMax: int = max(itrfIntPhi, itrfIntAmp)

    # Technically, this wastes a small amount of operations filling values that will be overwritten by the intermediate.
    # In practice the combined method is so much faster that it justifies the wasted computation
    # and it would unnecessarily increase code complexity to avoid it.
    if itrfIntMax > 0:
        AmpPhaseSeriesInsAnsatz(waveform, imr_params, PixelGenericRange(0, itrfIntMax, 1., 0.))  # Ins range

    #   split the calculation to just 1 of 3 possible mutually exclusive ranges
    if itrfIntAmp < itrfMRDAmp:
        AmpIntAnsatzInplace(waveform, imr_params, PixelGenericRange(itrfIntAmp, itrfMRDAmp, 1., 0.))  # Intermediate range
    if itrfMRDAmp < itrFCut:
        AmpMRDAnsatzInplace(waveform, imr_params, PixelGenericRange(itrfMRDAmp, itrFCut, 1., 0.))  # MRD range

    if itrfIntPhi < itrfMRDPhi:
        PhiSeriesIntAnsatz(waveform, imr_params, PixelGenericRange(itrfIntPhi, itrfMRDPhi, 1., 0.))  # intermediate range
    if itrfMRDPhi < itrFCut:
        PhiSeriesMRDAnsatz(waveform, imr_params, PixelGenericRange(itrfMRDPhi, itrFCut, 1., 0.))  # MRD range

    if itrFCut < nf_lim.nx_max:
        waveform.AF[itrFCut:] = 0.
        waveform.PF[itrFCut:] = 0.
        waveform.TF[itrFCut:] = 0.
        waveform.TFp[itrFCut:] = 0.

    return itrFCut, imr_params


@njit()
def IMRPhenDAmpPhaseFI(waveform: StationaryWaveformFreq, params: BinaryIntrinsicParams, nf_lim: PixelGenericRange, MfRef_in: float, phi0: float, amp_mult: float, imr_default_t: int = 0, t_offset: float = 0., MfRef_max: float = np.inf, TTRef_in: float = np.nan) -> tuple[int, IMRPhenomDParams]:
    """Get both amplitude and phase in place at the same time given input FI at MfRef_in if imr_default_t is true, use the phasing convention from IMRPhenomD,
    otherwise try to set MfRef_in=Mf at t=0
    """
    return IMRPhenDAmpPhase_tc(waveform=waveform, params=params, nf_lim=nf_lim, TTRef_in=TTRef_in, phi0=phi0, amp_mult=amp_mult, t_offset=t_offset, MfRef_max=MfRef_max, imr_default_t=imr_default_t, MfRef_in=MfRef_in)
