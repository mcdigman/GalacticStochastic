"""functions to compute rigid adiabatic response in frequency domain"""
from typing import NamedTuple

import numpy as np
from numba import njit

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

GAMMA = 0.57721566490153286060651209008240243104215933593992  # Euler-Mascheroni constant

# TODO everything here should use PNPhasingSeries for compactness
# TODO make this the only place the code gets the waveforms from
# TODO check timing models
# TODO allow efficient selection of model without calling inplace methods


class TaylorF2BasicParams(NamedTuple):
    Mt: float
    Mc: float
    FI: float
    logD: float
    phic: float


class TaylorF2AlignedSpinParams(NamedTuple):
    Mt: float
    Mc: float
    FI: float
    logD: float
    phic: float
    chis: float
    chia: float
    tc: float


class TaylorF2EccParams(NamedTuple):
    Mt: float
    Mc: float
    FI: float
    logD: float
    phic: float
    e0: float


@njit()
def TaylorF2_inplace(intrinsic_waveform: StationaryWaveformFreq, params_intrinsic: TaylorF2BasicParams, nf_lim: PixelGenericRange, t_offset: float = 0., tc_mode: int = 0) -> float:
    """This is the TaylorF2 model to 2PN order. DOI: 10.1103/PhysRevD.80.084043"""
    # TODO enforce keeping frequency below lso
    FS = intrinsic_waveform.F
    PSI = intrinsic_waveform.PF
    TS = intrinsic_waveform.TF
    TPS = intrinsic_waveform.TFp
    AS = intrinsic_waveform.AF

    Mt = float(params_intrinsic.Mt)
    Mc = float(params_intrinsic.Mc)
    FI = float(params_intrinsic.FI)
    phic = float(params_intrinsic.phic)
    logD = float(params_intrinsic.logD)

    eta: float = float((Mc / Mt)**(5 / 3))
    flso: float = 1 / (6**(3 / 2) * np.pi * Mt)
    nulso: float = (np.pi * Mt * flso)**(1 / 3)
    # see cutler 1998 for fisher analytics

    c0 = 3 / (128 * eta)
    c1 = 20 / 9 * (743 / 336 + 11 / 4 * eta)
    c2 = -16 * np.pi
    c3 = 10 * (3058673 / 1016064 + 5429 / 1008 * eta + 617 / 144 * eta**2)
    c4 = np.pi * (38645 / 756 - 65 / 9 * eta)
    c5 = 11583231236531 / 4694215680 - 640 / 3 * np.pi**2 - 6848 / 21 * GAMMA + (-15737765635 / 3048192 + 2255 * np.pi**2 / 12) * eta + 76055 / 1728 * eta**2 - 127825 / 1296 * eta**3
    c6 = np.pi * (77096675 / 254016 + 378515 / 1512 * eta - 74045 / 756 * eta**2)

    p0 = 5 * c0 * Mt / 6
    p1 = 3 / 5 * c1
    p2 = 2 / 5 * c2
    p3 = 1 / 5 * c3
    p4 = -3 / 5 * c4
    p5 = 6848 - 1 / 5 * c5
    p6 = -2 / 5 * c6

    f0 = 20 / 9 * np.pi * Mt**2 * c0
    f1 = 9 / 20 * c1
    f2 = c2 / 4
    f3 = c3 / 10
    f4 = -9 * c4 / 40
    f5 = 856 / 105 - c5 / 20
    f6 = -c6 / 20

    # TODO logs can be optimized using identities
    nuI: float = (np.pi * Mt * FI)**(1 / 3)
    # TODO using TTRef defined locally seems more self consistent with definitions but causes greater deviation from TaylorT3
    if tc_mode:
        TTRef: float = t_offset
    else:
        TTRef = p0 / nuI**8 * (1 + p1 * nuI**2 + p2 * nuI**3 + p3 * nuI**4 + p4 * nuI**5 + (6848 / 105 * np.log(4 * nuI) - 712192 / 105 + p5) * nuI**6 + p6 * nuI**7) + t_offset
    PSII: float = 2 * np.pi * TTRef * FI - np.pi / 4 + c0 / nuI**5 * (1 + c1 * nuI**2 + c2 * nuI**3 + c3 * nuI**4 + c4 * (1 + 3 * np.log(nuI / nulso)) * nuI**5 + (c5 - 6848 / 21 * np.log(4 * nuI)) * nuI**6 + c6 * nuI**7)
    PSI_ref: float = -2 * phic - PSII - np.pi / 4

    Amp: float = np.sqrt(5 / 6) / np.pi**(2 / 3) * Mc**(5 / 6) / np.exp(logD)
    for n in range(nf_lim.nx_min, nf_lim.nx_max):
        nu: float = (np.pi * Mt * FS[n])**(1 / 3)
        m4: float = 1 + 3 * np.log(nu / nulso)
        fp5: float = 1712 / 105 * np.log(4 * nu)
        ac5: float = -6848 / 21 * np.log(4 * nu)
        ap5: float = 6848 / 105 * np.log(4 * nu) - 712192 / 105
        PSI[n] = 2 * np.pi * TTRef * FS[n] + PSI_ref - np.pi / 4 + c0 / nu**5 * (1 + c1 * nu**2 + c2 * nu**3 + c3 * nu**4 + c4 * m4 * nu**5 + (c5 + ac5) * nu**6 + c6 * nu**7)
        TS[n] = TTRef - p0 / nu**8 * (1 + p1 * nu**2 + p2 * nu**3 + p3 * nu**4 + p4 * nu**5 + (ap5 + p5) * nu**6 + p6 * nu**7)
        TPS[n] = f0 / nu**11 * (1 + f1 * nu**2 + f2 * nu**3 + f3 * nu**4 + f4 * nu**5 + (f5 + fp5) * nu**6 + f6 * nu**7)
        # TODO carry out amp to appropriate order
        AS[n] = Amp * (FS[n]**(-7 / 6) + (9 / 40 * np.pi**(2 / 3)) * (Mt**(2 / 3) * c1) * FS[n]**(-1 / 2) + np.pi / 8 * (Mt * c2) * FS[n]**(-1 / 6))
    return float(TTRef)


@njit()
def TaylorF2_eccentric_inplace(intrinsic_waveform: StationaryWaveformFreq, params_intrinsic: TaylorF2EccParams, nf_lim: PixelGenericRange, amplitude_pn: int = 0, t_offset: float = 0., tc_mode: int = 0) -> float:
    """This is the TaylorF2 model with eccentricity but no spin from DOI: 10.1103/PhysRevD.93.124061"""
    Mc = float(params_intrinsic.Mc)
    Mt = float(params_intrinsic.Mt)
    FI = float(params_intrinsic.FI)
    phic = float(params_intrinsic.phic)
    logD = float(params_intrinsic.logD)
    eta = float((Mc / Mt)**(5 / 3))
    # note that PSII and PSI_ref are slightly different from the no eccentricity case, but it seems to just be a convention that cancels in the results
    # see cutler 1998 for fisher analytics
    # see arXiv : 2001.11412 v1.pdf
    # TODO write tests for self consistency and consistency with other waveform models
    FS = intrinsic_waveform.F
    PSI = intrinsic_waveform.PF
    TS = intrinsic_waveform.TF
    TPS = intrinsic_waveform.TFp
    AS = intrinsic_waveform.AF

    # c, p, and f coefficients are same as without eccentricity, although factored differently in arXiv : 2001.11412 v1.pdf
    c0 = 3 / (128 * eta)
    c1 = 20 / 9 * (743 / 336 + 11 / 4 * eta)
    c2 = -16 * np.pi
    c3 = 10 * (3058673 / 1016064 + 5429 / 1008 * eta + 617 / 144 * eta**2)
    c4 = np.pi * (38645 / 756 - 65 / 9 * eta)
    c5 = 11583231236531 / 4694215680 - 640 / 3 * np.pi**2 - 6848 / 21 * GAMMA + (-15737765635 / 3048192 + 2255 * np.pi**2 / 12) * eta + 76055 / 1728 * eta**2 - 127825 / 1296 * eta**3
    c6 = np.pi * (77096675 / 254016 + 378515 / 1512 * eta - 74045 / 756 * eta**2)

    p0 = 5 * c0 * Mt / 6
    p1 = 3 / 5 * c1
    p2 = 2 / 5 * c2
    p3 = 1 / 5 * c3
    p4 = -3 / 5 * c4
    p5 = 6848 - 1 / 5 * c5
    p6 = -2 / 5 * c6

    f0 = 20 / 9 * np.pi * Mt**2 * c0
    f1 = 9 / 20 * c1
    f2 = c2 / 4
    f3 = c3 / 10
    f4 = -9 * c4 / 40
    f5 = 856 / 105 - c5 / 20
    f6 = -c6 / 20

    nuI: float = (np.pi * Mt * FI)**(1 / 3)

    y0 = -(2355 / 1462) * params_intrinsic.e0**2
    y01 = (299076223 / 81976608) + (18766963 / 2927736) * eta
    y02 = -(2819123 / 282600) * np.pi
    y03 = (16237683263 / 3330429696) \
            + (24133060753 / 971375328) * eta \
            + (1562608261 / 69383952) * eta**2
    y04 = -((2831492681 / 118395270) + (11552066831 / 270617760) * eta) * np.pi
    y05: float = -(43603153867072577087 / 132658535116800000) \
            + (536803271 / 19782000) * GAMMA \
            + (15722503703 / 325555200) * np.pi**2 \
            + ((299172861614477 / 689135247360) - (15075413 / 1446912) * np.pi**2) * eta \
            + (3455209264991 / 41019955200) * eta**2 \
            + (50612671711 / 878999040) * eta**3 \
            + (3843505163 / 59346000) * np.log(2) \
            - (1121397129 / 17584000) * np.log(3)
    y10 = (2833 / 1008) - (197 / 36) * eta
    y20 = (377 / 72) * np.pi
    y30 = -(1193251 / 3048192) - (66317 / 9072) * eta + (18155 / 1296) * eta**2
    y40 = ((764881 / 90720) - (949457 / 22680) * eta) * np.pi
    y50: float = (26531900578691 / 168991764480) \
        - (3317 / 126) * GAMMA \
        + (122833 / 10368) * np.pi**2 \
        + ((9155185261 / 548674560) - (3977 / 1152) * np.pi**2) * eta \
        - (5732473 / 1306368) * eta**2 \
        - (3090307 / 139968) * eta**3 \
        + (87419 / 1890) * np.log(2) \
        - (26001 / 560) * np.log(3)
    y11 = (847282939759 / 82632420864) \
            - (718901219 / 368894736) * eta \
            - (3697091711 / 105398496) * eta**2
    y12 = (-(7986575459 / 284860800) + (555367231 / 10173600) * eta) * np.pi
    y21 = ((112751736071 / 5902315776) + (7075145051 / 210796992) * eta) * np.pi
    y13 = (46001356684079 / 3357073133568) \
            + (253471410141755 / 5874877983744) * eta \
            - (1693852244423 / 23313007872) * eta**2 \
            - (307833827417 / 2497822272) * eta**3
    y31 = -(356873002170973 / 249880440692736) \
           - (260399751935005 / 8924301453312) * eta \
           + (150484695827 / 35413894656) * eta**2 \
           + (340714213265 / 3794345856) * eta**3
    y22 = -(1062809371 / 20347200) * np.pi**2
    ay50: float = -(3317 * 2 / 252) * np.log(4 * nuI)

    dy0 = 34 / 15 * y0

    dy10 = y10
    dy20 = y20
    dy30 = y30
    dy40 = y40
    dy50 = y50
    day50 = ay50
    dy01 = 28 / 34 * y01
    dy11 = 28 / 34 * y11
    dy21 = 28 / 34 * y21
    dy31 = 28 / 34 * y31
    dy02 = 25 / 34 * y02
    dy12 = 25 / 34 * y12
    dy22 = 25 / 34 * y22
    dy03 = 22 / 34 * y03
    dy13 = 22 / 34 * y13
    dy04 = 19 / 34 * y04
    dy05 = 16 / 34 * y05
    ddy0 = 43 / 24 * dy0

    ddy01 = 37 / 43 * dy01
    ddy02 = 34 / 43 * dy02
    ddy03 = 31 / 43 * dy03
    ddy04 = 28 / 43 * dy04
    ddy05 = 25 / 43 * dy05
    ddy11 = 37 / 43 * dy11
    ddy21 = 37 / 43 * dy21
    ddy31 = 37 / 43 * dy31
    ddy12 = 34 / 43 * dy12
    ddy22 = 34 / 43 * dy22
    ddy13 = 31 / 43 * dy13
    ddy10 = dy10
    ddy20 = dy20
    ddy30 = dy30
    ddy40 = dy40
    ddy50 = dy50
    dday50 = day50

    # starting values of coefficients that change in the loop
    m4I: float = 1 + 3 * np.log(nuI)
    ac5I: float = -(3424 / 21) * 2 * np.log(4 * nuI)
    ap5I: float = (3424 / 105) * 2 * np.log(4 * nuI) - 712192 / 105
    ay05I: float = (536803271 / 39564000) * 2 * np.log(4 * nuI)
    day05I = 16 / 34 * ay05I - 31576663 / 13188000

    if tc_mode:
        TTRef: float = t_offset
    else:
        TTRef = p0 / nuI**8 * \
                (1 + dy0
                     + (p1 + dy0 * (dy01 + dy10)) * nuI**2
                     + (p2 + dy0 * (dy02 + dy20)) * nuI**3
                     + (p3 + dy0 * (dy03 + dy30 + dy11)) * nuI**4
                     + (p4 + dy0 * (dy04 + dy40 + dy12 + dy21)) * nuI**5
                     + (p5 + ap5I + dy0 * (dy05 + day05I + dy13 + dy31 + dy22 + dy50 + day50)) * nuI**6
                     + p6 * nuI**7
                 ) + t_offset

    PSII: float = 2 * np.pi * TTRef * FI - np.pi / 4 + c0 / nuI**5 * \
        (1 + y0
            + (c1 + y0 * (y01 + y10)) * nuI**2
            + (c2 + y0 * (y02 + y20)) * nuI**3
            + (c3 + y0 * (y03 + y11 + y30)) * nuI**4
            + (c4 * m4I + y0 * (y04 + y12 + y21 + y40)) * nuI**5
            + (c5 + ac5I + y0 * (y05 + ay05I + y13 + y22 + y31 + y50 + ay50)) * nuI**6
            + c6 * nuI**7
        )
    PSI_ref: float = -2 * phic - PSII - np.pi / 4

    Amp: float = np.sqrt(5 / 6) / np.pi**(2 / 3) * Mc**(5 / 6) / np.exp(logD)
    for n in range(nf_lim.nx_min, nf_lim.nx_max):
        nu: float = (np.pi * Mt * FS[n])**(1 / 3)

        # get correct values for the coefficients that change
        m4: float = 1 + 3 * np.log(nu)
        ac5: float = -(3424 / 21) * 2 * np.log(4 * nu)
        ap5: float = (3424 / 105) * 2 * np.log(4 * nu) - 712192 / 105
        af5: float = (856 / 105) * 2 * np.log(4 * nu)
        ay05: float = (536803271 / 39564000) * 2 * np.log(4 * nu)
        day05 = 16 / 34 * ay05 - 31576663 / 13188000
        dday05 = 25 / 43 * day05 - 734341 / 824250

        PSI[n] = 2 * np.pi * TTRef * FS[n] + PSI_ref - np.pi / 4 + c0 / nu**5 * \
            (1
              + c1 * nu**2
              + c2 * nu**3
              + c3 * nu**4
              + m4 * c4 * nu**5
              + (c5 + ac5) * nu**6
              + c6 * nu**7
              + y0 * (nuI / nu)**(19 / 3) *
                (1
                    + y01 * nu**2
                    + y10 * nuI**2
                    + y02 * nu**3
                    + y20 * nuI**3
                    + y03 * nu**4
                    + y11 * nu**2 * nuI**2
                    + y30 * nuI**4
                    + y04 * nu**5
                    + y12 * nu**3 * nuI**2
                    + y21 * nu**2 * nuI**3
                    + y40 * nuI**5
                    + (y05 + ay05) * nu**6
                    + y13 * nu**4 * nuI**2
                    + y22 * nu**3 * nuI**3
                    + y31 * nu**2 * nuI**4
                    + (y50 + ay50) * nuI**6
                )
            )

        TS[n] = TTRef - p0 / nu**8 *\
                (1
                + p1 * nu**2
                + p2 * nu**3
                + p3 * nu**4
                + p4 * nu**5
                + (p5 + ap5) * nu**6
                + p6 * nu**7
                + dy0 * (nuI / nu)**(19 / 3) *
                    (1
                       + dy01 * nu**2
                       + dy10 * nuI**2
                       + dy02 * nu**3
                       + dy20 * nuI**3 + dy03 * nu**4
                       + dy11 * nu**2 * nuI**2
                       + dy30 * nuI**4
                       + dy04 * nu**5
                       + dy12 * nu**3 * nuI**2
                       + dy21 * nu**2 * nuI**3
                       + dy40 * nuI**5
                       + (dy05 + day05) * nu**6
                       + dy13 * nu**4 * nuI**2
                       + dy22 * nu**3 * nuI**3
                       + dy31 * nu**2 * nuI**4
                       + (dy50 + day50) * nuI**6
                    )
                )
        TPS[n] = f0 / nu**11 *\
                   (1
                    + f1 * nu**2
                    + f2 * nu**3
                    + f3 * nu**4
                    + f4 * nu**5
                    + (f5 + af5) * nu**6
                    + f6 * nu**7
                    + ddy0 * (nuI / nu)**(19 / 3) *
                        (1
                         + ddy01 * nu**2
                         + ddy10 * nuI**2
                         + ddy02 * nu**3
                         + ddy20 * nuI**3
                         + ddy03 * nu**4
                         + ddy11 * nu**2 * nuI**2
                         + ddy30 * nuI**4
                         + ddy04 * nu**5
                         + ddy12 * nu**3 * nuI**2
                         + ddy21 * nu**2 * nuI**3
                         + ddy40 * nuI**5
                         + (ddy05 + dday05) * nu**6
                         + ddy13 * nu**4 * nuI**2
                         + ddy22 * nu**3 * nuI**3
                         + ddy31 * nu**2 * nuI**4
                         + (ddy50 + dday50) * nuI**6
                        )
                   )

        # TODO carry out amp to appropriate order
        if amplitude_pn:
            AS[n] = Amp * (FS[n]**(-7 / 6) + (9 / 40 * np.pi**(2 / 3)) * (Mt**(2 / 3) * c1) * FS[n]**(-1 / 2) + np.pi / 8 * (Mt * c2) * FS[n]**(-1 / 6))
        else:
            # in the paper, amplitude is carried out to 0pn order
            AS[n] = Amp * FS[n]**(-7 / 6)
    return float(TTRef)


@njit()
def TaylorF2_eccentricity_solve(params_intrinsic: TaylorF2EccParams, FI2: float) -> float:
    """Get an inferred eccentricity at a later frequency based on the taylorf2 eccentricity model"""
    e01 = params_intrinsic.e0
    Mt = float(params_intrinsic.Mt)
    Mc = float(params_intrinsic.Mc)
    FI = float(params_intrinsic.FI)
    eta = float((Mc / Mt)**(5 / 3))
    # note that PSII and PSI_ref are slightly different from the no eccentricity case, but it seems to just be a convention that cancels in the results
    # see cutler 1998 for fisher analytics
    # see arXiv : 2001.11412 v1.pdf
    # TODO write tests for self consistency and consistency with other waveform models

    nuI1: float = (np.pi * Mt * FI)**(1 / 3)
    nuI2: float = (np.pi * Mt * FI2)**(1 / 3)

    y0 = -(2355 / 1462) * e01**2
    y01 = (299076223 / 81976608) + (18766963 / 2927736) * eta
    y02 = -(2819123 / 282600) * np.pi
    y03 = (16237683263 / 3330429696) \
            + (24133060753 / 971375328) * eta \
            + (1562608261 / 69383952) * eta**2
    y04 = -((2831492681 / 118395270) + (11552066831 / 270617760) * eta) * np.pi
    y05: float = -(43603153867072577087 / 132658535116800000) \
            + (536803271 / 19782000) * GAMMA \
            + (15722503703 / 325555200) * np.pi**2 \
            + ((299172861614477 / 689135247360) - (15075413 / 1446912) * np.pi**2) * eta \
            + (3455209264991 / 41019955200) * eta**2 \
            + (50612671711 / 878999040) * eta**3 \
            + (3843505163 / 59346000) * np.log(2) \
            - (1121397129 / 17584000) * np.log(3)
    y10 = (2833 / 1008) - (197 / 36) * eta
    y20 = (377 / 72) * np.pi
    y30 = -(1193251 / 3048192) - (66317 / 9072) * eta + (18155 / 1296) * eta**2
    y40 = ((764881 / 90720) - (949457 / 22680) * eta) * np.pi
    y50: float = (26531900578691 / 168991764480) \
        - (3317 / 126) * GAMMA \
        + (122833 / 10368) * np.pi**2 \
        + ((9155185261 / 548674560) - (3977 / 1152) * np.pi**2) * eta \
        - (5732473 / 1306368) * eta**2 \
        - (3090307 / 139968) * eta**3 \
        + (87419 / 1890) * np.log(2) \
        - (26001 / 560) * np.log(3)
    y11 = (847282939759 / 82632420864) \
            - (718901219 / 368894736) * eta \
            - (3697091711 / 105398496) * eta**2
    y12 = (-(7986575459 / 284860800) + (555367231 / 10173600) * eta) * np.pi
    y21 = ((112751736071 / 5902315776) + (7075145051 / 210796992) * eta) * np.pi
    y13 = (46001356684079 / 3357073133568) \
            + (253471410141755 / 5874877983744) * eta \
            - (1693852244423 / 23313007872) * eta**2 \
            - (307833827417 / 2497822272) * eta**3
    y31 = -(356873002170973 / 249880440692736) \
           - (260399751935005 / 8924301453312) * eta \
           + (150484695827 / 35413894656) * eta**2 \
           + (340714213265 / 3794345856) * eta**3
    y22 = -(1062809371 / 20347200) * np.pi**2

    dy0 = 34 / 15 * y0

    dy10 = y10
    dy20 = y20
    dy30 = y30
    dy40 = y40
    dy50 = y50
    dy01 = 28 / 34 * y01
    dy11 = 28 / 34 * y11
    dy21 = 28 / 34 * y21
    dy31 = 28 / 34 * y31
    dy02 = 25 / 34 * y02
    dy12 = 25 / 34 * y12
    dy22 = 25 / 34 * y22
    dy03 = 22 / 34 * y03
    dy13 = 22 / 34 * y13
    dy04 = 19 / 34 * y04
    dy05 = 16 / 34 * y05
    ddy0 = 43 / 24 * dy0

    ddy01 = 37 / 43 * dy01
    ddy02 = 34 / 43 * dy02
    ddy03 = 31 / 43 * dy03
    ddy04 = 28 / 43 * dy04
    ddy05 = 25 / 43 * dy05
    ddy11 = 37 / 43 * dy11
    ddy21 = 37 / 43 * dy21
    ddy31 = 37 / 43 * dy31
    ddy12 = 34 / 43 * dy12
    ddy22 = 34 / 43 * dy22
    ddy13 = 31 / 43 * dy13
    ddy10 = dy10
    ddy20 = dy20
    ddy30 = dy30
    ddy40 = dy40
    ddy50 = dy50

    ddy0_derived: float = ddy0 * (nuI1 / nuI2)**(19 / 3) *\
          (
            1
            + ddy10 * nuI1**2
            + ddy20 * nuI1**3
            + ddy30 * nuI1**4
            + ddy40 * nuI1**5
            + ddy50 * nuI1**6
            - (3317 * nuI1**6 * np.log(4 * nuI1)) / 126
            + ddy01 * nuI2**2
            + ddy02 * nuI2**3
            + ddy03 * nuI2**4
            + ddy04 * nuI2**5
            - (30107981 * nuI2**6) / 13188000 + ddy05 * nuI2**6 + (734341 * nuI2**6 * np.log(4 * nuI2)) / 98910
            + ddy11 * nuI1**2 * nuI2**2
            + ddy21 * nuI1**3 * nuI2**2
            + ddy31 * nuI1**4 * nuI2**2
            + ddy12 * nuI1**2 * nuI2**3
            + ddy22 * nuI1**3 * nuI2**3
            + ddy13 * nuI1**2 * nuI2**4
          )\
        /\
          (
            1
            + ddy01 * nuI2**2
            + ddy10 * nuI2**2
            + ddy02 * nuI2**3
            + ddy20 * nuI2**3
            + ddy03 * nuI2**4
            + ddy11 * nuI2**4
            + ddy30 * nuI2**4
            + ddy04 * nuI2**5
            + ddy12 * nuI2**5
            + ddy21 * nuI2**5
            + ddy40 * nuI2**5
            - (30107981 * nuI2**6) / 13188000 + ddy05 * nuI2**6 + ddy13 * nuI2**6 + ddy22 * nuI2**6 + ddy31 * nuI2**6 + ddy50 * nuI2**6 - (44512 * nuI2**6 * np.log(4 * nuI2)) / 2355
          )

    # return params_intrinsic.e02_derived
    return float(np.sqrt(-24 / 157 * ddy0_derived))


@njit()
def TaylorF2_eccentric_TTRef(params_intrinsic: TaylorF2EccParams) -> float:
    """Get just TTRef for input parameters DOI: 10.1103/PhysRevD.93.124061"""
    Mt = float(params_intrinsic.Mt)
    Mc = float(params_intrinsic.Mc)
    FI = float(params_intrinsic.FI)
    eta = float((Mc / Mt)**(5 / 3))
    # note that PSII and PSI_ref are slightly different from the no eccentricity case, but it seems to just be a convention that cancels in the results
    # see cutler 1998 for fisher analytics
    # see arXiv : 2001.11412 v1.pdf
    # TODO write tests for self consistency and consistency with other waveform models

    # c, p, and f coefficients are same as without eccentricity, although factored differently in arXiv : 2001.11412 v1.pdf
    c0 = 3 / (128 * eta)
    c1 = 20 / 9 * (743 / 336 + 11 / 4 * eta)
    c2 = -16 * np.pi
    c3 = 10 * (3058673 / 1016064 + 5429 / 1008 * eta + 617 / 144 * eta**2)
    c4 = np.pi * (38645 / 756 - 65 / 9 * eta)
    c5 = 11583231236531 / 4694215680 - 640 / 3 * np.pi**2 - 6848 / 21 * GAMMA + (-15737765635 / 3048192 + 2255 * np.pi**2 / 12) * eta + 76055 / 1728 * eta**2 - 127825 / 1296 * eta**3
    c6 = np.pi * (77096675 / 254016 + 378515 / 1512 * eta - 74045 / 756 * eta**2)

    p0 = 5 * c0 * Mt / 6
    p1 = 3 / 5 * c1
    p2 = 2 / 5 * c2
    p3 = 1 / 5 * c3
    p4 = -3 / 5 * c4
    p5 = 6848 - 1 / 5 * c5
    p6 = -2 / 5 * c6

    nuI: float = (np.pi * Mt * FI)**(1 / 3)

    y0 = -(2355 / 1462) * params_intrinsic.e0**2
    y01 = (299076223 / 81976608) + (18766963 / 2927736) * eta
    y02 = -(2819123 / 282600) * np.pi
    y03 = (16237683263 / 3330429696) \
            + (24133060753 / 971375328) * eta \
            + (1562608261 / 69383952) * eta**2
    y04 = -((2831492681 / 118395270) + (11552066831 / 270617760) * eta) * np.pi
    y05: float = -(43603153867072577087 / 132658535116800000) \
            + (536803271 / 19782000) * GAMMA \
            + (15722503703 / 325555200) * np.pi**2 \
            + ((299172861614477 / 689135247360) - (15075413 / 1446912) * np.pi**2) * eta \
            + (3455209264991 / 41019955200) * eta**2 \
            + (50612671711 / 878999040) * eta**3 \
            + (3843505163 / 59346000) * np.log(2) \
            - (1121397129 / 17584000) * np.log(3)
    y10 = (2833 / 1008) - (197 / 36) * eta
    y20 = (377 / 72) * np.pi
    y30 = -(1193251 / 3048192) - (66317 / 9072) * eta + (18155 / 1296) * eta**2
    y40 = ((764881 / 90720) - (949457 / 22680) * eta) * np.pi
    y50: float = (26531900578691 / 168991764480) \
        - (3317 / 126) * GAMMA \
        + (122833 / 10368) * np.pi**2 \
        + ((9155185261 / 548674560) - (3977 / 1152) * np.pi**2) * eta \
        - (5732473 / 1306368) * eta**2 \
        - (3090307 / 139968) * eta**3 \
        + (87419 / 1890) * np.log(2) \
        - (26001 / 560) * np.log(3)
    y11 = (847282939759 / 82632420864) \
            - (718901219 / 368894736) * eta \
            - (3697091711 / 105398496) * eta**2
    y12 = (-(7986575459 / 284860800) + (555367231 / 10173600) * eta) * np.pi
    y21 = ((112751736071 / 5902315776) + (7075145051 / 210796992) * eta) * np.pi
    y13 = (46001356684079 / 3357073133568) \
            + (253471410141755 / 5874877983744) * eta \
            - (1693852244423 / 23313007872) * eta**2 \
            - (307833827417 / 2497822272) * eta**3
    y31 = -(356873002170973 / 249880440692736) \
           - (260399751935005 / 8924301453312) * eta \
           + (150484695827 / 35413894656) * eta**2 \
           + (340714213265 / 3794345856) * eta**3
    y22 = -(1062809371 / 20347200) * np.pi**2
    ay50: float = -(3317 * 2 / 252) * np.log(4 * nuI)

    dy0 = 34 / 15 * y0

    dy10 = y10
    dy20 = y20
    dy30 = y30
    dy40 = y40
    dy50 = y50
    day50 = ay50
    dy01 = 28 / 34 * y01
    dy11 = 28 / 34 * y11
    dy21 = 28 / 34 * y21
    dy31 = 28 / 34 * y31
    dy02 = 25 / 34 * y02
    dy12 = 25 / 34 * y12
    dy22 = 25 / 34 * y22
    dy03 = 22 / 34 * y03
    dy13 = 22 / 34 * y13
    dy04 = 19 / 34 * y04
    dy05 = 16 / 34 * y05

    # starting values of coefficients that change in the loop
    ap5: float = (3424 / 105) * 2 * np.log(4 * nuI) - 712192 / 105
    ay05: float = (536803271 / 39564000) * 2 * np.log(4 * nuI)
    day05 = 16 / 34 * ay05 - 31576663 / 13188000

    return float(p0 / nuI**8 *
            (1 + dy0
                 + (p1 + dy0 * (dy01 + dy10)) * nuI**2
                 + (p2 + dy0 * (dy02 + dy20)) * nuI**3
                 + (p3 + dy0 * (dy03 + dy30 + dy11)) * nuI**4
                 + (p4 + dy0 * (dy04 + dy40 + dy12 + dy21)) * nuI**5
                 + (p5 + ap5 + dy0 * (dy05 + day05 + dy13 + dy31 + dy22 + dy50 + day50)) * nuI**6
                 + p6 * nuI**7
             ))


@njit()
def TaylorF2_time_fix_helper(params_intrinsic: TaylorF2AlignedSpinParams, delta: float) -> float:
    """This is the TaylorF2 model to 2PN order. DOI: 10.1103/PhysRevD.80.084043
    helper for a proposal that proposes jumps in final time and delta instead of Mt and Mc
    this is a heuristic for SOBHBs, so doesn't need the 3PN spin terms (or the full IMRPhenomD model)
    """
    eta = float((1 - delta**2) / 4)
    chis = float(params_intrinsic.chis)
    chia = float(params_intrinsic.chia)
    Mt = float(params_intrinsic.Mt)
    FI = float(params_intrinsic.FI)

    c0 = 3 / (128 * eta)
    c1 = 20 / 9 * (743 / 336 + 11 / 4 * eta)
    c2 = -16 * np.pi + 113 / 3 * delta * chia + (113 / 3 - 76 / 3 * eta) * chis
    c3 = 15293365 / 508032 + 27145 / 504 * eta + 3085 / 72 * eta**2 + (-405 / 8 + 200 * eta) * chia**2 - 405 / 4 * delta * chia * chis + (-405 / 8 + 5 / 2 * eta) * chis**2
    c4 = 38645 / 756 * np.pi - 65 / 9 * np.pi * eta + delta * (-732985 / 2268 - 140 / 9 * eta) * chia + (-732985 / 2268 + 24260 / 81 * eta + 340 / 9 * eta**2) * chis
    c5 = 11583231236531 / 4694215680 - 6848 / 21 * GAMMA - 640 / 3 * np.pi**2 + (-15737765635 / 3048192 + 2255 / 12 * np.pi**2) * eta \
            + 76055 / 1728 * eta**2 - 127825 / 1296 * eta**3 \
            + 2270 / 3 * np.pi * delta * chia \
            + (2270 / 3 * np.pi - 520 * np.pi * eta) * chis
    c6 = 77096675 / 254016 * np.pi + 378515 / 1512 * np.pi * eta - 74045 / 756 * np.pi * eta**2 \
            + delta * (-25150083775 / 3048192 + 26804935 / 6048 * eta - 1985 / 48 * eta**2) * chia \
            + (-25150083775 / 3048192 + 10566655595 / 762048 * eta - 1042165 / 3024 * eta**2 + 5345 / 36 * eta**3) * chis

    p0 = 5 * c0 * Mt / 6
    p1 = 3 / 5 * c1
    p2 = 2 / 5 * c2
    p3 = 1 / 5 * c3
    p4 = -3 / 5 * c4
    p5 = 6848 - 1 / 5 * c5
    p6 = -2 / 5 * c6

    nuI: float = (np.pi * Mt * FI)**(1 / 3)
    return float(p0 / nuI**8 * (1 + p1 * nuI**2 + p2 * nuI**3 + p3 * nuI**4 + p4 * nuI**5 + (6848 / 105 * np.log(4 * nuI) - 712192 / 105 + p5) * nuI**6 + p6 * nuI**7))


@njit()
def TaylorF2_ref_time_match(params_intrinsic: TaylorF2AlignedSpinParams, include_pn_SS3: int = 0) -> float:
    """This is the TaylorF2 model to 2PN order. DOI: 10.1103/PhysRevD.80.084043"""
    # TODO need to use imrphenomd instead
    Mt = float(params_intrinsic.Mt)
    Mc = float(params_intrinsic.Mc)
    FI = float(params_intrinsic.FI)
    chis = float(params_intrinsic.chis)
    chia = float(params_intrinsic.chia)
    eta = float((Mc / Mt)**(5 / 3))

    if eta >= 0.25:
        delta = 0.
    else:
        delta = np.sqrt(1 - 4 * eta)

    c0 = 3 / (128 * eta)
    c1 = 20 / 9 * (743 / 336 + 11 / 4 * eta)
    c2 = -16 * np.pi + 113 / 3 * delta * chia + (113 / 3 - 76 / 3 * eta) * chis
    c3 = 15293365 / 508032 + 27145 / 504 * eta + 3085 / 72 * eta**2 + (-405 / 8 + 200 * eta) * chia**2 - 405 / 4 * delta * chia * chis + (-405 / 8 + 5 / 2 * eta) * chis**2
    c4 = 38645 / 756 * np.pi - 65 / 9 * np.pi * eta + delta * (-732985 / 2268 - 140 / 9 * eta) * chia + (-732985 / 2268 + 24260 / 81 * eta + 340 / 9 * eta**2) * chis
    c5 = 11583231236531 / 4694215680 - 6848 / 21 * GAMMA - 640 / 3 * np.pi**2 + (-15737765635 / 3048192 + 2255 / 12 * np.pi**2) * eta \
            + 76055 / 1728 * eta**2 - 127825 / 1296 * eta**3 \
            + 2270 / 3 * np.pi * delta * chia \
            + (2270 / 3 * np.pi - 520 * np.pi * eta) * chis
    c6 = 77096675 / 254016 * np.pi + 378515 / 1512 * np.pi * eta - 74045 / 756 * np.pi * eta**2 \
            + delta * (-25150083775 / 3048192 + 26804935 / 6048 * eta - 1985 / 48 * eta**2) * chia \
            + (-25150083775 / 3048192 + 10566655595 / 762048 * eta - 1042165 / 3024 * eta**2 + 5345 / 36 * eta**3) * chis

    if include_pn_SS3:
        # the spin-spin term for the 3pn post newtonian term
        chiT = chis**2 + chia**2
        chiD = chis**2 - chia**2
        pn_SS3 = 5 / 4032 * ((7 * (30206 - (86732 + 19616 * eta) * eta) * chiT + 7 * (60412 - 52640 * eta) * chia * chis * delta) + (235260 + 249760 * eta) * eta * chiD)
        c5 = c5 + pn_SS3

    p0 = 5 * c0 * Mt / 6
    p1 = 3 / 5 * c1
    p2 = 2 / 5 * c2
    p3 = 1 / 5 * c3
    p4 = -3 / 5 * c4
    p5 = 6848 - 1 / 5 * c5
    p6 = -2 / 5 * c6

    nuI: float = (np.pi * Mt * FI)**(1 / 3)
    return float(p0 / nuI**8 * (1 + p1 * nuI**2 + p2 * nuI**3 + p3 * nuI**4 + p4 * nuI**5 + (6848 / 105 * np.log(4 * nuI) - 712192 / 105 + p5) * nuI**6 + p6 * nuI**7))


@njit()
def TaylorF2_aligned_inplace(intrinsic_waveform: StationaryWaveformFreq, params_intrinsic: TaylorF2AlignedSpinParams, nf_lim: PixelGenericRange, *, include_phenom_amp: int = 1, include_pn_SS3: int = 0, t_offset: float = 0., tc_mode: int = 0) -> float:
    """This is the TaylorF2 model to 3.5PN order. DOI: 10.1103/PhysRevD.80.084043, matching IMRPhenom phenomenological coefficents, see
    for reference AmpInsAnsatz and TaylorF2AlignedPhasing from IMRPhenomD_internals.c
    if include_phenom_amp, include the 3 phenomenilogical coefficients in the amplitude beyond 3.5PN.
    if include_pn_SS3 is true include the 3PN spin-spin term, which was not known when the phenomonelogical amplitude coefficients set by phenom_amp were fit so may not be compatible
    """
    FS = intrinsic_waveform.F
    PSI = intrinsic_waveform.PF
    TS = intrinsic_waveform.TF
    TPS = intrinsic_waveform.TFp
    AS = intrinsic_waveform.AF
    # spins from arXiv : 1508.07253
    # TODO enforce keeping frequency below lso
    # TODO if this is always on a grid eliminate Fs argument
    Mt = params_intrinsic.Mt
    Mc = params_intrinsic.Mc
    FI = params_intrinsic.FI
    phic = params_intrinsic.phic
    eta = float((Mc / Mt)**(5.0 / 3.0))
    if eta >= 0.25:
        delta = 0.
    else:
        delta = float(np.sqrt(1 - 4 * eta))

    chis = float(params_intrinsic.chis)
    chia = float(params_intrinsic.chia)
    logD = float(params_intrinsic.logD)

    flso: float = float(1 / (6**(3 / 2) * np.pi * Mt))
    nulso: float = float((np.pi * Mt * flso)**(1.0 / 3.0))
    # see cutler 1998 for fisher analytics
    c0 = 3 / (128 * eta)
    c1 = 20 / 9 * (743 / 336 + 11 / 4 * eta)
    c2 = -16 * np.pi + 113 / 3 * delta * chia + (113 / 3 - 76 / 3 * eta) * chis
    c3 = 15293365 / 508032 + 27145 / 504 * eta + 3085 / 72 * eta**2 + (-405 / 8 + 200 * eta) * chia**2 - 405 / 4 * delta * chia * chis + (-405 / 8 + 5 / 2 * eta) * chis**2
    c4 = 38645 / 756 * np.pi - 65 / 9 * np.pi * eta + delta * (-732985 / 2268 - 140 / 9 * eta) * chia + (-732985 / 2268 + 24260 / 81 * eta + 340 / 9 * eta**2) * chis
    c5 = 11583231236531 / 4694215680 - 6848 / 21 * GAMMA - 640 / 3 * np.pi**2 + (-15737765635 / 3048192 + 2255 / 12 * np.pi**2) * eta \
            + 76055 / 1728 * eta**2 - 127825 / 1296 * eta**3 \
            + 2270 / 3 * np.pi * delta * chia \
            + (2270 / 3 * np.pi - 520 * np.pi * eta) * chis
    c6 = 77096675 / 254016 * np.pi + 378515 / 1512 * np.pi * eta - 74045 / 756 * np.pi * eta**2 \
            + delta * (-25150083775 / 3048192 + 26804935 / 6048 * eta - 1985 / 48 * eta**2) * chia \
            + (-25150083775 / 3048192 + 10566655595 / 762048 * eta - 1042165 / 3024 * eta**2 + 5345 / 36 * eta**3) * chis

    if include_pn_SS3:
        # the spin-spin term for the 3pn post newtonian term
        chiT = chis**2 + chia**2
        chiD = chis**2 - chia**2
        pn_SS3 = 5 / 4032 * ((7 * (30206 - (86732 + 19616 * eta) * eta) * chiT + 7 * (60412 - 52640 * eta) * chia * chis * delta) + (235260 + 249760 * eta) * eta * chiD)
        c5 = c5 + pn_SS3

    p0 = 5 * c0 * Mt / 6
    p1 = 3 / 5 * c1
    p2 = 2 / 5 * c2
    p3 = 1 / 5 * c3
    p4 = -3 / 5 * c4
    p5 = 6848 - 1 / 5 * c5
    p6 = -2 / 5 * c6

    f0 = 20 / 9 * np.pi * Mt**2 * c0
    f1 = 9 / 20 * c1
    f2 = c2 / 4
    f3 = c3 / 10
    f4 = -9 * c4 / 40
    f5 = 856 / 105 - c5 / 20
    f6 = -c6 / 20

    # TODO figure out the amplitude discrepancy
    a0: float = 1 / 2 * np.sqrt(5 / 6) / np.pi**(2 / 3) * Mc**(5 / 6) / np.exp(logD) * (np.pi * Mt)**(7 / 6)
    a1 = -323 / 224 + 451 / 168 * eta
    a2 = 27 / 8 * delta * chia + (27 / 8 - 11 * eta / 6) * chis
    a3 = -27312085 / 8128512 - 1975055 / 338688 * eta + 105271 / 24192 * eta**2 + (-81 / 32 + 8 * eta) * chia**2 - 81 / 16 * delta * chia * chis + (-81 / 32 + 17 / 8 * eta) * chis**2
    a4 = -85 / 64 * np.pi + 85 / 16 * np.pi * eta + delta * (285197 / 16128 - 1579 / 4032 * eta) * chia + (285197 / 16128 - 15317 / 672 * eta - 2227 / 1008 * eta**2) * chis
    a5 = (-177520268561 / 8583708672 + (545384828789 / 5007163392 - 205 / 48 * np.pi**2) * eta
            - 3248849057 / 178827264 * eta**2 + 34473079 / 6386688 * eta**3
            + (1614569 / 64512 - 1873643 / 16128 * eta + 2167 / 42 * eta**2) * chia**2 + (31 / 12 * np.pi - 7 / 3 * np.pi * eta) * chis
            + (1614569 / 64512 - 61391 / 1344 * eta + 57451 / 4032 * eta**2) * chis**2 + delta * chia * (31 / 12 * np.pi + (1614569 / 32256 - 165961 / 2688 * eta) * chis))
    if include_phenom_amp:
        # PN reduced spin  See Eq 5.9 in http://arxiv.org/pdf/1107.1267v2.pdf
        chiPN = chis * (1. - 76 / 113 * eta) + delta * chia
        xi = -1 + chiPN
        # coefficients a6-a8 are rhos to match phenomenological model from Table 5 in arXiv:1508.07253
        rho1 = 3931.8979897196696 - 17395.758706812805 * eta \
                + (3132.375545898835 + 343965.86092361377 * eta - 1.2162565819981997e6 * eta**2) * xi \
                + (-70698.00600428853 + 1.383907177859705e6 * eta - 3.9662761890979446e6 * eta**2) * xi**2 \
                + (-60017.52423652596 + 803515.1181825735 * eta - 2.091710365941658e6 * eta**2) * xi**3
        rho2 = -40105.47653771657 + 112253.0169706701 * eta \
                + (23561.696065836168 - 3.476180699403351e6 * eta + 1.137593670849482e7 * eta**2) * xi \
                + (754313.1127166454 - 1.308476044625268e7 * eta + 3.6444584853928134e7 * eta**2) * xi**2 \
                + (596226.612472288 - 7.4277901143564405e6 * eta + 1.8928977514040343e7 * eta**2) * xi**3
        rho3 = 83208.35471266537 - 191237.7264145924 * eta \
                + (-210916.2454782992 + 8.71797508352568e6 * eta - 2.6914942420669552e7 * eta**2) * xi \
                + (-1.9889806527362722e6 + 3.0888029960154563e7 * eta - 8.390870279256162e7 * eta**2) * xi**2 \
                + (-1.4535031953446497e6 + 1.7063528990822166e7 * eta - 4.2748659731120914e7 * eta**2) * xi**3
        a6: float = 1 / np.pi**(7 / 3) * rho1
        a7: float = 1 / np.pi**(8 / 3) * rho2
        a8 = 1 / np.pi**3 * rho3
    else:
        a6 = 0.
        a7 = 0.
        a8 = 0.

    nuI: float = (np.pi * Mt * FI)**(1 / 3)
    ac5I: float = -6848 / 21 * np.log(4 * nuI)
    ap5I: float = 6848 / 105 * np.log(4 * nuI) - 712192 / 105
    m4I: float = 1 + 3 * np.log(nuI / nulso)

    if tc_mode:
        TTRef: float = t_offset
    else:
        TTRef = p0 / nuI**8 * (1 + p1 * nuI**2 + p2 * nuI**3 + p3 * nuI**4 + p4 * nuI**5 + (ap5I + p5) * nuI**6 + p6 * nuI**7) + t_offset
    PSII: float = 2 * np.pi * TTRef * FI - np.pi / 4 + c0 / nuI**5 * (1 + c1 * nuI**2 + c2 * nuI**3 + c3 * nuI**4 + c4 * m4I * nuI**5 + (c5 + ac5I) * nuI**6 + c6 * nuI**7)
    PSI_ref: float = -2 * phic - PSII - np.pi / 4

    for n in range(nf_lim.nx_min, nf_lim.nx_max):
        nu: float = (np.pi * Mt)**(1 / 3) * FS[n]**(1 / 3)
        m4: float = 1 - 3 * np.log(nulso) + 3 * np.log(nu)
        fp5: float = 1712 / 105 * (np.log(4) + np.log(nu))
        ac5: float = -6848 / 21 * (np.log(4) + np.log(nu))
        ap5: float = 6848 / 105 * (np.log(4) + np.log(nu)) - 712192 / 105
        PSI[n] = 2 * np.pi * TTRef * FS[n] + PSI_ref - np.pi / 4 + c0 / nu**5 * (1 + c1 * nu**2 + c2 * nu**3 + c3 * nu**4 + c4 * m4 * nu**5 + (c5 + ac5) * nu**6 + c6 * nu**7)
        TS[n] = TTRef - p0 / nu**8 * (1 + p1 * nu**2 + p2 * nu**3 + p3 * nu**4 + p4 * nu**5 + (ap5 + p5) * nu**6 + p6 * nu**7)
        TPS[n] = f0 / nu**11 * (1 + f1 * nu**2 + f2 * nu**3 + f3 * nu**4 + f4 * nu**5 + (f5 + fp5) * nu**6 + f6 * nu**7)
        # AS[n] = a0*FS[n]**(-7/6)*(1+a1*(np.pi*FS[n])**(2/3)+a2*(np.pi*FS[n])+a3*(np.pi*FS[n])**(4/3)+a4*(np.pi*FS[n])**(5/3)+a5*(np.pi*FS[n])**2)
        AS[n] = a0 / np.sqrt(nu)**7 * (1 + a1 * nu**2 + a2 * nu**3 + a3 * nu**4 + a4 * nu**5 + a5 * nu**6 + a6 * nu**7 + a7 * nu**8 + a8 * nu**9)
    return float(TTRef)
