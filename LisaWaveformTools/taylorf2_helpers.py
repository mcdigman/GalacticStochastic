"""Functions to compute rigid adiabatic response in frequency domain."""


import numpy as np
from numba import njit

from LisaWaveformTools.binary_params_manager import BinaryIntrinsicParams
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange

GAMMA = 0.57721566490153286060651209008240243104215933593992  # Euler-Mascheroni constant

# TODO everything here should use PNPhasingSeries for compactness
# TODO make this the only place the code gets the waveforms from
# TODO check timing models
# TODO allow efficient selection of model without calling inplace methods


@njit()
def TaylorF2_inplace(intrinsic_waveform: StationaryWaveformFreq, params_intrinsic: BinaryIntrinsicParams, nf_lim: PixelGenericRange, *, amplitude_pn_mode: int = 1, t_offset: float = 0., tc_mode: int = 0) -> float:
    """Compute TaylorF2 model to 2PN order.

    DOI: 10.1103/PhysRevD.80.084043
    """
    FS = intrinsic_waveform.F
    PSI = intrinsic_waveform.PF
    TS = intrinsic_waveform.TF
    TPS = intrinsic_waveform.TFp
    AS = intrinsic_waveform.AF

    Mt = params_intrinsic.mass_total_detector_sec
    Mc = params_intrinsic.mass_chirp_detector_sec
    f_i = params_intrinsic.frequency_i_hz
    phic = params_intrinsic.phase_c
    log_dl = params_intrinsic.ln_luminosity_distance_m

    eta: float = float((Mc / Mt)**(5 / 3))
    f_lso: float = 1 / (6**(3 / 2) * np.pi * Mt)
    nu_lso: float = float((np.pi * Mt * f_lso)**(1 / 3))
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

    nu_i: float = float((np.pi * Mt * f_i)**(1 / 3))
    if tc_mode:
        TTRef: float = t_offset
    else:
        TTRef = float(p0 / nu_i**8 * (1 + p1 * nu_i**2 + p2 * nu_i**3 + p3 * nu_i**4 + p4 * nu_i**5 + (6848 / 105 * np.log(4 * nu_i) - 712192 / 105 + p5) * nu_i**6 + p6 * nu_i**7) + t_offset)
    psi_i: float = float(2 * np.pi * TTRef * f_i - np.pi / 4 + c0 / nu_i**5 * (1 + c1 * nu_i**2 + c2 * nu_i**3 + c3 * nu_i**4 + c4 * (1 + 3 * np.log(nu_i / nu_lso)) * nu_i**5 + (c5 - 6848 / 21 * np.log(4 * nu_i)) * nu_i**6 + c6 * nu_i**7))
    psi_ref: float = -2 * phic - psi_i - np.pi / 4

    Amp: float = float(np.sqrt(5 / 6) / np.pi**(2 / 3) * Mc**(5 / 6) / np.exp(log_dl))
    for n in range(nf_lim.nx_min, nf_lim.nx_max):
        nu: float = float((np.pi * Mt * FS[n])**(1 / 3))
        m4: float = float(1 + 3 * np.log(nu / nu_lso))
        fp5: float = float(1712 / 105 * np.log(4 * nu))
        ac5: float = float(-6848 / 21 * np.log(4 * nu))
        ap5: float = float(6848 / 105 * np.log(4 * nu) - 712192 / 105)
        PSI[n] = 2 * np.pi * TTRef * FS[n] + psi_ref - np.pi / 4 + c0 / nu**5 * (1 + c1 * nu**2 + c2 * nu**3 + c3 * nu**4 + c4 * m4 * nu**5 + (c5 + ac5) * nu**6 + c6 * nu**7)
        TS[n] = TTRef - p0 / nu**8 * (1 + p1 * nu**2 + p2 * nu**3 + p3 * nu**4 + p4 * nu**5 + (ap5 + p5) * nu**6 + p6 * nu**7)
        TPS[n] = f0 / nu**11 * (1 + f1 * nu**2 + f2 * nu**3 + f3 * nu**4 + f4 * nu**5 + (f5 + fp5) * nu**6 + f6 * nu**7)
        if amplitude_pn_mode == 0:
            AS[n] = Amp * FS[n]**(-7 / 6)
        else:
            AS[n] = Amp * (FS[n]**(-7 / 6) + (9 / 40 * np.pi**(2 / 3)) * (Mt**(2 / 3) * c1) * FS[n]**(-1 / 2) + np.pi / 8 * (Mt * c2) * FS[n]**(-1 / 6))
    return float(TTRef)


@njit()
def TaylorF2_eccentric_inplace(intrinsic_waveform: StationaryWaveformFreq, params_intrinsic: BinaryIntrinsicParams, nf_lim: PixelGenericRange, amplitude_pn_mode: int = 0, t_offset: float = 0., tc_mode: int = 0) -> float:
    """Compute TaylorF2 model with eccentricity but no spin.

    From DOI: 10.1103/PhysRevD.93.124061
    """
    Mc = params_intrinsic.mass_chirp_detector_sec
    Mt = params_intrinsic.mass_total_detector_sec
    f_i = params_intrinsic.frequency_i_hz
    phic = params_intrinsic.phase_c
    log_dl = params_intrinsic.ln_luminosity_distance_m
    eta = params_intrinsic.symmetric_mass_ratio
    # note that psi_i and psi_ref are slightly different from the no eccentricity case, but it seems to just be a convention that cancels in the results
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

    nu_i: float = float((np.pi * Mt * f_i)**(1 / 3))

    y0 = -(2355 / 1462) * params_intrinsic.eccentricity_i**2
    y01 = (299076223 / 81976608) + (18766963 / 2927736) * eta
    y02 = -(2819123 / 282600) * np.pi
    y03 = (16237683263 / 3330429696) \
            + (24133060753 / 971375328) * eta \
            + (1562608261 / 69383952) * eta**2
    y04 = -((2831492681 / 118395270) + (11552066831 / 270617760) * eta) * np.pi
    y05: float = float(-(43603153867072577087 / 132658535116800000)
            + (536803271 / 19782000) * GAMMA
            + (15722503703 / 325555200) * np.pi**2
            + ((299172861614477 / 689135247360) - (15075413 / 1446912) * np.pi**2) * eta
            + (3455209264991 / 41019955200) * eta**2
            + (50612671711 / 878999040) * eta**3
            + (3843505163 / 59346000) * np.log(2)
            - (1121397129 / 17584000) * np.log(3))
    y10 = (2833 / 1008) - (197 / 36) * eta
    y20 = (377 / 72) * np.pi
    y30 = -(1193251 / 3048192) - (66317 / 9072) * eta + (18155 / 1296) * eta**2
    y40 = ((764881 / 90720) - (949457 / 22680) * eta) * np.pi
    y50: float = float((26531900578691 / 168991764480)
        - (3317 / 126) * GAMMA
        + (122833 / 10368) * np.pi**2
        + ((9155185261 / 548674560) - (3977 / 1152) * np.pi**2) * eta
        - (5732473 / 1306368) * eta**2
        - (3090307 / 139968) * eta**3
        + (87419 / 1890) * np.log(2)
        - (26001 / 560) * np.log(3))
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
    ay50: float = float(-(3317 * 2 / 252) * np.log(4 * nu_i))

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
    m4_i: float = float(1 + 3 * np.log(nu_i))
    ac5_i: float = float(-(3424 / 21) * 2 * np.log(4 * nu_i))
    ap5_i: float = float((3424 / 105) * 2 * np.log(4 * nu_i) - 712192 / 105)
    ay05I: float = float((536803271 / 39564000) * 2 * np.log(4 * nu_i))
    day05I = 16 / 34 * ay05I - 31576663 / 13188000

    if tc_mode:
        TTRef: float = t_offset
    else:
        TTRef = p0 / nu_i**8 * \
                (1 + dy0
                     + (p1 + dy0 * (dy01 + dy10)) * nu_i**2
                     + (p2 + dy0 * (dy02 + dy20)) * nu_i**3
                     + (p3 + dy0 * (dy03 + dy30 + dy11)) * nu_i**4
                     + (p4 + dy0 * (dy04 + dy40 + dy12 + dy21)) * nu_i**5
                     + (p5 + ap5_i + dy0 * (dy05 + day05I + dy13 + dy31 + dy22 + dy50 + day50)) * nu_i**6
                     + p6 * nu_i**7
                 ) + t_offset

    psi_i: float = 2 * np.pi * TTRef * f_i - np.pi / 4 + c0 / nu_i**5 * \
        (1 + y0
            + (c1 + y0 * (y01 + y10)) * nu_i**2
            + (c2 + y0 * (y02 + y20)) * nu_i**3
            + (c3 + y0 * (y03 + y11 + y30)) * nu_i**4
            + (c4 * m4_i + y0 * (y04 + y12 + y21 + y40)) * nu_i**5
            + (c5 + ac5_i + y0 * (y05 + ay05I + y13 + y22 + y31 + y50 + ay50)) * nu_i**6
            + c6 * nu_i**7
        )
    psi_ref: float = -2 * phic - psi_i - np.pi / 4

    Amp: float = float(np.sqrt(5 / 6) / np.pi**(2 / 3) * Mc**(5 / 6) / np.exp(log_dl))
    for n in range(nf_lim.nx_min, nf_lim.nx_max):
        nu: float = float((np.pi * Mt * FS[n])**(1 / 3))

        # get correct values for the coefficients that change
        m4: float = float(1 + 3 * np.log(nu))
        ac5: float = float(-(3424 / 21) * 2 * np.log(4 * nu))
        ap5: float = float((3424 / 105) * 2 * np.log(4 * nu) - 712192 / 105)
        af5: float = float((856 / 105) * 2 * np.log(4 * nu))
        ay05: float = float((536803271 / 39564000) * 2 * np.log(4 * nu))
        day05 = 16 / 34 * ay05 - 31576663 / 13188000
        dday05 = 25 / 43 * day05 - 734341 / 824250

        PSI[n] = 2 * np.pi * TTRef * FS[n] + psi_ref - np.pi / 4 + c0 / nu**5 * \
            (1
              + c1 * nu**2
              + c2 * nu**3
              + c3 * nu**4
              + m4 * c4 * nu**5
              + (c5 + ac5) * nu**6
              + c6 * nu**7
              + y0 * (nu_i / nu)**(19 / 3) *
                (1
                    + y01 * nu**2
                    + y10 * nu_i**2
                    + y02 * nu**3
                    + y20 * nu_i**3
                    + y03 * nu**4
                    + y11 * nu**2 * nu_i**2
                    + y30 * nu_i**4
                    + y04 * nu**5
                    + y12 * nu**3 * nu_i**2
                    + y21 * nu**2 * nu_i**3
                    + y40 * nu_i**5
                    + (y05 + ay05) * nu**6
                    + y13 * nu**4 * nu_i**2
                    + y22 * nu**3 * nu_i**3
                    + y31 * nu**2 * nu_i**4
                    + (y50 + ay50) * nu_i**6
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
                + dy0 * (nu_i / nu)**(19 / 3) *
                    (1
                       + dy01 * nu**2
                       + dy10 * nu_i**2
                       + dy02 * nu**3
                       + dy20 * nu_i**3 + dy03 * nu**4
                       + dy11 * nu**2 * nu_i**2
                       + dy30 * nu_i**4
                       + dy04 * nu**5
                       + dy12 * nu**3 * nu_i**2
                       + dy21 * nu**2 * nu_i**3
                       + dy40 * nu_i**5
                       + (dy05 + day05) * nu**6
                       + dy13 * nu**4 * nu_i**2
                       + dy22 * nu**3 * nu_i**3
                       + dy31 * nu**2 * nu_i**4
                       + (dy50 + day50) * nu_i**6
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
                    + ddy0 * (nu_i / nu)**(19 / 3) *
                        (1
                         + ddy01 * nu**2
                         + ddy10 * nu_i**2
                         + ddy02 * nu**3
                         + ddy20 * nu_i**3
                         + ddy03 * nu**4
                         + ddy11 * nu**2 * nu_i**2
                         + ddy30 * nu_i**4
                         + ddy04 * nu**5
                         + ddy12 * nu**3 * nu_i**2
                         + ddy21 * nu**2 * nu_i**3
                         + ddy40 * nu_i**5
                         + (ddy05 + dday05) * nu**6
                         + ddy13 * nu**4 * nu_i**2
                         + ddy22 * nu**3 * nu_i**3
                         + ddy31 * nu**2 * nu_i**4
                         + (ddy50 + dday50) * nu_i**6
                        )
                   )

        # TODO carry out amp to appropriate order
        if amplitude_pn_mode == 0:
            # in the paper, amplitude is carried out to 0pn order
            AS[n] = Amp * FS[n]**(-7 / 6)
        else:
            AS[n] = Amp * (FS[n]**(-7 / 6) + (9 / 40 * np.pi**(2 / 3)) * (Mt**(2 / 3) * c1) * FS[n]**(-1 / 2) + np.pi / 8 * (Mt * c2) * FS[n]**(-1 / 6))

    return float(TTRef)


@njit()
def TaylorF2_eccentricity_solve(params_intrinsic: BinaryIntrinsicParams, f_i2: float) -> float:
    """Get an inferred eccentricity at a later frequency based on the taylorf2 eccentricity model."""
    e01 = params_intrinsic.eccentricity_i
    Mt = params_intrinsic.mass_total_detector_sec
    f_i = params_intrinsic.frequency_i_hz
    eta = params_intrinsic.symmetric_mass_ratio
    # note that psi_i and psi_ref are slightly different from the no eccentricity case, but it seems to just be a convention that cancels in the results
    # see cutler 1998 for fisher analytics
    # see arXiv : 2001.11412 v1.pdf
    # TODO write tests for self consistency and consistency with other waveform models
    assert f_i > 0.0
    assert f_i2 > 0.0

    nu_i1: float = float((np.pi * Mt * f_i)**(1 / 3))
    nu_i2: float = float((np.pi * Mt * f_i2)**(1 / 3))

    y0 = -(2355 / 1462) * e01**2
    y01 = (299076223 / 81976608) + (18766963 / 2927736) * eta
    y02 = -(2819123 / 282600) * np.pi
    y03 = (16237683263 / 3330429696) \
            + (24133060753 / 971375328) * eta \
            + (1562608261 / 69383952) * eta**2
    y04 = -((2831492681 / 118395270) + (11552066831 / 270617760) * eta) * np.pi
    y05: float = float(-(43603153867072577087 / 132658535116800000)
            + (536803271 / 19782000) * GAMMA
            + (15722503703 / 325555200) * np.pi**2
            + ((299172861614477 / 689135247360) - (15075413 / 1446912) * np.pi**2) * eta
            + (3455209264991 / 41019955200) * eta**2
            + (50612671711 / 878999040) * eta**3
            + (3843505163 / 59346000) * np.log(2)
            - (1121397129 / 17584000) * np.log(3))
    y10 = (2833 / 1008) - (197 / 36) * eta
    y20 = (377 / 72) * np.pi
    y30 = -(1193251 / 3048192) - (66317 / 9072) * eta + (18155 / 1296) * eta**2
    y40 = ((764881 / 90720) - (949457 / 22680) * eta) * np.pi
    y50: float = float((26531900578691 / 168991764480)
        - (3317 / 126) * GAMMA
        + (122833 / 10368) * np.pi**2
        + ((9155185261 / 548674560) - (3977 / 1152) * np.pi**2) * eta
        - (5732473 / 1306368) * eta**2
        - (3090307 / 139968) * eta**3
        + (87419 / 1890) * np.log(2)
        - (26001 / 560) * np.log(3))
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

    ddy0_derived: float = float(ddy0 * (nu_i1 / nu_i2)**(19 / 3) *
          (
            1
            + ddy10 * nu_i1**2
            + ddy20 * nu_i1**3
            + ddy30 * nu_i1**4
            + ddy40 * nu_i1**5
            + ddy50 * nu_i1**6
            - (3317 * nu_i1**6 * np.log(4 * nu_i1)) / 126
            + ddy01 * nu_i2**2
            + ddy02 * nu_i2**3
            + ddy03 * nu_i2**4
            + ddy04 * nu_i2**5
            - (30107981 * nu_i2**6) / 13188000 + ddy05 * nu_i2**6 + (734341 * nu_i2**6 * np.log(4 * nu_i2)) / 98910
            + ddy11 * nu_i1**2 * nu_i2**2
            + ddy21 * nu_i1**3 * nu_i2**2
            + ddy31 * nu_i1**4 * nu_i2**2
            + ddy12 * nu_i1**2 * nu_i2**3
            + ddy22 * nu_i1**3 * nu_i2**3
            + ddy13 * nu_i1**2 * nu_i2**4
          )
        /
          (
            1
            + ddy01 * nu_i2**2
            + ddy10 * nu_i2**2
            + ddy02 * nu_i2**3
            + ddy20 * nu_i2**3
            + ddy03 * nu_i2**4
            + ddy11 * nu_i2**4
            + ddy30 * nu_i2**4
            + ddy04 * nu_i2**5
            + ddy12 * nu_i2**5
            + ddy21 * nu_i2**5
            + ddy40 * nu_i2**5
            - (30107981 * nu_i2**6) / 13188000 + ddy05 * nu_i2**6 + ddy13 * nu_i2**6 + ddy22 * nu_i2**6 + ddy31 * nu_i2**6 + ddy50 * nu_i2**6 - (44512 * nu_i2**6 * np.log(4 * nu_i2)) / 2355
          ))

    return float(np.sqrt(-24 / 157 * ddy0_derived))


@njit()
def TaylorF2_eccentric_TTRef(params_intrinsic: BinaryIntrinsicParams) -> float:
    """
    Get just TTRef for input parameters.

    DOI: 10.1103/PhysRevD.93.124061
    """
    Mt = params_intrinsic.mass_total_detector_sec
    f_i = params_intrinsic.frequency_i_hz
    eta = params_intrinsic.symmetric_mass_ratio
    # note that psi_i and psi_ref are slightly different from the no eccentricity case, but it seems to just be a convention that cancels in the results
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

    nu_i: float = float((np.pi * Mt * f_i)**(1 / 3))

    y0 = -(2355 / 1462) * params_intrinsic.eccentricity_i**2
    y01 = (299076223 / 81976608) + (18766963 / 2927736) * eta
    y02 = -(2819123 / 282600) * np.pi
    y03 = (16237683263 / 3330429696) \
            + (24133060753 / 971375328) * eta \
            + (1562608261 / 69383952) * eta**2
    y04 = -((2831492681 / 118395270) + (11552066831 / 270617760) * eta) * np.pi
    y05: float = float(-(43603153867072577087 / 132658535116800000)
            + (536803271 / 19782000) * GAMMA
            + (15722503703 / 325555200) * np.pi**2
            + ((299172861614477 / 689135247360) - (15075413 / 1446912) * np.pi**2) * eta
            + (3455209264991 / 41019955200) * eta**2
            + (50612671711 / 878999040) * eta**3
            + (3843505163 / 59346000) * np.log(2)
            - (1121397129 / 17584000) * np.log(3))
    y10 = (2833 / 1008) - (197 / 36) * eta
    y20 = (377 / 72) * np.pi
    y30 = -(1193251 / 3048192) - (66317 / 9072) * eta + (18155 / 1296) * eta**2
    y40 = ((764881 / 90720) - (949457 / 22680) * eta) * np.pi
    y50: float = float((26531900578691 / 168991764480)
        - (3317 / 126) * GAMMA
        + (122833 / 10368) * np.pi**2
        + ((9155185261 / 548674560) - (3977 / 1152) * np.pi**2) * eta
        - (5732473 / 1306368) * eta**2
        - (3090307 / 139968) * eta**3
        + (87419 / 1890) * np.log(2)
        - (26001 / 560) * np.log(3))
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
    ay50: float = float(-(3317 * 2 / 252) * np.log(4 * nu_i))

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
    ap5: float = float((3424 / 105) * 2 * np.log(4 * nu_i) - 712192 / 105)
    ay05: float = float((536803271 / 39564000) * 2 * np.log(4 * nu_i))
    day05 = 16 / 34 * ay05 - 31576663 / 13188000

    return float(p0 / nu_i**8 *
            (1 + dy0
                 + (p1 + dy0 * (dy01 + dy10)) * nu_i**2
                 + (p2 + dy0 * (dy02 + dy20)) * nu_i**3
                 + (p3 + dy0 * (dy03 + dy30 + dy11)) * nu_i**4
                 + (p4 + dy0 * (dy04 + dy40 + dy12 + dy21)) * nu_i**5
                 + (p5 + ap5 + dy0 * (dy05 + day05 + dy13 + dy31 + dy22 + dy50 + day50)) * nu_i**6
                 + p6 * nu_i**7
             ))


@njit()
def TaylorF2_time_fix_helper(params_intrinsic: BinaryIntrinsicParams, delta: float) -> float:
    """Compute the reference time in the TaylorF2 model to 2PN order.

    DOI: 10.1103/PhysRevD.80.084043
    """
    eta = params_intrinsic.symmetric_mass_ratio
    chis = params_intrinsic.chi_s
    chia = params_intrinsic.chi_a
    Mt = params_intrinsic.mass_total_detector_sec
    f_i = params_intrinsic.frequency_i_hz

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

    nu_i: float = float((np.pi * Mt * f_i)**(1 / 3))
    return float(p0 / nu_i**8 * (1 + p1 * nu_i**2 + p2 * nu_i**3 + p3 * nu_i**4 + p4 * nu_i**5 + (6848 / 105 * np.log(4 * nu_i) - 712192 / 105 + p5) * nu_i**6 + p6 * nu_i**7))


@njit()
def TaylorF2_ref_time_match(params_intrinsic: BinaryIntrinsicParams, include_pn_ss3: int = 0) -> float:
    """
    Compute the reference time in the TaylorF2 model to 2PN order.

    DOI: 10.1103/PhysRevD.80.084043
    """
    # TODO need to use imrphenomd instead
    Mt = params_intrinsic.mass_total_detector_sec
    f_i = params_intrinsic.frequency_i_hz
    chis = params_intrinsic.chi_s
    chia = params_intrinsic.chi_a
    eta = params_intrinsic.symmetric_mass_ratio
    delta = params_intrinsic.mass_delta

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

    if include_pn_ss3:
        # the spin-spin term for the 3pn post newtonian term
        chi_t = chis**2 + chia**2
        chi_d = chis**2 - chia**2
        pn_ss3 = 5 / 4032 * ((7 * (30206 - (86732 + 19616 * eta) * eta) * chi_t + 7 * (60412 - 52640 * eta) * chia * chis * delta) + (235260 + 249760 * eta) * eta * chi_d)
        c5 = c5 + pn_ss3

    p0 = 5 * c0 * Mt / 6
    p1 = 3 / 5 * c1
    p2 = 2 / 5 * c2
    p3 = 1 / 5 * c3
    p4 = -3 / 5 * c4
    p5 = 6848 - 1 / 5 * c5
    p6 = -2 / 5 * c6

    nu_i: float = float((np.pi * Mt * f_i)**(1 / 3))
    return float(p0 / nu_i**8 * (1 + p1 * nu_i**2 + p2 * nu_i**3 + p3 * nu_i**4 + p4 * nu_i**5 + (6848 / 105 * np.log(4 * nu_i) - 712192 / 105 + p5) * nu_i**6 + p6 * nu_i**7))


@njit()
def TaylorF2_aligned_inplace(intrinsic_waveform: StationaryWaveformFreq, params_intrinsic: BinaryIntrinsicParams, nf_lim: PixelGenericRange, *, amplitude_pn_mode: int = 2, include_pn_ss3: int = 0, t_offset: float = 0., tc_mode: int = 0) -> float:
    """
    Compute the TaylorF2 model to 3.5PN order.

    DOI: 10.1103/PhysRevD.80.084043, matching IMRPhenom phenomenological coefficents, see
    for reference AmpInsAnsatz and TaylorF2AlignedPhasing from IMRPhenomD_internals.c
    if include_phenom_amp, include the 3 phenomenological coefficients in the amplitude beyond 3.5PN.
    if include_pn_ss3 is true include the 3PN spin-spin term, which was not known when the phenomonelogical amplitude coefficients set by phenom_amp were fit so may not be compatible
    """
    FS = intrinsic_waveform.F
    PSI = intrinsic_waveform.PF
    TS = intrinsic_waveform.TF
    TPS = intrinsic_waveform.TFp
    AS = intrinsic_waveform.AF
    # spins from arXiv : 1508.07253
    # TODO enforce keeping frequency below lso
    # TODO if this is always on a grid eliminate Fs argument
    Mt = params_intrinsic.mass_total_detector_sec
    Mc = params_intrinsic.mass_chirp_detector_sec
    f_i = params_intrinsic.frequency_i_hz
    phic = params_intrinsic.phase_c
    eta = params_intrinsic.symmetric_mass_ratio
    delta = params_intrinsic.mass_delta

    chis = params_intrinsic.chi_s
    chia = params_intrinsic.chi_a
    log_dl = params_intrinsic.ln_luminosity_distance_m

    f_lso: float = float(1 / (6**(3 / 2) * np.pi * Mt))
    nu_lso: float = float((np.pi * Mt * f_lso)**(1.0 / 3.0))
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

    if include_pn_ss3:
        # the spin-spin term for the 3pn post newtonian term
        chi_t = chis**2 + chia**2
        chi_d = chis**2 - chia**2
        pn_ss3 = 5 / 4032 * ((7 * (30206 - (86732 + 19616 * eta) * eta) * chi_t + 7 * (60412 - 52640 * eta) * chia * chis * delta) + (235260 + 249760 * eta) * eta * chi_d)
        c5 = c5 + pn_ss3

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
    a0: float = float(1 / 2 * np.sqrt(5 / 6) / np.pi**(2 / 3) * Mc**(5 / 6) / np.exp(log_dl) * (np.pi * Mt)**(7 / 6))

    if amplitude_pn_mode == 0:
        a1: float = 0.0
        a2: float = 0.0
        a3: float = 0.0
        a4: float = 0.0
        a5: float = 0.0
    else:
        a1 = -323 / 224 + 451 / 168 * eta
        a2 = 27 / 8 * delta * chia + (27 / 8 - 11 * eta / 6) * chis
        a3 = -27312085 / 8128512 - 1975055 / 338688 * eta + 105271 / 24192 * eta**2 + (-81 / 32 + 8 * eta) * chia**2 - 81 / 16 * delta * chia * chis + (-81 / 32 + 17 / 8 * eta) * chis**2
        a4 = -85 / 64 * np.pi + 85 / 16 * np.pi * eta + delta * (285197 / 16128 - 1579 / 4032 * eta) * chia + (285197 / 16128 - 15317 / 672 * eta - 2227 / 1008 * eta**2) * chis
        a5 = (-177520268561 / 8583708672 + (545384828789 / 5007163392 - 205 / 48 * np.pi**2) * eta
                - 3248849057 / 178827264 * eta**2 + 34473079 / 6386688 * eta**3
                + (1614569 / 64512 - 1873643 / 16128 * eta + 2167 / 42 * eta**2) * chia**2 + (31 / 12 * np.pi - 7 / 3 * np.pi * eta) * chis
                + (1614569 / 64512 - 61391 / 1344 * eta + 57451 / 4032 * eta**2) * chis**2 + delta * chia * (31 / 12 * np.pi + (1614569 / 32256 - 165961 / 2688 * eta) * chis))
    if amplitude_pn_mode == 2:
        # PN reduced spin  See Eq 5.9 in http://arxiv.org/pdf/1107.1267v2.pdf
        chi_pn = chis * (1. - 76 / 113 * eta) + delta * chia
        xi = -1 + chi_pn
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
        a6: float = float(1 / np.pi**(7 / 3) * rho1)
        a7: float = float(1 / np.pi**(8 / 3) * rho2)
        a8: float = float(1 / np.pi**3 * rho3)
    else:
        a6 = 0.
        a7 = 0.
        a8 = 0.

    nu_i: float = float((np.pi * Mt * f_i)**(1 / 3))
    ac5_i: float = float(-6848 / 21 * np.log(4 * nu_i))
    ap5_i: float = float(6848 / 105 * np.log(4 * nu_i) - 712192 / 105)
    m4_i: float = float(1 + 3 * np.log(nu_i / nu_lso))

    if tc_mode:
        TTRef: float = t_offset
    else:
        TTRef = p0 / nu_i**8 * (1 + p1 * nu_i**2 + p2 * nu_i**3 + p3 * nu_i**4 + p4 * nu_i**5 + (ap5_i + p5) * nu_i**6 + p6 * nu_i**7) + t_offset
    psi_i: float = 2 * np.pi * TTRef * f_i - np.pi / 4 + c0 / nu_i**5 * (1 + c1 * nu_i**2 + c2 * nu_i**3 + c3 * nu_i**4 + c4 * m4_i * nu_i**5 + (c5 + ac5_i) * nu_i**6 + c6 * nu_i**7)
    psi_ref: float = -2 * phic - psi_i - np.pi / 4

    for n in range(nf_lim.nx_min, nf_lim.nx_max):
        nu: float = float((np.pi * Mt)**(1 / 3) * FS[n]**(1 / 3))
        m4: float = float(1 - 3 * np.log(nu_lso) + 3 * np.log(nu))
        fp5: float = float(1712 / 105 * (np.log(4) + np.log(nu)))
        ac5: float = float(-6848 / 21 * (np.log(4) + np.log(nu)))
        ap5: float = float(6848 / 105 * (np.log(4) + np.log(nu)) - 712192 / 105)
        PSI[n] = 2 * np.pi * TTRef * FS[n] + psi_ref - np.pi / 4 + c0 / nu**5 * (1 + c1 * nu**2 + c2 * nu**3 + c3 * nu**4 + c4 * m4 * nu**5 + (c5 + ac5) * nu**6 + c6 * nu**7)
        TS[n] = TTRef - p0 / nu**8 * (1 + p1 * nu**2 + p2 * nu**3 + p3 * nu**4 + p4 * nu**5 + (ap5 + p5) * nu**6 + p6 * nu**7)
        TPS[n] = f0 / nu**11 * (1 + f1 * nu**2 + f2 * nu**3 + f3 * nu**4 + f4 * nu**5 + (f5 + fp5) * nu**6 + f6 * nu**7)
        AS[n] = a0 / np.sqrt(nu)**7 * (1 + a1 * nu**2 + a2 * nu**3 + a3 * nu**4 + a4 * nu**5 + a5 * nu**6 + a6 * nu**7 + a7 * nu**8 + a8 * nu**9)
    return float(TTRef)
