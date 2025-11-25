# type shouldn't be needed after atropy 7.2.0
"""A series of small helper functions to load cosmic data from HDF5 files."""

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, Galactocentric, GeocentricTrueEcliptic
from numpy.typing import NDArray
from pandas import DataFrame, HDFStore


def get_amplitude(dat: DataFrame) -> NDArray[np.floating]:
    """Compute the gravitational wave amplitude for each source in the dataset.

    Parameters
    ----------
    dat : DataFrame
        pandas DataFrame containing source parameters including mass_1, mass_2, f_gw, and dist_sun

    Returns
    -------
    amplitude : numpy.ndarray
        Gravitational wave amplitude for each source with appropriate units
    """
    mass_1: NDArray[np.float64] = dat.mass_1.to_numpy().astype(np.float64)
    mass_2: NDArray[np.float64] = dat.mass_2.to_numpy().astype(np.float64)
    f_gw: NDArray[np.float64] = dat.f_gw.to_numpy().astype(np.float64)

    mc = (mass_1 * mass_2)**(3 / 5) / (mass_1 + mass_2)**(1 / 5) * u.Msun
    term1 = 64 / 5 * (const.G * mc)**(10 / 3)
    term2 = (np.pi * f_gw * u.s**(-1))**(4 / 3)
    denom1 = const.c**8 * (dat.dist_sun.to_numpy() * u.kpc)**2
    amplitude = np.sqrt(term1.to(u.m**10 / u.s**(20 / 3)) * term2 / denom1.to(u.m**10 / u.s**8)).value
    return amplitude


def get_Gx_positions(dat: DataFrame) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert Galactocentric Cartesian coordinates to ecliptic longitude and latitude.

    Parameters
    ----------
    dat : DataFrame
        pandas DataFrame containing xGx, yGx, zGx columns for Galactocentric Cartesian coordinates

    Returns
    -------
    lon : numpy.ndarray
        Ecliptic longitude in radians
    lat : numpy.ndarray
        Ecliptic latitude in radians
    """
    galcen = Galactocentric(x=dat.xGx.to_numpy() * u.kpc, y=dat.yGx.to_numpy() * u.kpc, z=dat.zGx.to_numpy() * u.kpc)
    icrs = galcen.transform_to(ICRS())
    ecl = icrs.transform_to(GeocentricTrueEcliptic())
    return ecl.lon.to(u.rad).value, ecl.lat.to(u.rad).value


def get_chirp(dat: DataFrame) -> NDArray[np.floating]:
    """Compute the chirp (df/dt) for each source in the dataset.

    Parameters
    ----------
    dat : DataFrame
        pandas DataFrame containing source parameters including mass_1, mass_2, and f_gw

    Returns
    -------
    chirp : numpy.ndarray
        Chirp (df/dt) for each source with appropriate units
    """
    mass_1 = dat.mass_1.to_numpy().astype(np.float64)
    mass_2 = dat.mass_2.to_numpy().astype(np.float64)
    mc = (mass_1 * mass_2)**(3 / 5) / (mass_1 + mass_2)**(1 / 5) * u.Msun
    fgw = dat.f_gw.to_numpy().astype(np.float64) * u.s**(-1)
    term1 = (const.G * mc)**(5 / 3) / (const.c)**5
    term2 = (np.pi * fgw)**(11 / 3)
    chirp = 96 / (5 * np.pi) * term1 * term2
    return chirp.to(u.s**(-2)).value


def get_inc_phase_pol(n_dat: int, seed: int = 314159265) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Generate random inclination, phase, and polarization angles for each source.

    Parameters
    ----------
    n_dat : int
        Number of sources in the dataset
    seed : int
        Random seed for rng


    Returns
    -------
    inc : numpy.ndarray
        Inclination angles in radians
    phase : numpy.ndarray
        Phase angles in radians
    pol : numpy.ndarray
        Polarization angles in radians
    """
    rng = np.random.default_rng(seed)
    inc = np.arccos(rng.uniform(0, 1, n_dat))
    phase = rng.uniform(0, np.pi, n_dat)
    pol = rng.uniform(0, np.pi, n_dat)
    return inc, phase, pol


def create_dat_in(dat: DataFrame) -> NDArray[np.floating]:
    """Create input data array for waveform generation from source parameters.

    Parameters
    ----------
    dat : DataFrame
        pandas DataFrame containing source parameters including mass_1, mass_2, f_gw, dist_sun, xGx, yGx, zGx

    Returns
    -------
    dat_in : numpy.ndarray
        Array of shape (N, 8) where N is the number of sources, containing the following columns:
        [amplitude, ecliptic longitude, ecliptic latitude, f_gw, chirp, inclination, phase, polarization]
    """
    h = get_amplitude(dat)
    lon, lat = get_Gx_positions(dat)
    chirp = get_chirp(dat)
    inc, phase, pol = get_inc_phase_pol(len(dat))
    dat_in = np.vstack([h, lat, lon, dat.f_gw.to_numpy().astype(np.float64), chirp, inc, phase, pol]).T
    print(np.min(lat), np.max(lat))
    return dat_in


def load_cosmic(filename: str, categories: list[str]) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    if categories != ['dgb']:
        msg = 'COSMIC data only supports dgb category'
        raise ValueError(msg)
    with HDFStore(filename, mode='r') as store:
        galaxy = store['data']
        assert isinstance(galaxy, DataFrame), 'Unexpected format of input file'
        params_gb = create_dat_in(galaxy)
        ns_got = np.array([params_gb.shape[0]], dtype=np.int64)
        return params_gb, ns_got
