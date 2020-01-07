import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body
from astropy import units as u


def sgl_position(RA, DE, distance, time, z, verbose=False):
    """Position of the SGL for a given star, time, and heliocentric distance z

    Parameters
    ----------
    RA : angle
        Position of star in RA (usually in degrees)
    DE : angle
        Position of star in DE (usually in degrees)
    distance : float
        Distance to the star (usually a few parsecs)
    time : astropy.time
        Time of the transmission (GMT)
    z : float
        Heliocentric distance

    Returns
    -------
    sgl_coord : astropy.coordinates
        Location of the focus in the SGL at the time of transmission
    """

    star = SkyCoord(ra=RA, dec=DE, distance=distance)
    loc = EarthLocation(lon=51.4934, lat=0.0098)  # Greenwich
    # approximate Light travel time to sgl (+-1 au is +-15min: irrelevant)
    ltt = (z.to(u.lightyear)).value * u.year  
    sun = get_body('sun', time - ltt, loc, ephemeris='builtin')  # ephemeris='jpl' or 'builtin'
    vector_x = star.cartesian.x - sun.cartesian.x
    vector_y = star.cartesian.y - sun.cartesian.y
    vector_z = star.cartesian.z - sun.cartesian.z      
    length_of_vector = (vector_x**2 + vector_y**2 + vector_z**2)**(1/2)
    #distance_to_sgl = 
    sgl = SkyCoord(
        x=sun.cartesian.x - vector_x * z.to(u.pc) / length_of_vector,
        y=sun.cartesian.y - vector_y * z.to(u.pc) / length_of_vector,
        z=sun.cartesian.z - vector_z * z.to(u.pc) / length_of_vector,
        unit='pc',
        representation_type='cartesian')
    if verbose:
        print('ltt', ltt)
        print('sun  xyz', sun.cartesian.x, sun.cartesian.y, sun.cartesian.z)
        print('star xyz', star.cartesian.x, star.cartesian.y, star.cartesian.z)
        print('star xyz', star.cartesian.x.to(u.au), star.cartesian.y.to(u.au), star.cartesian.z.to(u.au))
        print('length_of_vector', length_of_vector)
        print('sgl', sgl.x.to(u.au), sgl.y.to(u.au), sgl.z.to(u.au))
    sgl.representation_type = 'spherical'
    return sgl


def sgl_position_scatter(RA, e_RA, DE, e_DE, distance, e_distance, time, z, orbit_axis=0*u.au, n_points=10):
    """Create SGL positions with scatter from uncertainty

    Parameters
    ----------
    RA : angle
        Position of star in RA (usually in degrees)
    e_RA: angle
        Uncertainty in the position in RA (usually in arcsec)
        Should include uncertainty from proper motion
    DE : angle
        Position of star in DE (usually in degrees)
    e_DE: angle
        Uncertainty in the position in DE (usually in arcsec)
        Should include uncertainty from proper motion
    time : astropy.time
        Time of the transmission (GMT)
    z : float
        Heliocentric distance
    distance : float
        Distance to the star (usually a few parsecs)
    e_dist: float
        Uncertainty in the distance (usually a few parsecs)
    orbit_axis: float
        Maximum separation between transmitter and host star (usually of order au)
    n_points: int
        Number of points to generate (default 10)

    Returns
    -------
    sgl_coords : list of astropy.coordinates
        Locations of the foci in the SGL at the times of transmissions with random scatter
    """

    sgl_coords = []
    for i in range(n_points):
        # Scatter from uncertainty in distance 
        scatter_distance = e_distance * np.random.normal()

        # Scatter from uncertainty in position
        scatter_RA = e_RA * np.random.normal()
        scatter_DE = e_DE * np.random.normal()

        # Scatter from distance between star and transmitter
        orbit_offset = (orbit_axis / distance)
        scatter_axis = ((orbit_offset * np.random.uniform(-1, 1)) * u.pc / u.au) * u.arcsec

        pos_receiver = sgl_position(
            RA=RA+scatter_RA+scatter_axis,
            DE=DE+scatter_DE+scatter_axis,
            distance=distance+scatter_distance,
            time=time,
            z=z,
            )
        sgl_coords.append(pos_receiver)
    return SkyCoord(sgl_coords)
