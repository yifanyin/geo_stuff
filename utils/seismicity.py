'''Functions used for seismicity analysis.'''
import pandas as pd
import pyproj
import shapely
import numpy as np
import scipy as sp
import copy


def get_seismicity_profile(
    catalog: pd.DataFrame,
    pt1: list | np.ndarray,
    pt2: list | np.ndarray,
    dist: float = 5000
    ) -> tuple[pd.DataFrame, list, list]:
    '''
    Given a line and a distance from the line, select the earthquakes within
    the distance from the line.

    Args:
        catalog: pd.DataFrame
            The earthquake catalog with 'latitude' and 'longitude' columns.
        pt1: list | np.ndarray
            The first point of the line [lon, lat].
        pt2: list | np.ndarray
            The second point of the line [lon, lat].
        dist: float
            The distance from the line in meters.
    Returns:
        selected_catalog: pd.DataFrame
            The selected earthquake catalog within the distance from the line.
            The returned catalog has additional 'x' and 'y' columns in
            projected coordinates.
        p1: list
            The first point of the line in projected coordinates [x, y].
        p2: list
            The second point of the line in projected coordinates [x, y].
    '''
    # Transfer them to equator
    xc = (catalog.longitude.max()+catalog.longitude.min())/2
    yc = (catalog.latitude.max()+catalog.latitude.min())/2
    utm_crs = pyproj.CRS("EPSG:4326")
    tmerc_crs = pyproj.CRS.from_proj4(
        f"+proj=tmerc +lat_0={yc} +lon_0={xc} +datum=WGS84 +units=m")

    # Transfer the seismicity, first
    ccopy = copy.deepcopy(catalog)
    transformer = pyproj.Transformer.from_crs(
        utm_crs, tmerc_crs, always_xy=True)
    ccopy['x'], ccopy['y'] = transformer.transform(
        catalog.longitude.values, catalog.latitude.values)
    
    p1 = [0, 0]
    p2 = [0, 0]
    p1[0], p1[1] = transformer.transform(pt1[0], pt1[1])
    p2[0], p2[1] = transformer.transform(pt2[0], pt2[1])

    line = shapely.geometry.LineString([p1, p2])
    buffer = line.buffer(dist)
    points = [shapely.geometry.Point(xy) for xy in zip(ccopy['x'], ccopy['y'])]
    mask = np.array([buffer.contains(point) for point in points])

    selected_catalog = ccopy[mask]
    return selected_catalog, p1, p2


def get_rotated_profile(
    selected_catalog: pd.DataFrame,
    pt1: list | np.ndarray,
    pt2: list | np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
    '''
    Rotate the selected catalog so that the profile is aligned with the
    x-axis. The profile is rotated counter-clockwise by the forward azimuth
    from pt1 to pt2.
    Args:
        selected_catalog: pd.DataFrame
            The selected earthquake catalog with 'x' and 'y' columns.
        pt1: list | np.ndarray
            The first point of the line [lon, lat].
        pt2: list | np.ndarray
            The second point of the line [lon, lat].
    Returns:
        rot: np.ndarray
            The rotated coordinates of the selected catalog.
        faz: float
            The forward azimuth from pt1 to pt2.
        baz: float
            The back azimuth from pt2 to pt1.
        dist: float
            The distance between pt1 and pt2.
    '''
    ggg = pyproj.Geod(ellps='WGS84')
    faz, baz, dist = ggg.inv(pt1[0], pt1[1], pt2[0], pt2[1])
    rrr = sp.spatial.transform.Rotation.from_euler('z', -faz, degrees=True)
    to_rotate = np.array([selected_catalog['x'].values,
                          selected_catalog['y'].values,
                          np.zeros(len(selected_catalog['x']))]
                         ).T
    rot = np.matmul(to_rotate, rrr.as_matrix())
    return rot, faz, baz, dist