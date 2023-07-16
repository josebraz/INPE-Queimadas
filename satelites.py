import math

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd

from geopy import distance
from geopy.point import Point
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

from pyorbital.orbital import Orbital
from functools import lru_cache

from constants import *

def is_geostationary(satellite_name: str) -> bool:
    return not math.isnan(satellite_data.get(satellite_name)[1])

class AltAzimuthRange(object):
    default_lat = None
    default_long = None
    default_elv = None

    def __init__(self):
        if AltAzimuthRange.default_lat and AltAzimuthRange.default_long and AltAzimuthRange.default_elv:
            self.a = {'lat': AltAzimuthRange.default_lat, 'lon': AltAzimuthRange.default_long,
                      'elv': AltAzimuthRange.default_elv}
        else:
            self.a = None
        self.b = None
        pass

    def observer(self, lat: float, long: float, altitude: float):
        # latitude, longitude, meters above sea level (can be lower than zero)
        self.a = {'lat': lat, 'lon': long, 'elv': altitude}

    def target(self, lat: float, long: float, altitude: float):
        # latitude, longitude, meters above sea level (can be lower than zero)
        self.b = {'lat': lat, 'lon': long, 'elv': altitude}

    def calculate(self) -> dict:
        if not self.a:
            raise Exception(
                "Observer is not defined. Fix this by using instance_name.observer(lat,long,altitude) method")
        if not self.b:
            raise Exception(
                "Target location is not defined. Fix this by using instance_name.target(lat,long,altitude) method")
        ap, bp = AltAzimuthRange.LocationToPoint(self.a), AltAzimuthRange.LocationToPoint(self.b)
        br = AltAzimuthRange.RotateGlobe(self.b, self.a, bp['radius'])
        dist = round(AltAzimuthRange.Distance(ap, bp), 2)
        if br['z'] * br['z'] + br['y'] * br['y'] > 1.0e-6:
            theta = math.atan2(br['z'], br['y']) * 180.0 / math.pi
            azimuth = 90.0 - theta
            if azimuth < 0.0:
                azimuth += 360.0
            if azimuth > 360.0:
                azimuth -= 360.0
            azimuth = round(azimuth, 2)
            bma = AltAzimuthRange.NormalizeVectorDiff(bp, ap)
            if bma:
                elevation = 90.0 - (180.0 / math.pi) * math.acos(
                    bma['x'] * ap['nx'] + bma['y'] * ap['ny'] + bma['z'] * ap['nz'])
                elevation = round(elevation, 2)
            else:
                elevation = None
        else:
            azimuth = None
            elevation = None
        return {"azimuth": azimuth, "elevation": elevation, "distance": dist}

    @staticmethod
    def default_observer(lat: float, long: float, altitude: float):
        AltAzimuthRange.default_lat = lat
        AltAzimuthRange.default_long = long
        AltAzimuthRange.default_elv = altitude

    @staticmethod
    def Distance(ap, bp):
        dx = ap['x'] - bp['x']
        dy = ap['y'] - bp['y']
        dz = ap['z'] - bp['z']
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def GeocentricLatitude(lat):
        e2 = 0.00669437999014
        clat = math.atan((1.0 - e2) * math.tan(lat))
        return clat

    @staticmethod
    def EarthRadiusInMeters(latituderadians):
        a = 6378137.0
        b = 6356752.3
        cos = math.cos(latituderadians)
        sin = math.sin(latituderadians)
        t1 = a * a * cos
        t2 = b * b * sin
        t3 = a * cos
        t4 = b * sin
        return math.sqrt((t1 * t1 + t2 * t2) / (t3 * t3 + t4 * t4))

    @staticmethod
    def LocationToPoint(c):
        lat = c['lat'] * math.pi / 180.0
        lon = c['lon'] * math.pi / 180.0
        radius = AltAzimuthRange.EarthRadiusInMeters(lat)
        clat = AltAzimuthRange.GeocentricLatitude(lat)

        cos_lon = math.cos(lon)
        sin_lon = math.sin(lon)
        cos_lat = math.cos(clat)
        sin_lat = math.sin(clat)
        x = radius * cos_lon * cos_lat
        y = radius * sin_lon * cos_lat
        z = radius * sin_lat

        cos_glat = math.cos(lat)
        sin_glat = math.sin(lat)

        nx = cos_glat * cos_lon
        ny = cos_glat * sin_lon
        nz = sin_glat

        x += c['elv'] * nx
        y += c['elv'] * ny
        z += c['elv'] * nz

        return {'x': x, 'y': y, 'z': z, 'radius': radius, 'nx': nx, 'ny': ny, 'nz': nz}

    @staticmethod
    def NormalizeVectorDiff(b, a):
        dx = b['x'] - a['x']
        dy = b['y'] - a['y']
        dz = b['z'] - a['z']
        dist2 = dx * dx + dy * dy + dz * dz
        if dist2 == 0:
            return None
        dist = math.sqrt(dist2)
        return {'x': (dx / dist), 'y': (dy / dist), 'z': (dz / dist), 'radius': 1.0}

    @staticmethod
    def RotateGlobe(b, a, b_radius):
        br = {'lat': b['lat'], 'lon': (b['lon'] - a['lon']), 'elv': b['elv']}
        brp = AltAzimuthRange.LocationToPoint(br)

        alat = AltAzimuthRange.GeocentricLatitude(-a['lat'] * math.pi / 180.0)
        acos = math.cos(alat)
        asin = math.sin(alat)

        bx = (brp['x'] * acos) - (brp['z'] * asin)
        by = brp['y']
        bz = (brp['x'] * asin) + (brp['z'] * acos)

        return {'x': bx, 'y': by, 'z': bz, 'radius': b_radius}

class SatellitesMeasureGeometry:
    def __init__(self, data: pd.DataFrame, crs: str = 'EPSG:4326'):
        self.data = data
        self.crs = crs
        self.cache_areas: gpd.GeoDataFrame = None

    def get_satelites_measures_area(self) -> gpd.GeoDataFrame:
        """"Returns a GeoDataFrame contains only the name of the satellite and the geometry of measuase"""
        if self.cache_areas is None:
            squares = self._get_squares()
            polis = squares.map(lambda x: Polygon(x))
            simp_satellites = self.data['simp_satelite'].astype('str')
            satellites = self.data['satelite'].astype('str')
            self.cache_areas = gpd.GeoDataFrame(
                { 'simp_satelite': simp_satellites, 'satelite' : satellites }, 
                crs=self.crs,
                geometry=polis
            )
        return self.cache_areas 

    @staticmethod
    @lru_cache
    def get_orbital(satelite: str) -> Orbital:
        return Orbital(satelite, line1=tle_data[satelite][0], line2=tle_data[satelite][1])

    @staticmethod
    def _rotate_square(square: np.array, theta_x: float=0.0, theta_y: float=0.0) -> np.array:
        rotate = np.array([
            [np.cos(theta_x), -np.sin(theta_y)],
            [np.sin(theta_x),  np.cos(theta_y)]
        ])
        translate = square.mean(axis=0)
        out = square - translate
        out = (rotate @ out.T).T
        out = out + translate
        return out

    @staticmethod
    def _get_square(y: float, x: float, resolution: float, satellite_name: str, inclination: float,
                satellite_lon: float, satellite_lat: float, alt: float) -> np.array:
        need_adjust = alt >= 1000
        if need_adjust:
            satellite = AltAzimuthRange()
            satellite.observer(y, x, 0)
            satellite.target(satellite_lat, satellite_lon, alt * 1000)
            data = satellite.calculate()
            dist = data['distance']
            azimuth = data['azimuth']

            change_dist = ((dist / 1000) - alt) * 8 # fator de ajuste (validar)
            real_x_distance = alt + change_dist * math.cos(math.radians(azimuth % 90)) # in km
            real_y_distance = alt + change_dist * math.sin(math.radians(azimuth % 90)) # in km

            adjust_x = (real_x_distance * resolution/2.0) / alt
            adjust_y = (real_y_distance * resolution/2.0) / alt
            resolution_x, resolution_y = adjust_x * 2, adjust_y * 2
            inclination_x, inclination_y = 0.0, math.radians(5) #math.radians(satellite_lon - x), math.radians(satellite_lat - y) 
        else:
            resolution_x, resolution_y = resolution, resolution
            inclination_x, inclination_y = inclination, inclination
        geodesic_x = distance.geodesic(kilometers=resolution_x / 2.0)
        geodesic_y = distance.geodesic(kilometers=resolution_y / 2.0)

        top: Point = geodesic_y.destination((y, x), 0)
        bottom: Point = geodesic_y.destination((y, x), 180)
        right: Point = geodesic_x.destination((y, x), 90)
        left: Point = geodesic_x.destination((y, x), -90)
        simple_square = np.array([
            [left.longitude, top.latitude], # top left
            [right.longitude, top.latitude], # top rigth
            [right.longitude, bottom.latitude], # bottom rigth
            [left.longitude, bottom.latitude] # bottom left
        ])
        return SatellitesMeasureGeometry._rotate_square(simple_square, inclination_x, inclination_y)

    def _get_satellite_data(self, time: pd.Timestamp, satellite_name: str) -> tuple[str, float, float, float, float]:
        """Returns satellite name, inclination, longitude, latitude and alture"""
        if is_geostationary(satellite_name):
            return (satellite_name, *satellite_data[satellite_name])

        orb = SatellitesMeasureGeometry.get_orbital(satellite_name)
        loc1 = orb.get_lonlatalt(time)

        # the inclination is negative when the satellite is moving from top to bottom
        # of the earth. we get it comparing the latitude of two points close in the time
        loc2 = orb.get_lonlatalt(time + timedelta(seconds=5))
        descending = loc1[1] > loc2[1]
        inclination = orb.orbit_elements.inclination * (-1 if descending else 1)
        result = satellite_name, inclination, *loc1
        return result

    def _get_squares(self) -> pd.Series:
        if len(self.data) == 0: return pd.Series([])
        points = pd.DataFrame({
            'latitude': self.data['latitude'], 
            'longitude': self.data['longitude'], 
            'resolution': self.data['sensor'].map(resolution_map)
        })
        data_temp = self.data[['datahora', 'simp_satelite']].apply(
            lambda d: self._get_satellite_data(*d.values), axis=1, result_type='expand')
        points[['name', 'inclination', 'satellite_lon', 'satellite_lat', 'satellite_alt']] = data_temp
        return points.apply(lambda d: SatellitesMeasureGeometry._get_square(*d.values), axis=1)


def draw_orbit(range: pd.DatetimeIndex, satelite: str, title: str = "", color = None, ax: plt.Axes=plt.axes()):
    orb = SatellitesMeasureGeometry.get_orbital(satelite)
    data = pd.DataFrame(index=pd.to_datetime(range, utc=True))
    positions = data.index.map(orb.get_lonlatalt)
    data['longitude'] = positions.map(lambda x: x[0])
    data['latitude'] = positions.map(lambda x: x[1])
    if color is None:
        ax.plot(data['longitude'], data['latitude'], '-x', linewidth=1, markersize=5, label=satelite)
    else:
        ax.plot(data['longitude'], data['latitude'], '-x', linewidth=1, markersize=5, label=satelite, color=color)
        
    ax.legend(loc="lower left")
    ax.set_xlim([-85, -35])
    ax.set_ylim([-35, 5])
    ax.set_title(title)
    for index in [data.index[3], data.index[-3]]:
        ax.annotate(index.strftime("%H:%M"), (data['longitude'][index], data['latitude'][index]))