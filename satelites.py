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

# resolution of all sensors in km
resolution_map = { 
    'VIIRS': 0.5,
    'AVHRR-3': 1.1,
    'MODIS': 1.0,
    'VIRS': 2.0,
    'AVHRR': 1.1,
    'ABI': 2.0,
    'SEVIRI': 3.0,
    'GOES I-M': 4.0
}

# cada satélite tem um sensor
satelite_sensors = {
    'Suomi NPP': 'VIIRS',
    'NOAA-20': 'VIIRS',
    'GOES-16': 'ABI',
    'GOES-13': 'GOES I-M',
    'GOES-12': 'GOES I-M',
    'GOES-10': 'GOES I-M',
    'GOES-08': 'GOES I-M',
    'AQUA': 'MODIS',
    'TERRA': 'MODIS',
    'NOAA-18': 'AVHRR-3',
    'NOAA-19': 'AVHRR-3',
    'NOAA-17': 'AVHRR-3',
    'NOAA-16': 'AVHRR-3',
    'NOAA-15': 'AVHRR-3',
    'NOAA-14': 'AVHRR',
    'NOAA-12': 'AVHRR',
    'MSG-03': 'SEVIRI', 
    'MSG-02': 'SEVIRI',
    'METOP-B': 'AVHRR-3', 
    'METOP-C': 'AVHRR-3'
}

satelite_map = {
    'NPP-375D': 'Suomi NPP',
    'NPP-375': 'Suomi NPP',
    'AQUA_M-T': 'AQUA',
    'AQUA_M-M': 'AQUA',
    'AQUA_M': 'AQUA',
    'TERRA_M-T': 'TERRA',
    'TERRA_M-M': 'TERRA',
    'TERRA_M': 'TERRA',
    'NOAA-18D': 'NOAA-18',
    'NOAA-19D': 'NOAA-19',
    'NOAA-16N': 'NOAA-16',
    'NOAA-15D': 'NOAA-15',
    'NOAA-12D': 'NOAA-12'
}

# (inclination, longitude, latitude and alture) float.nan means not valid
satellite_data = {
    'GOES-17': (0.0, -132.2, 0.0, 36000.0),
    'GOES-16': (0.0, -75.2, 0.0, 36000.0),
    'GOES-13': (0.0, -75.2, 0.0, 36000.0),
    'GOES-12': (0.0, -60.0, 0.0, 36000.0),
    'GOES-10': (0.0, -135.0, 0.0, 36000.0),
    'GOES-08': (0.0, -75.0, 0.0, 36000.0),
    'MSG-03': (0.0, 9.5, 0.0, 35000.0),
    'MSG-02': (0.0, 4.5, 0.0, 35000.0),
    'MSG-01': (0.0, 41.5, 0.0, 35000.0),
    'Suomi NPP': (1.72, float('nan'), float('nan'), 830),
    'NOAA-20': (1.7235108910151484, float('nan'), float('nan'), 834),
    'TERRA': (1.7121104003411214, float('nan'), float('nan'), 705),
    'AQUA': (1.715377656700855, float('nan'), float('nan'), 702),
    'NOAA-19': (1.7296998285427203, float('nan'), float('nan'), 855),
    'NOAA-18': (1.7263173804523555, float('nan'), float('nan'), 883),
    'NOAA-17': (1.7263173804523555, float('nan'), float('nan'), 883),
    'NOAA-16': (1.7263173804523555, float('nan'), float('nan'), 883),
    'NOAA-15': (1.7263173804523555, float('nan'), float('nan'), 883),
    'NOAA-14': (1.7263173804523555, float('nan'), float('nan'), 883),
    'NOAA-12': (1.7263173804523555, float('nan'), float('nan'), 883),
    'METOP-B': (1.7263173804523555, float('nan'), float('nan'), 883),
    'METOP-C': (1.7263173804523555, float('nan'), float('nan'), 883)
}

satellites_colors = {
    'AQUA': (1.0, 0.0, 0.0, 1.0),
    'GOES-13': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    'GOES-10': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0),
    'NOAA-17': (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    'TRMM': (1.0, 0.7333333333333333, 0.47058823529411764, 1.0),
    'NOAA-14': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    'ATSR': (0.596078431372549, 0.8745098039215686, 0.5411764705882353, 1.0),
    'METOP-B': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
    'NOAA-12': (1.0, 0.596078431372549, 0.5882352941176471, 1.0),
    'GOES-12': (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
    'NOAA-18': (0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1.0),
    'Suomi NPP': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    'MSG-02': (0.7686274509803922, 0.611764705882353, 0.5803921568627451, 1.0),
    'NOAA-19': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
    'GOES-08': (0.9686274509803922, 0.7137254901960784, 0.8235294117647058, 1.0),
    'METOP-C': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
    'NOAA-20': (0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1.0),
    'NOAA-16': (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
    'MSG-03': (0.8588235294117647, 0.8588235294117647, 0.5529411764705883, 1.0),
    'TERRA': (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
    'NOAA-15': (0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0),
    'GOES-16': (0.0, 1.0, 1.0, 1.0)
}

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
    orbital_cache = {}

    def __init__(self, data: pd.DataFrame, crs: str = 'EPSG:4326'):
        self.data = data
        self.crs = crs
        self.cache_areas: gpd.GeoDataFrame = None
        self.data_cache: dict[tuple[datetime, str], tuple] = dict()

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
    def get_orbital(satelite: str) -> Orbital:
        if satelite not in SatellitesMeasureGeometry.orbital_cache.keys():
            SatellitesMeasureGeometry.orbital_cache[satelite] = Orbital(satelite)
        return SatellitesMeasureGeometry.orbital_cache[satelite]

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
        # print(satellite_name, satellite_lon, satellite_lat, alt, 'distance =', (abs(satellite_lon-x), abs(satellite_lat-y)))
        simple_square = np.array([
            [left.longitude, top.latitude], # top left
            [right.longitude, top.latitude], # top rigth
            [right.longitude, bottom.latitude], # bottom rigth
            [left.longitude, bottom.latitude] # bottom left
        ])
        # print(real_x_distance, real_y_distance)
        return SatellitesMeasureGeometry._rotate_square(simple_square, inclination_x, inclination_y)

    def _get_satellite_data(self, time: pd.Timestamp, satellite_name: str) -> tuple[str, float, float, float, float]:
        """Returns satellite name, inclination, longitude, latitude and alture"""
        key = (time.to_pydatetime(), satellite_name)
        if key in self.data_cache:
            return self.data_cache[key]
        
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
        
        self.data_cache[key] = result
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