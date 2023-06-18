import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd

import rasterio
import xarray
import datashader

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from geopy import distance
from shapely import Polygon, MultiPolygon, LineString
from shapely.ops import split
from shapely.geometry import box
from shapely.geometry import shape

from rasterio.features import shapes as rio_shapes

from constants import *

def read_burn_df() -> pd.DataFrame:
    if os.path.exists(data_parsed_file):
        df = pd.read_hdf(data_parsed_file, key='df')
    else:
        column_types = {
            'precipitacao': 'float32',
            'riscofogo': 'float32',
            'latitude': 'float64',
            'longitude': 'float64',
            'frp': 'float32'
        }
        # Create the DataFrame from csv
        df = dd.read_csv(
            os.path.join(burn_folder, "*.csv"), parse_dates=["datahora"], dtype=column_types
        )
        # Optimize data and setup types
        df['diasemchuva'] = df['diasemchuva'].fillna(invalid_value).astype("int16")
        df['riscofogo'] = df['riscofogo'].mask(df['riscofogo'] == invalid_value, 0)
        df['riscofogo'] = df['riscofogo'].fillna(0).astype("bool")
        df['satelite'] = df['satelite'].str.upper().astype("category")
        df['pais'] = df['pais'].str.upper().astype("category")
        df['estado'] = df['estado'].str.upper().astype("category")
        df['municipio'] = df['municipio'].str.upper().astype("category")
        df['bioma'] = df['bioma'].str.upper().astype("category")
        df['datahora'] = df['datahora'].dt.tz_localize(timezone.utc).dt.tz_convert(data_timezone)
        df['simp_satelite'] = df['satelite'].map(satelite_map).fillna(df['satelite']).astype("category")
        df['sensor'] = df['simp_satelite'].map(satelite_sensors).astype("category")
        df['regiao'] = df['estado'].map(estados_regioes).astype("category")
        df: pd.DataFrame = df.compute()
        df.to_hdf(data_parsed_file, key='df', mode='w', format="table")
    return df

def sub_space(data: pd.DataFrame, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> pd.DataFrame:
    """Returns a sub data filtered by lat and lon boundary"""
    return data.query('@min_lon <= longitude <= @max_lon & @min_lat <= latitude <= @max_lat')

def get_bounds(lat: float, lon: float, size: float) -> tuple[float, float, float, float]:
    if lat == None or lon == None or size == None:
        return (None, None, None, None)
    return (lat - size/2.0, lat + size/2.0, lon - size/2.0, lon + size/2.0)

def sub_space_by_center(data: pd.DataFrame, lat: float, lon: float, size: float) -> pd.DataFrame:
    """Returns a sub data filtered by lat and lon boundary"""
    min_lat, max_lat, min_lon, max_lon = get_bounds(lat, lon, size)
    return data.query('@min_lon <= longitude <= @max_lon & @min_lat <= latitude <= @max_lat')

def get_landsat_geometry(path: int, row: int) -> Polygon:
    if not hasattr(get_landsat_geometry, "wrs2"):
        # reference: https://www.usgs.gov/landsat-missions/landsat-shapefiles-and-kml-files
        get_landsat_geometry.wrs2: gpd.GeoDataFrame = gpd.read_file('tiff/WRS2_descending_0')
    return get_landsat_geometry.wrs2.query('PATH == @path & ROW == @row').iloc[0].geometry

def sub_space_by_landsat(df: pd.DataFrame, path: int, row: int) -> pd.DataFrame:
    temp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    geometry = get_landsat_geometry(path, row)
    return df.loc[temp.intersects(geometry).values]

from typing import Iterator

def split_by_range(data: pd.DataFrame, range: pd.DatetimeIndex, time_column='datahora') -> Iterator[tuple[pd.DataFrame, pd.Timestamp]]:
    """Returns a list of dataframe with time_column grouped for each value of range"""
    if len(range) <= 1:
        return
    for (i, current) in enumerate(range):
        if i < len(range) - 1:
            next = range[i+1]
            yield (data.query(f'@current <= {time_column} < @next'), current)
            # yield (data[(data[time_column] >= current) & (data[time_column] < next)], current)

def grid_to_dataframe(grid: xarray.DataArray) -> pd.DataFrame:
    data = grid.to_pandas().unstack()
    frame =  data.where(data > 0).dropna().reset_index().rename(columns={0: "value"}) 
    return frame

from matplotlib_scalebar.scalebar import ScaleBar

def configure_geografic_axes(ax: plt.Axes, min_lon: float, max_lon: float, 
                             min_lat: float, max_lat: float, with_scale: bool = True):
    if min_lat != None and max_lat != None and min_lon != None and max_lon != None:
        if with_scale:
            width = distance.distance((max_lat, min_lon), (max_lat, max_lon)).m
            ax.add_artist(ScaleBar(width / abs(max_lon - min_lon), units='m'))
        x_lim = [min_lon, max_lon]
        y_lim = [min_lat, max_lat]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(4))
    ax.xaxis.set_major_formatter(lambda x, pos: '{:.2f}{}'.format(abs(x), 'W' if x < 0 else 'E'))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
    ax.yaxis.set_major_formatter(lambda y, pos: '{:.2f}{}'.format(abs(y), 'S' if y < 0 else 'N'))
    # x_size = abs(max_lon - min_lon)
    # y_size = abs(max_lat - min_lat)
    # dem = Grid(nxny=(4, 4), dxdy=(x_size/4, y_size/4), x0y0=(max_lat, max_lon))
    # print(dem.ll_coordinates)
    # # g = GoogleVisibleMap(x=x_lim, y=y_lim, crs=dem, maptype='satellite')
    # # ggl_img = g.get_vardata()
    # sm = Map(dem, factor=1, countries=False)
    # sm.visualize(ax=ax)

def compute_grid(data: pd.DataFrame, min_lat: float = None, max_lat: float = None, 
                 min_lon: float = None, max_lon: float = None, 
                 aggr_dist = distance.Distance(kilometers=1)) -> xarray.DataArray:
    if not min_lat:
        min_lat = data['latitude'].min()
    if not max_lat:
        max_lat = data['latitude'].max()
    if not min_lon:
        min_lon = data['longitude'].min()
    if not max_lon:
        max_lon = data['longitude'].max()

     # calculate the grid (agregate)
    width_bottom = distance.distance((min_lat, min_lon), (min_lat, max_lon))
    width_top = distance.distance((max_lat, min_lon), (max_lat, max_lon))

    height_left = distance.distance((min_lat, min_lon), (max_lat, min_lon))
    height_right = distance.distance((min_lat, max_lon), (max_lat, max_lon))

    width = max(width_top, width_bottom)
    height = max(height_left, height_right)

    cvs = datashader.Canvas(
        plot_width=int(width/aggr_dist), 
        plot_height=int(height/aggr_dist),
        x_range=(min_lon, max_lon),
        y_range=(min_lat, max_lat)
    )
    return cvs.points(data, x="longitude", y="latitude")

def grid_gdf(data: gpd.GeoDataFrame, poly: Polygon = None, quadrat_width: float=0.005) -> gpd.GeoDataFrame:
    if poly == None:
        bounds = data.total_bounds
    else:
        bounds = poly.bounds
    step = 1/quadrat_width
    xmin, ymin, xmax, ymax = [int(x * step) / step for x in bounds]
    grid_cells = []
    
    for x0 in np.arange(xmin, xmax+quadrat_width, quadrat_width):
        for y0 in np.arange(ymin, ymax+quadrat_width, quadrat_width):
            x1 = x0-quadrat_width
            y1 = y0+quadrat_width
            shape = box(x0, y0, x1, y1)
            grid_cells.append(shape)
    temp = gpd.GeoDataFrame(geometry=grid_cells, crs=data.crs)
    if poly is None:
        return temp
    else:
        return temp[temp.intersects(poly).values].reset_index()

def normalize_gdf(data: gpd.GeoDataFrame, bounds: Polygon = None, quadrat_width: float=0.005) -> gpd.GeoDataFrame:
    if bounds != None: 
        xmin, ymin, xmax, ymax = bounds.bounds
        data = data.cx[xmin:xmax, ymin:ymax]
    grid_df = grid_gdf(data, bounds, quadrat_width)
    join_dataframe = gpd.sjoin(data, grid_df, predicate="intersects")
    
    values = np.zeros(len(grid_df))
    for index in join_dataframe['index_right'].unique():
        polygon = grid_df.iloc[index].geometry
        matches = join_dataframe[join_dataframe['index_right'] == index]
        intersection_areas = np.array([polygon.intersection(x.buffer(0)).area for x in matches.geometry])
        values[index] = intersection_areas.sum() / polygon.area if len(intersection_areas) > 0 else 0
    return gpd.GeoDataFrame({ 'value': values }, geometry=grid_df.geometry, crs=data.crs)

def read_gdf_from_tiff(data_file: str, name: str = 'value') -> gpd.GeoDataFrame:
    with rasterio.open(data_file) as src:
        data = src.read(1)
        data = data.astype('int16')
        shapes = rio_shapes(data, transform=src.transform)
        crs = src.crs.to_string()

    values = []
    geometry = []
    for shapedict, value in shapes:
        values.append(int(value))
        geometry.append(shape(shapedict))

    return gpd.GeoDataFrame(
        { name: values, 'geometry': geometry }, 
        crs = crs)

flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]
