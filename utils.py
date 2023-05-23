import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import datashader

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from geopy import distance

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


flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]
