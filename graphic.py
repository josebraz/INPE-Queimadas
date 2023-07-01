import os
import math

from datetime import timezone, datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import pysal
import contextily
import xarray
import datashader

from geopy import distance
from geopy.point import Point

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely.ops import split

import contextily as cx

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns; sns.set()

from utils import *
from satelites import *

def show_separated_satelittes(data: pd.DataFrame, center_lat: float = None, 
                              center_lon: float = None, size: float = None, 
                              width: float = 20, axs: list[plt.Axes] = None):
    min_lat, max_lat, min_lon, max_lon = get_bounds(center_lat, center_lon, size)

    unique_satelites = data['satelite'].unique()
    n = len(unique_satelites)
    if n == 0: return

    if axs is None:
        max_col = min(4, n)
        rows = math.ceil(n / max_col)
        fig, axs = plt.subplots(rows, max_col, sharex= True, sharey=True, figsize=(width, (width/max_col) * rows))
        fig.tight_layout()
        axs = axs.flat

    for i, ax in enumerate(axs):
        if i >= n: break
        ax: plt.Axes
        configure_geografic_axes(ax, min_lon, max_lon, min_lat, max_lat)
        satelite = unique_satelites[i]
        data_satelite = data[data['satelite'] == satelite]
        satellites_data = SatellitesMeasureGeometry(data_satelite)
        dataframe = satellites_data.get_satelites_measures_area()
        color = satellites_colors[data_satelite.iloc[0]['simp_satelite']]
        dataframe.plot(
            ax=ax,
            color=color,
            alpha=0.5,
            legend=True,
            edgecolor=color,
            linewidth=1
        )
        ax.set_title(satelite)

def show_satelites_points(data: pd.DataFrame, ax: plt.Axes, markersize: float = 3):
    satelites_data = data['simp_satelite'].value_counts().where(lambda x : x != 0).dropna()
    for (i, satelite) in enumerate(satelites_data.index.tolist()):
        current = data[data['simp_satelite'] == satelite]
        gpd.GeoDataFrame(
            current,
            geometry=gpd.points_from_xy(current.longitude, current.latitude),
            crs="EPSG:4326"
        ).plot(
            ax=ax,
            color=satellites_colors[satelite],
            markersize=markersize,
            label="{} - {}".format(satelite, int(satelites_data[satelite]))
        )
    # ax.legend(markerscale=3)

def bar_limited(serie: pd.Series, ax: plt.Axes=plt.axes(), min_percent=0.02, title: str='', xlabel: str='', ylabel: str=''):
    pie_temp = serie.loc[lambda x : x > 0]
    total = pie_temp.sum()
    greatter = pie_temp.loc[lambda x : x/total >= min_percent].map(lambda x : x/total * 100)
    little = pie_temp.loc[lambda x : x/total < min_percent]
    print('outros:', little)
    pd.concat([greatter, pd.Series(data=[little.sum()/total * 100], index=['Outros'])]).plot.bar(
        y='Detecções',
        ax=ax,
        title=title
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def time_measure_dist(df: pd.DataFrame, satelite='AQUA_M-T', ax: plt.Axes=plt.axes()):
    filtered_df = df[(df["simp_satelite"] == satelite)]
    time_counts = filtered_df.groupby([filtered_df['datahora'].dt.time])['datahora'].count()
    time_counts.index = pd.TimedeltaIndex(data=time_counts.index.astype('str'))
    
    if '00:00:00' not in time_counts.index:
        time_counts['00:00:00'] = 0
    if '23:59:59' not in time_counts.index:
        time_counts['23:59:59'] = 0
    time_counts = time_counts.resample('30min').sum().reindex().rename(satelite)
    time_counts.plot(ax=ax, color=satellites_colors[satelite], legend=True)

def plot_colortable(colors, *, ncols=4) -> plt.Figure:
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

burn_colors = {
    -2: (0.0, 0.0, 1.0, 1.0),
    -1: (0.0, 0.0, 0.0, 1.0),
    0: (0.0, 0.0, 0.0, 0.0)
}
for i in range(1, 367):
    burn_colors[i] = (1.0, 0.0, 0.0, 1.0)

legend_handler = [
    patches.Patch(color=burn_colors[1], label='Queimado'),
    patches.Patch(color=burn_colors[-2], label='Água'),
    patches.Patch(color=burn_colors[-1], label='Sem dado'),
]

def show_nasa_burn_area(fp: str, **kwargs):
    ax = kwargs.get('ax')
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    ax.legend(handles=legend_handler)
    data = read_gdf_from_tiff(fp)
    color = data['value'].map(burn_colors)
    data.plot(color=color, ax=ax, linewidth=0, **kwargs)
    cx.add_basemap(ax, crs=data.crs.to_string())

fire_colors = {
    0: (0.0, 0.0, 0.0, 1.0), # 0 = missing input data
    1: (0.0, 0.0, 0.0, 1.0), # 1 = obsolete; not used since Collection 1
    2: (0.0, 0.0, 0.0, 1.0), # 2 = other reason
    3: (0.0, 0.0, 0.0, 0.0), # 3 =  non-fire water pixel
    4: (0.5, 0.5, 0.5, 1.0), # 4 =  cloud
    5: (0.0, 0.0, 0.0, 0.0), # 5 =  non-fire land pixel
    6: (1.0, 1.0, 1.0, 1.0), # 6 =  unknown (land or water)
    7: (0.5, 0.0, 0.0, 1.0), # 7 =  fire
    8: (0.75, 0.0, 0.0, 1.0), # 8 =  fire
    9: (1.0, 0.0, 0.0, 1.0), # 9 =  fire
}

fire_legend_handler = [
    patches.Patch(color=fire_colors[0], label='Sem dado'),
    patches.Patch(color=fire_colors[4], label='Núvem'),
    patches.Patch(color=fire_colors[6], label='Desconhecido'),
    patches.Patch(color=fire_colors[9], label='Fogo'),
]

def show_active_fire(fp: str, **kwargs):
    ax = kwargs.get('ax')
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    ax.legend(handles=fire_legend_handler)
    data = read_gdf_from_tiff(fp)
    color = data['value'].map(fire_colors)
    data.plot(color=color, ax=ax, linewidth=0, **kwargs)
    cx.add_basemap(ax, crs=data.crs.to_string())