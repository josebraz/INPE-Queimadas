import numpy as np
import pandas as pd
import geopandas as gpd
import contextily

from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from utils import *
from satelites import *
from graphic import *

from pyproj import Geod
from typing import Callable

default_threshold: float = 5

BurnedAreaCalc = Callable[[float, np.ndarray], float]

def threshold_burned_area_calc(threshold: float = default_threshold) -> BurnedAreaCalc:
    return lambda value, values: 1.0 if value >= threshold else 0.0

def linear_burned_area_calc(min_range: float, max_range: float) -> BurnedAreaCalc:
    value_range = (max_range - min_range)
    return lambda value, values: min(1, max(0, (value - min_range) / value_range))

def polinomial_burned_area_calc(min_range: float, max_range: float, exponent: int) -> BurnedAreaCalc:
    value_range = (max_range - min_range)
    return lambda value, values: min(1, max(0, ((value - min_range) / value_range))**exponent)

def get_default_cmap():
    colors_lst = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
    return colors.LinearSegmentedColormap.from_list('mycmap', colors_lst)

class BurnedAreaCalcPercentile:

    def __init__(self, pmin: float=65, pmax: float=85, vmin: float=2.5, vmax: float=15, exponent=1):
        self.pmin = pmin
        self.pmax = pmax
        self.vmin = vmin
        self.vmax = vmax
        self.exponent = exponent
        self.value_range = None
        self.min_range = None
        self.max_range = None

    def __call__(self, value: float, values: np.ndarray) -> float:
        if self.value_range is None:
            non_zero_values = values[values > 0]
            mean = (self.vmin + self.vmax) / 2.0
            self.min_range = min(mean, max(self.vmin, np.percentile(non_zero_values, self.pmin)))
            self.max_range = max(mean, min(self.vmax, np.percentile(non_zero_values, self.pmax)))
            self.value_range = (self.max_range - self.min_range)
        return min(1, max(0, ((value - self.min_range) / self.value_range))**self.exponent)
    
    def __str__(self) -> str:
        return f'''min_range: {self.min_range} max_range: {self.max_range}'''

class SatellitesExplore:

    default_burned_area_calc = BurnedAreaCalcPercentile()
    default_cmap = get_default_cmap()
    default_min_area_percentage = 0.2
    default_threshold_satellite = 3

    def __init__(self, 
                 data: pd.DataFrame, 
                 delimited_region: Polygon = None,
                 quadrat_width: float = 0.005, 
                 min_area_percentage: float = default_min_area_percentage,
                 threshold_satellite: float = default_threshold_satellite,
                 burned_area_calc: BurnedAreaCalc = default_burned_area_calc):
        self.data = data
        self.delimited_region = delimited_region
        self.satellites_data = SatellitesMeasureGeometry(data)
        self.geod = Geod(ellps="WGS84")
        self.dataframe = self.satellites_data.get_satelites_measures_area()
        self.data_color = self.dataframe['simp_satelite'].map(satellites_colors)
        self.quadrat_width = quadrat_width
        self.burned_area_calc = burned_area_calc
        self.min_area_percentage = min_area_percentage
        self.threshold_satellite = threshold_satellite

        self.all_evaluated_quads: gpd.GeoDataFrame = None

    def plot(self, width: int = 20, center_lat: float = None, center_lon: float = None, 
             size: float = None, only_quads_evaluated: bool = False, only_quads_areas: bool = False, 
             fig: plt.Figure = None, axs: list[plt.Axes] = None, cmap = default_cmap,
             with_color_bar: bool = True, with_base_map: bool = False, 
             simple_plot: bool = False) -> tuple[plt.Figure, list[plt.Axes]]:
        with_color_bar = with_color_bar and not simple_plot
        if fig == None:
            n = 1 if only_quads_evaluated else 4 
            if with_color_bar:
                height = (width - width * 0.15) / n # color bar space 
            else:
                height = width / n
            tight_plot = not simple_plot
            fig, axs = plt.subplots(1, n, sharey=tight_plot, sharex=tight_plot, figsize=(width, height))
            if tight_plot:
                fig.tight_layout(pad=0.0)
            axs = axs.flat
        
        if only_quads_evaluated: 
            self.show_satellites_quads_evaluated(axs[0], with_color_bar=with_color_bar, 
                                                 cmap=cmap, evaluated_quads=False)
        elif only_quads_areas:
            self.show_satellites_quads_areas(axs[0], with_color_bar=with_color_bar, cmap=cmap)
        else:
            self.show_satellites_points(axs[0], with_base_map=with_base_map)
            self.show_satellites_areas(axs[1], with_base_map=with_base_map)
            self.show_satellites_quads_evaluated(axs[2], with_color_bar=with_color_bar, cmap=cmap)
            self.show_satellites_quads_areas(axs[3], with_color_bar=with_color_bar, cmap=cmap, with_areas=not simple_plot)
        
        min_lat, max_lat, min_lon, max_lon = get_bounds(center_lat, center_lon, size)
        for ax in axs:
            configure_geografic_axes(ax, min_lon, max_lon, min_lat, max_lat, with_scale=not simple_plot)
            if simple_plot:
                ax.set_axis_off()
                ax.set_facecolor('white')
        return fig, axs


    def show_satellites_points(self, ax: plt.Axes, with_base_map: bool = False, markersize: float = 3):
        if len(self.dataframe) == 0: return
        ax.scatter(self.data['longitude'], self.data['latitude'], c=self.data_color, s=markersize)
        if with_base_map:
            contextily.add_basemap(ax, crs=self.dataframe.crs)

    def show_satellites_areas(self, ax: plt.Axes, with_base_map: bool = False):
        if len(self.dataframe) == 0: return
        self.dataframe.plot(ax=ax, color=self.data_color, categorical=True, alpha=0.2,
            legend=True, edgecolor=self.data_color, linewidth=1)
        if with_base_map:
            contextily.add_basemap(ax, crs=self.dataframe.crs)
    
    def show_satellites_quads_evaluated(self, ax: plt.Axes, with_color_bar: bool = True, 
                                        cmap = default_cmap, evaluated_quads: bool = True,
                                        linewidth=0.5):
        if len(self.dataframe) == 0: return
        points: gpd.GeoDataFrame = self.get_all_evaluated_quads()[self.get_all_evaluated_quads()['value'] > 0.0]
        if evaluated_quads:
            points.plot(ax=ax, edgecolor='k', alpha=0.5, 
                        linewidth=linewidth, column='value', cmap='hot', 
                        legend=with_color_bar)
        else:
            points.plot(ax=ax, edgecolor='k', linewidth=linewidth)

    def show_satellites_quads_areas(self, ax: plt.Axes, with_color_bar: bool = True, 
                                    with_areas: bool = True, cmap = default_cmap, 
                                    linewidth=0.5):
        if len(self.dataframe) == 0: return
        area_km2 = self.get_total_area_m_2() / 1000000
        if with_areas:
            ax.legend(title = "{:.2f}Km²".format(area_km2), loc='lower left')
        points: gpd.GeoDataFrame = self.get_all_evaluated_quads()[self.get_all_evaluated_quads()['value'] > 0.0]
        points.plot(column='burned_fac', cmap=cmap, legend=with_color_bar,
            ax=ax, edgecolor='k', linewidth=linewidth)
    
    def get_total_area_m_2(self) -> float:
        points: gpd.GeoDataFrame = self.get_burned_areas()
        return points['burned_are'].sum()
    
    def get_burned_areas(self) -> gpd.GeoDataFrame:
        points: gpd.GeoDataFrame = self.get_all_evaluated_quads()
        if 'burned_are' not in points.columns:
            values = points['value'].values
            points['burned_fac'] = points['value'].map(lambda value: self.burned_area_calc(value, values))
            points['area'] = points['geometry'].map(lambda geo: abs(self.geod.geometry_area_perimeter(geo)[0]))
            points['burned_are'] = points['area'] * points['burned_fac']
        return points

    def recalcule_burned_area(self, burned_area_calc: BurnedAreaCalc) -> pd.Series:
        self.burned_area_calc = burned_area_calc
        points: gpd.GeoDataFrame = self.get_all_evaluated_quads()
        return SatellitesExplore.recalcule_burned_area_static(burned_area_calc, points, self.geod)

    @staticmethod
    def recalcule_burned_area_static(burned_area_calc: BurnedAreaCalc, points: gpd.GeoDataFrame, 
                                     geod: Geod = Geod(ellps="WGS84")) -> pd.Series:
        if 'area' not in points.columns:
            points['area'] = points['geometry'].map(lambda geo: abs(geod.geometry_area_perimeter(geo)[0]))
        values = points['value'].values
        points['burned_fac'] = points['value'].map(lambda value: burned_area_calc(value, values))
        points['burned_are'] = points['area'] * points['burned_fac']
        return points['burned_are']

    def get_all_evaluated_quads(self) -> gpd.GeoDataFrame:
        if self.all_evaluated_quads is None:
            quads_df = grid_gdf(self.dataframe, poly=self.delimited_region, quadrat_width=self.quadrat_width)
            join_dataframe = gpd.sjoin(self.dataframe, quads_df, op="intersects")

            values = np.zeros(len(quads_df))
            for index in join_dataframe['index_right'].unique():
                values[index] = self._evaluate_quads(index, quads_df.iloc[index].geometry, join_dataframe)

            self.all_evaluated_quads = gpd.GeoDataFrame(
                { 'value': values, 'geometry': quads_df['geometry'] },
                crs = quads_df.crs
            )

        return self.all_evaluated_quads
    
    def _evaluate_quads(self, index: int, polygon: Polygon, join_dataframe: gpd.GeoDataFrame) -> float:
        """
        Evaluate the polygon with some criterias:
            
        """
        polygon_area = polygon.area
        min_area = polygon_area * self.min_area_percentage

        precise_matches: gpd.GeoDataFrame = join_dataframe[join_dataframe['index_right'] == index]
        intersection_areas = np.array([polygon.intersection(x).area for x in precise_matches.geometry])
        filtered_matches: gpd.GeoDataFrame = precise_matches[intersection_areas > min_area]

        uniques_satellites = filtered_matches['satelite'].unique()
        intersection_areas_per_poly = intersection_areas.sum() / polygon_area
        penality = 1 - min(1, len(uniques_satellites) / self.threshold_satellite)

        return (len(uniques_satellites) ** 2) + intersection_areas_per_poly - intersection_areas_per_poly * penality
