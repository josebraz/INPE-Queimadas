import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd

import rasterio
import xarray as xr
import datashader
import shapely
import dask_geopandas

import os
from datetime import timezone, datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from geopy import distance
from shapely import Polygon, MultiPolygon, LineString
from shapely.ops import split
from shapely.geometry import box
from shapely.geometry import shape
from pyproj import Geod

from rasterio.features import shapes as rio_shapes
from functools import lru_cache

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
        # reference: http://www.dgi.inpe.br/documentacao/grades em Grade TM da America do Sul no formato (Shape File)
        get_landsat_geometry.wrs2: gpd.GeoDataFrame = gpd.read_file('aux/grade_tm_am_do_sul')
    return get_landsat_geometry.wrs2.query('ORBITA == @path & PONTO == @row').iloc[0].geometry

# def get_landsat_geometry(path: int, row: int) -> Polygon:
#     if not hasattr(get_landsat_geometry, "wrs2"):
#         # reference: https://www.usgs.gov/landsat-missions/landsat-shapefiles-and-kml-files
#         get_landsat_geometry.wrs2: gpd.GeoDataFrame = gpd.read_file('aux/WRS2_descending_0')
#     return get_landsat_geometry.wrs2.query('PATH == @path & ROW == @row').iloc[0].geometry

def sub_space_by_landsat(df: pd.DataFrame, path: int, row: int) -> pd.DataFrame:
    geometry = get_landsat_geometry(path, row)
    xmin, ymin, xmax, ymax = geometry.bounds
    df = sub_space(df, ymin, ymax, xmin, xmax)
    temp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    return df.loc[temp.intersects(geometry).values]

def sub_space_by_landsat_geo(df: gpd.GeoDataFrame, path: int, row: int) -> gpd.GeoDataFrame:
    geometry = get_landsat_geometry(path, row)
    xmin, ymin, xmax, ymax = geometry.bounds
    df = df.cx[xmin:xmax, ymin:ymax]
    return df.loc[df.intersects(geometry).values]

from typing import Iterator

def split_by_range_index(range: pd.DatetimeIndex) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Returns a list of dataframe with time_column grouped for each value of range"""
    if len(range) <= 1:
        return
    for (i, current) in enumerate(range):
        if i < len(range) - 1:
            next = range[i+1]
            yield (current, next)

def split_by_range(data: pd.DataFrame, range: pd.DatetimeIndex, time_column='datahora') -> Iterator[tuple[pd.DataFrame, pd.Timestamp]]:
    """Returns a list of dataframe with time_column grouped for each value of range"""
    for (current, next) in split_by_range_index(range):
        yield (data.query(f'@current <= {time_column} < @next'), current)

def grid_to_dataframe(grid: xr.DataArray) -> pd.DataFrame:
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
                 aggr_dist = distance.Distance(kilometers=1)) -> xr.DataArray:
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

_grid_cache: tuple[tuple, gpd.GeoDataFrame] = None

def grid_gdf(data: gpd.GeoDataFrame, poly: Polygon = None, quadrat_width: float=0.002) -> gpd.GeoDataFrame:
    global _grid_cache
    bounds = data.total_bounds if poly == None else poly.bounds
    step = 1/quadrat_width
    xmin, ymin, xmax, ymax = [int(x * step) / step for x in bounds]
    key = (xmin, ymin, xmax, ymax, quadrat_width)
    cache = _grid_cache
    temp = None if cache is None or cache[0] != key or poly == None else cache[1]
    from_cache = temp is not None
    if temp is None:
        xs = np.arange(xmin, xmax+quadrat_width, quadrat_width)
        ys = np.arange(ymin, ymax+quadrat_width, quadrat_width)
        xss, yss = np.meshgrid(xs, ys, copy=False)
        fv = np.vectorize(lambda x0, y0: box(x0, y0, x0-quadrat_width, y0+quadrat_width))
        grid_cells = fv(xss.flatten("F"), yss.flatten("F"))
        temp = gpd.GeoDataFrame(geometry=grid_cells, crs=data.crs)
        if poly != None:
            temp = temp[temp.intersects(poly).values].reset_index()
            temp.drop('index', axis=1, inplace=True)
        _grid_cache = (key, temp)
    if poly is None or from_cache:
        return temp
    else:
        temp = temp[temp.intersects(poly).values].reset_index()
        temp.drop('index', axis=1, inplace=True)
        return temp

def normalize_gdf(data: gpd.GeoDataFrame, bounds: Polygon = None, 
                  quadrat_width: float=0.005, column: str = None) -> gpd.GeoDataFrame:
    if bounds != None: 
        xmin, ymin, xmax, ymax = bounds.bounds
        data = data.cx[xmin:xmax, ymin:ymax]
    if len(data) == 0: return data
    grid_df = grid_gdf(data, bounds, quadrat_width)
    if column != None: # optimize
        data = data[data[column] > 0]

    dask_gdf = dask_geopandas.GeoDataFrame = dask_geopandas.from_geopandas(data, npartitions=16)
    join_dataframe = dask_gdf.sjoin(grid_df, predicate="intersects").compute()

    values = np.zeros(len(grid_df))
    for index in join_dataframe['index_right'].unique():
        polygon: Polygon = grid_df.iloc[index].geometry
        matches: gpd.GeoDataFrame = join_dataframe[join_dataframe['index_right'] == index]
        matches['geometry'] = matches['geometry'].buffer(0)
        intersection_polys = matches.intersection(polygon)
        if len(intersection_polys) == 0:
            intersection_area = 0
        elif column != None:
            intersection_area = (intersection_polys.area * matches[column]).sum()
        else:
            intersection_area = shapely.union_all(intersection_polys).area
        values[index] = intersection_area / polygon.area
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

def create_dataarray(data: gpd.GeoDataFrame, value_column: str = 'value', sparse: bool = True) -> xr.DataArray:
    if len(data) == 0: return xr.DataArray()
    data.loc[data['geometry'].duplicated(), value_column] = 1
    data = data.drop_duplicates(subset = 'geometry')
    temp = data[data[value_column] > 0.0]
    points = temp.representative_point()
    idx = pd.MultiIndex.from_arrays(arrays=[points.y, points.x], names=["y", "x"])
    s = pd.Series(temp[value_column].values, index=idx)
    array = xr.DataArray.from_series(s, sparse=sparse)
    if not sparse:
        array = array.fillna(0.0)
    return array

def create_gpd(data: xr.DataArray, value_dim: str = 'value', poly: Polygon = None, quadrat_width: float = 0.005) -> gpd.GeoDataFrame:
    frame = data.to_dataframe(name=value_dim)
    frame = frame[frame['value'] > 0]
    points = gpd.GeoDataFrame(
        { 'value' : frame['value'] }, 
        geometry=gpd.points_from_xy(x=frame['x'], y=frame['y']),
        crs="EPSG:4326")
    grid = grid_gdf(points, poly=poly, quadrat_width=quadrat_width).copy()
    grid.drop('index', axis=1, inplace=True) # todo remove this
    join_dataframe: gpd.GeoDataFrame = gpd.sjoin(grid, points, op="contains")
    values = np.zeros(len(grid))
    for index in join_dataframe['index_right'].unique():
        values[index] = points.iloc[index]['value']
    grid['value'] = values
    return grid

def evaluate_gpd(reference: gpd.GeoDataFrame, other: gpd.GeoDataFrame,
                 reference_value_column: str = 'value', 
                 other_value_column: str = 'value'):
    """
    | other \ reference | queimada | não queimada |
    |-------------------|----------|--------------|
    |     queimada      |    TP    |      FP      |
    |   não queimada    |    FN    |      TN      |
    """
    
    original_geometry = other['geometry']
    other['geometry'] = other.representative_point()
    join_gpd = gpd.sjoin(reference, other, op="contains", lsuffix='reference', rsuffix='other')
    other['geometry'] = original_geometry
    
    same_names = reference_value_column == other_value_column
    burned_reference = np.array(join_gpd[reference_value_column + ('_reference' if same_names else '')])
    burned_other = np.array(join_gpd[other_value_column + ('_other' if same_names else '')])
    unburned_reference = np.array(1.0 - join_gpd[reference_value_column + ('_reference' if same_names else '')])
    unburned_other = np.array(1.0 - join_gpd[other_value_column + ('_other' if same_names else '')])

    def calculate_same(array1, array2):
        min_array = np.array([array1, array2]).min(axis=0)
        min_array[np.isnan(min_array)] = 0.0
        return min_array.sum()
    
    def calculate_diff(array1, array2):
        temp1 = array1[array1 > array2]
        temp2 = array2[array1 > array2]
        diff = (temp1 - temp2)
        return diff.sum()

    TP = calculate_same(burned_reference, burned_other) # True Positive
    FN = calculate_diff(burned_reference, burned_other) # False Negative
    FP = calculate_diff(burned_other, burned_reference) # False Positive
    TN = calculate_same(unburned_reference, unburned_other) # True Negative

    ACC = (TP + TN) / (TP + FP + FN + TN)
    TPR = 0 if TP + FN == 0 else TP / (TP + FN) # true positive rate
    TNR = 0 if TN + FP == 0 else TN / (TN + FP) # true negative rate
    PPV = 0 if TP + FP == 0 else TP / (TP + FP) # positive predictive value 
    NPV = 0 if TN + FN == 0 else TN / (TN + FN) # negative predictive value
    OE = 0 if FN + TP == 0 else FN / (FN + TP) 
    CE = 0 if FP + TP == 0 else FP / (FP + TP)
    B  = (TP + FP) / (TP + FN) # viés
    DC = 2 * TP / (2 * TP + FP + FN)
    CSI = TP / (TP + FP + FN)

    return { 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
             'ACC': ACC, 'CE': CE, 'OE': OE, 'B': B, 
             'DC': DC, 'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 
             'NPV': NPV, 'CSI': CSI }

def get_burned_area_km2(gdf: gpd.GeoDataFrame, column: str = 'value') -> float:
    geod = Geod(ellps="WGS84")
    positive_normalized = gdf[gdf[column] > 0]
    def perimeter(geo):
        try:
            return abs(geod.geometry_area_perimeter(geo)[0])
        except:
            return 0.0
    area = positive_normalized['geometry'].map(perimeter)
    return (area * positive_normalized[column]).sum() / 1000000

def get_year_date_pairs(year: int) -> list[tuple[str, str]]:
    date_range = pd.date_range(f'{year}-01-01', periods=12, freq='M').insert(0, f'{year}-01-01')
    return [((start + timedelta(days=0 if i == 0 else 1)).strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')) 
              for i, (start, end) in enumerate(split_by_range_index(date_range))]

def to_pretty_table_latex(dt: pd.DataFrame, columns: list[str], sort: list[str]) -> pd.Series:
    def print_item(column: str, item: any):
        if column == 'orbita_ponto':
            return "\\multirow{2}{*}{" + item + "}"
        elif column in ['start_dt', 'end_dt']:
            return datetime.strptime(item.split(' ')[0], '%Y-%m-%d').strftime('%d/%m/%Y')
        elif column in ['reference_area_km2', 'model_area_km2']:
            return "${:.1f}Km^2$".format(item)
        elif column in ["B", "CSI"]:
            return "{:.2f}".format(item)
        elif isinstance(item, float):
            return "{:.2f}\\%".format(int(item * 10000) / 100)
        else:
            return str(item)

    def print_row(row):
        values = [print_item(columns[i], item) for i, item in enumerate(row)]
        values.insert(1, "")
        top = values[0::2]
        bottom = values[1::2]
        return " & ".join(top) + " \\\\\n" + " " * 24 + " & ".join(bottom) + " \\\\\n\\hline"
    
    return dt.sort_values(sort, ascending=False).loc[:, columns].apply(lambda row: print_row(row.values), axis=1)

def read_file_normalized(gdf_normal: gpd.GeoDataFrame, file: str, 
                         region: Polygon=None, quadrat_width: float=0.005,
                         column: str = None) -> gpd.GeoDataFrame:
    norm_name = os.path.join(cache_folder, 'norm', f"""{quadrat_width}{'_' + column if column != None else ''}_{os.path.basename(os.path.splitext(file)[0])}""")
    os.makedirs(os.path.dirname(norm_name), exist_ok=True)
    if os.path.exists(norm_name):
        gdf_normalized = gpd.read_file(norm_name, engine="pyogrio" if region == None else "fiona", mask=region)
    else:
        gdf_normalized = normalize_gdf(gdf_normal, region, quadrat_width, column=column)
        gdf_normalized.to_file(norm_name, engine="pyogrio")
    return gdf_normalized

def read_file_normalized_cached(path, row, file, quadrat_width) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    region = get_landsat_geometry(path, row)
    gdf_normal = gpd.read_file(file, engine="pyogrio" if region == None else "fiona", mask=region)
    gdf_normalized = read_file_normalized(gdf_normal, region, file, quadrat_width)
    return (gdf_normal, gdf_normalized)

def get_infos_from_aq30m(file: str, hour: str = "14:00:00-03:00") -> tuple[int, int, str, str]:
    path, row, end_date = os.path.basename(file).split('_')[2:5]
    path, row, end_date = int(path), int(row), datetime.strptime(end_date, '%Y%m%d')
    start = (end_date - timedelta(days=16)).strftime('%Y-%m-%d') + f' {hour}'
    end = end_date.strftime('%Y-%m-%d') + f' {hour}'
    return path, row, start, end

import calendar

def get_infos_from_aq1km(file: str) -> tuple[str, str]:
    year, month, day = os.path.basename(file).split('_')[0:3]
    year, month, day = int(year), int(month), int(day)
    start_date = datetime.strptime(f'{year}{month}{day}', '%Y%m%d')
    last_day = calendar.monthrange(year, month)[1]
    end = (start_date + timedelta(days=last_day-1)).strftime('%Y-%m-%d') + ' 23:59:59-03:00'
    start = start_date.strftime('%Y-%m-%d') + ' 00:00:00-03:00'
    print(f"{file} {start} {end}")
    return start, end

import glob

def read_geopandas(files_regex: str, region: Polygon=None) -> gpd.GeoDataFrame:
    cache_file = os.path.join(cache_folder, 'join', os.path.basename(os.path.splitext(files_regex)[0]))
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        cached = gpd.read_file(cache_file, engine="pyogrio" if region == None else "fiona", mask=region)
    else:
        files = glob.glob(files_regex)
        gdfs = [gpd.read_file(file, engine="pyogrio" if region == None else "fiona", mask=region) 
                for file in files if os.path.isfile(file) or len(os.listdir(file)) > 0]
        if len(gdfs) == 0:
            raise ValueError("Read list empty")
        cached = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        cached.to_file(cache_file, engine="pyogrio")
    return cached

def get_quadrat_width(distance: distance.Distance) -> float:
    return distance.destination((0,0), bearing=0).latitude