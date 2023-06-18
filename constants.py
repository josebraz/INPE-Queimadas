
import os

directory = os.getcwd()
burn_folder = os.path.join(directory, "data")
image_folder = os.path.join(directory, "images")

data_parsed_file = os.path.join(directory, 'data_parsed.h5')

months_list = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 
               'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

estados_regioes = {
    "ACRE": "NORTE",
    "ALAGOAS": "NORDESTE",
    "AMAPA": "NORTE",
    "AMAZONAS": "NORTE",
    "BAHIA": "NORDESTE",
    "CEARA": "NORDESTE",
    "DISTRITO FEDERAL": "CENTRO-OESTE",
    "ESPIRITO SANTO": "SUDESTE",
    "GOIAS": "CENTRO-OESTE",
    "MARANHAO": "NORDESTE",
    "MATO GROSSO": "CENTRO-OESTE",
    "MATO GROSSO DO SUL": "CENTRO-OESTE",
    "MINAS GERAIS": "SUDESTE",
    "PARA": "NORTE",
    "PARAIBA": "NORDESTE",
    "PARANA": "SUL",
    "PERNAMBUCO": "NORDESTE",
    "PIAUI": "NORDESTE",
    "RIO DE JANEIRO": "SUDESTE",
    "RIO GRANDE DO NORTE": "NORDESTE",
    "RIO GRANDE DO SUL": "SUL",
    "RONDONIA": "NORTE",
    "RORAIMA": "NORTE",
    "SANTA CATARINA": "SUL",
    "SAO PAULO": "SUDESTE",
    "SERGIPE": "NORDESTE",
    "TOCANTINS": "NORTE"
}

invalid_value = -999
data_timezone = 'America/Sao_Paulo'

# resolution of all sensors in km
resolution_map = { 
    'VIIRS': 0.5,
    'AVHRR-3': 1.1,
    'MODIS': 0.92662543305, # fonte https://modis-fire.umd.edu/files/MODIS_C61_BA_User_Guide_1.1.pdf
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
    'GOES-16': (0.0, 1.0, 1.0, 1.0),
    'Outros': (0.0, 0.0, 0.0, 1.0)
}