
import os

directory = os.getcwd()
burn_folder = os.path.join(directory, "data")
cache_folder = os.path.join(directory, ".cache")
image_folder = os.path.join(directory, "images")
aux_folder = os.path.join(directory, "aux")
uf_folder = os.path.join(aux_folder, "ibge/BR_UF_2021")
municipios_folder = os.path.join(aux_folder, "ibge/BR_Municipios_2021")
biomas_folder = os.path.join(aux_folder, "ibge/Biomas_250mil")

data_parsed_file = os.path.join(cache_folder, 'data_parsed.h5')

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
    'NOAA-14': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    'METOP-B': (1.0, 0.7333333333333333, 0.47058823529411764, 1.0),
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

# https://service.eumetsat.int/tle/index.html e https://www.n2yo.com/satellite/
tle_data = {
    'METOP-A': ['1 29499U 06044A   23196.77326808  .00001684  00000-0  22646-3 0  9991\n', '2 29499  98.3991 258.0181 0193231 212.5948 301.9976 14.68659002871236\n'], 
    'TERRA': ['1 25994U 99068A   23197.47200441  .00000540  00000-0  12384-3 0  9995\n', '2 25994  98.0934 265.1329 0001598  31.2937  26.0486 14.59292508254057\n'], 
    'NOAA-15': ['1 25338U 98030A   23197.33890104  .00000257  00000-0  12500-3 0  9996\n', '2 25338  98.6042 225.8952 0009521 223.7297 136.3130 14.26352659309411\n'], 
    'AQUA': ['1 27424U 02022A   23197.38051892  .00000974  00000-0  22137-3 0  9991\n', '2 27424  98.2879 141.6968 0002185  81.3475  16.4660 14.58167009127614\n'], 
    'Suomi NPP': ['1 37849U 11061A   23197.22981308  .00000000  00000+0  75253-4 0 00014\n', '2 37849  98.7046 135.4565 0001333 118.6632  45.0185 14.19546962607062\n'],
    'NOAA-12': ['1 21263U 91032A   23196.82918263  .00000146  00000-0  78763-4 0  9995\n', '2 21263  98.5325 191.0719 0012436 268.5187  91.4569 14.26136836673054\n'], 
    'NOAA-14': ['1 23455U 94089A   23197.34699769  .00000072  00000-0  59158-4 0  9995\n', '2 23455  98.4253 207.2326 0008243 272.1375  87.8855 14.14284862472620\n'], 
    'NOAA-18': ['1 28654U 05018A   23197.45350906  .00000216  00000-0  13979-3 0  9994\n', '2 28654  98.9091 271.7017 0013218 298.9739  61.0107 14.12969840935703\n'], 
    'NOAA-17': ['1 27453U 02032A   23197.42362684  .00000177  00000-0  94015-4 0  9993\n', '2 27453  98.7170 144.1512 0011212 201.6327 158.4380 14.25326799 94988\n'], 
    'METOP-C': ['1 43689U 18087A   23197.40331831  .00000003  00000-0  21209-4 0  9990\n', '2 43689  98.6816 256.4389 0002463   5.5126 354.6078 14.21506488243271\n'], 
    'NOAA-16': ['1 26536U 00055A   23197.14343776  .00000102  00000-0  76591-4 0  9994\n', '2 26536  98.5815 246.2840 0011560 118.1890 242.0452 14.13445935176469\n'], 
    'METOP-B': ['1 38771U 12049A   23197.44862595  .00000149  00000-0  88030-4 0  9994\n', '2 38771  98.6771 256.3322 0000946  88.9847 331.3750 14.21510511561704\n'], 
    'NOAA-19': ['1 33591U 09005A   23197.36007622  .00000222  00000-0  14416-3 0  9998\n', '2 33591  99.0978 243.7506 0014352 157.1702 203.0110 14.12786812744068\n'],
    'NOAA-20': ['1 43013U 17073A   23197.37376626  .00000072  00000-0  54967-4 0  9995\n', '2 43013  98.7127 135.3880 0001507  70.8535 289.2805 14.19545535293118\n']
}