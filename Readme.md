
### TCC INPE Queimadas

#### Setup

1. Clonar o repositório
1. Baixar os [dados das queimadas](https://bit.ly/3IgHIXH)
1. Baixar os [dados auxiliares](https://bit.ly/3DYXow6)
1. Descompactar os zips na mesma pasta do repositório
1. Instalar as dependências do projeto, recomendamos criar um environment separado para as dependências de geociência:

```
conda create -n geo_env
conda activate geo_env
conda config --env --set channel_priority strict
conda install python=3 ipykernel scipy numpy geopandas shapely pyproj rasterio xarray datashader dask pysal
pip3 install contextily geopy matplotlib-scalebar pyorbital osmnx
```

