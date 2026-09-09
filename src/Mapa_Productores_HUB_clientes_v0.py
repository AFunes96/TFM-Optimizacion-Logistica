import pandas as pd
import geopandas as gpd
from DataLoader import DataLoader
import folium
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ruta = BASE_DIR / "data" / "raw" / "enllaç-v2.xlsx"
RUTA_SALIDA = BASE_DIR / "results" / "mapa_productores_hubs_clientes.html"
loader = DataLoader(ruta)

df_productores = loader.cargar_hoja('Productores')
df_hubs = loader.cargar_hoja('HUBs')
df_clientes = loader.cargar_hoja('Clientes')


gdf_productores = loader.crear_geodataframe(df_productores)
gdf_hubs = loader.crear_geodataframe(df_hubs)
gdf_clientes = loader.crear_geodataframe(df_clientes)

centro = [gdf_productores['Latitud'].mean(), gdf_productores['Longitud'].mean()]
m = folium.Map(location=centro, zoom_start=6, tiles="OpenStreetMap")


for _, row in gdf_productores.iterrows():
    tooltip_text = f"""
    ID: {row['Id']}<br>
    Población: {row['Población']}<br>
    Provincia: {row['Provincia']}
    """
    folium.CircleMarker(
        location=[row['Latitud'], row['Longitud']],
        radius=5,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        tooltip=tooltip_text
    ).add_to(m)


for _, row in gdf_hubs.iterrows():
    tooltip_text = f"""
    ID HUB: {row['Id']}<br>
    Nombre: {row['Nombre']}<br>
    Población: {row['Población']}<br>
    Provincia: {row['Provincia']}
    """
    folium.CircleMarker(
        location=[row['Latitud'], row['Longitud']],
        radius=15,
        color='red',
        fill=True,
        fill_opacity=0.9,
        tooltip=tooltip_text
    ).add_to(m)


for _, row in gdf_clientes.iterrows():
    tooltip_text = f"""
    ID Cliente: {row['Id']}<br>
    Población: {row['Población']}<br>
    Provincia: {row['Provincia']}
    """
    folium.CircleMarker(
        location=[row['Latitud'], row['Longitud']],
        radius=7,
        color='green',
        fill=True,
        fill_opacity=0.7,
        tooltip=tooltip_text
    ).add_to(m)

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
m.save(RUTA_SALIDA)
