import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

class DataLoader:
    def __init__(self, ruta_excel):
        self.ruta_excel = ruta_excel

    def cargar_hoja(self, nombre_hoja):
        try:
            df = pd.read_excel(self.ruta_excel, sheet_name=nombre_hoja)
            print(f"Hoja '{nombre_hoja}' cargada correctamente ({len(df)} filas).")
            return df
        except Exception as e:
            print(f"Error cargando hoja '{nombre_hoja}': {e}")
            return None

    def crear_geodataframe(self, df, lon_col='Longitud', lat_col='Latitud', crs="EPSG:4326"):
        if df is None:
            print("No hay DataFrame para convertir a GeoDataFrame.")
            return None
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=crs
        )
        return gdf
