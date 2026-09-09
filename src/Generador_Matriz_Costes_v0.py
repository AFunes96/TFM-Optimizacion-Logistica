import requests
import numpy as np
import pandas as pd
from DataLoader import DataLoader
from pathlib import Path

BASE_URL = "http://localhost:5000"
PROFILE = "driving"

BASE_DIR = Path(__file__).resolve().parents[1]
ruta = BASE_DIR / "data" / "raw" / "enllaç-v2.xlsx"
CARPETA_SALIDA = BASE_DIR / "data" / "processed"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
loader = DataLoader(ruta)

df_productores = loader.cargar_hoja("Productores")
df_hubs = loader.cargar_hoja("HUBs")

gdf_productores = loader.crear_geodataframe(df_productores)
gdf_hubs = loader.crear_geodataframe(df_hubs)

df_prod = gdf_productores[["Id", "Longitud", "Latitud"]].copy()
df_prod.columns = ["id", "lon", "lat"]
df_prod["id"] = df_prod["id"].astype(str)
df_prod["tipo"] = "productor"

df_hub = gdf_hubs[["Id", "Longitud", "Latitud"]].copy()
df_hub.columns = ["id", "lon", "lat"]
df_hub["id"] = df_hub["id"].astype(str)
df_hub["tipo"] = "hub"

if len(df_hub) != 1:
    raise ValueError(f"Se esperaba exactamente 1 HUB, pero hay {len(df_hub)}.")

df_nodos = pd.concat([df_prod, df_hub], ignore_index=True).reset_index(drop=True)

ids = df_nodos["id"].tolist()
tipos = df_nodos["tipo"].tolist()
n = len(df_nodos)

coords_str = ";".join(
    f"{row.lon},{row.lat}" for row in df_nodos[["lon", "lat"]].itertuples(index=False)
)

url = f"{BASE_URL}/table/v1/{PROFILE}/{coords_str}?annotations=duration,distance"

r = requests.get(url, timeout=300)
r.raise_for_status()
data = r.json()

if data.get("code") != "Ok":
    raise RuntimeError(data)

if data.get("durations") is None or data.get("distances") is None:
    raise RuntimeError("OSRM devolvió durations/distances=None.")

T = np.array(data["durations"], dtype=np.float32)
D = np.array(data["distances"], dtype=np.float32)

df_T = pd.DataFrame(T, index=ids, columns=ids)
df_D = pd.DataFrame(D, index=ids, columns=ids)

df_T.to_excel(CARPETA_SALIDA / f"matriz_tiempos_{n}.xlsx")
df_D.to_excel(CARPETA_SALIDA / f"matriz_distancias_{n}.xlsx")

np.save(CARPETA_SALIDA / f"osrm_durations_productores_mas_hub_{n}.npy", T)
np.save(CARPETA_SALIDA / f"osrm_distances_productores_mas_hub_{n}.npy", D)

pd.DataFrame({"id": ids, "tipo": tipos, "idx": list(range(n))}).to_csv(
    CARPETA_SALIDA / f"nodos_productores_mas_hub_{n}.csv", index=False
)

print("N nodos:", n)
print("HUB:", df_hub["id"].iloc[0], "-> idx", n - 1)
print("T shape:", T.shape)
print("D shape:", D.shape)
