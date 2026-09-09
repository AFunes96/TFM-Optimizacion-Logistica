import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_ENTRADA = BASE_DIR / "data" / "raw" / "Demanda.xlsx"
RUTA_SALIDA = BASE_DIR / "data" / "processed" / "Demanda_completa.xlsx"

df = pd.read_excel(RUTA_ENTRADA)

df.columns = df.columns.str.strip()

df = df.groupby(
    ["IdRuta", "IdProducto", "Anio", "Semana"],
    as_index=False
).agg({
    "Cantidad": "sum",
    "IngresoUnidad": "sum"
})

tiempo = pd.MultiIndex.from_product(
    [[2024, 2025], range(1, 53)],
    names=["Anio", "Semana"]
).to_frame(index=False)

productos = df[["IdRuta", "IdProducto"]].drop_duplicates()
base = productos.merge(tiempo, how="cross")
df_completo = base.merge(
    df,
    on=["IdRuta", "IdProducto", "Anio", "Semana"],
    how="left"
)

df_completo["Cantidad"] = df_completo["Cantidad"].fillna(0)
df_completo["IngresoUnidad"] = df_completo["IngresoUnidad"].fillna(0)
df_completo = df_completo.sort_values(
    ["IdRuta", "IdProducto", "Anio", "Semana"]
).reset_index(drop=True)

df_completo = df_completo.rename(columns={"Anio": "Año"})

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
df_completo.to_excel(RUTA_SALIDA, index=False)

print("Archivo creado correctamente.")
