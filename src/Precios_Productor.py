import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ruta_entrada = BASE_DIR / "data" / "raw" / "enllaç-v2.xlsx"
ruta_salida = BASE_DIR / "data" / "processed" / "enllaç-v2_precios_medios.xlsx"


df_productos = pd.read_excel(
    ruta_entrada,
    sheet_name="Productos"
)


df_productos["IdProducto"] = (
    df_productos["IdProducto"]
    .astype(str)
    .str.strip()
    .str.extract(r"A_EN_(\d+)", expand=False)
    .astype(int)
    .astype(str)
    + "A"
)


df_productos["PrecioPorUnidad"] = (
    df_productos["PrecioPorUnidad"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df_productos["PrecioPorUnidad"] = pd.to_numeric(
    df_productos["PrecioPorUnidad"],
    errors="coerce"
)


productos_unificados = (
    df_productos
    .groupby(
        [
            "IdProducto",
            "IdProductor",
            "NombreProducto",
            "UnidadMedida"
        ],
        as_index=False,
        dropna=False
    )
    .agg(
        PrecioPorUnidad=("PrecioPorUnidad", "mean")
    )
)

productos_unificados["PrecioPorUnidad"] = (
    productos_unificados["PrecioPorUnidad"].round(2)
)


ruta_salida.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(
    ruta_salida,
    engine="openpyxl"
) as writer:

    productos_unificados.to_excel(
        writer,
        sheet_name="Productos",
        index=False
    )

print("Archivo generado correctamente:")
print(ruta_salida)

print("\nNúmero de filas originales:", len(df_productos))
print("Número de filas unificadas:", len(productos_unificados))
