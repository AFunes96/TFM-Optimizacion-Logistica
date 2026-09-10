from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

BASE_DIR = Path(__file__).resolve().parents[1]

RUTA_DEMANDA = BASE_DIR / "data" / "processed" / "Demanda_completa.xlsx"
RUTA_ANALISIS = BASE_DIR / "data" / "processed" / "matriz_analisis_demanda.xlsx"
RUTA_SALIDA = BASE_DIR / "results" / "validacion_tendencia_mk_bh.xlsx"

df = pd.read_excel(RUTA_DEMANDA)

df = df.rename(columns={
    "Año": "Anio",
    "Cantidad": "Demanda"
})

df["Anio"] = pd.to_numeric(df["Anio"], errors="coerce")
df["Semana"] = pd.to_numeric(df["Semana"], errors="coerce")
df["Demanda"] = pd.to_numeric(df["Demanda"], errors="coerce")

df = df.dropna(subset=["IdProducto", "Anio", "Semana", "Demanda"])

df["Fecha"] = pd.to_datetime(
    df["Anio"].astype(int).astype(str)
    + df["Semana"].astype(int).astype(str).str.zfill(2)
    + "1",
    format="%G%V%u",
    errors="coerce"
)

df = df.dropna(subset=["Fecha"])

semanal = (
    df.groupby(["IdProducto", "Fecha"], as_index=False)["Demanda"]
    .sum()
    .sort_values(["IdProducto", "Fecha"])
)

rango = pd.date_range(
    start=semanal["Fecha"].min(),
    end=semanal["Fecha"].max(),
    freq="W-MON"
)

registros = []

for producto in semanal["IdProducto"].unique():
    serie = (
        semanal[semanal["IdProducto"] == producto]
        .set_index("Fecha")["Demanda"]
        .reindex(rango, fill_value=0.0)
    )

    y = serie.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)

    if np.unique(y).size <= 1:
        tau = np.nan
        p = np.nan
    else:
        resultado = kendalltau(
            x,
            y,
            alternative="two-sided",
            method="auto"
        )
        tau = float(resultado.statistic)
        p = float(resultado.pvalue)

    registros.append({
        "IdProducto": str(producto),
        "Tau_MK": tau,
        "p_MK": p,
        "DemandaTotal": float(serie.sum())
    })

resultado = pd.DataFrame(registros)

validos = resultado["p_MK"].notna()
pvalores = resultado.loc[validos, "p_MK"].to_numpy(dtype=float)

orden = np.argsort(pvalores)
p_ordenados = pvalores[orden]
m = len(p_ordenados)

ajustados_ordenados = (
    p_ordenados
    * m
    / np.arange(1, m + 1)
)

ajustados_ordenados = np.minimum.accumulate(
    ajustados_ordenados[::-1]
)[::-1]

ajustados_ordenados = np.clip(
    ajustados_ordenados,
    0.0,
    1.0
)

ajustados = np.empty_like(ajustados_ordenados)
ajustados[orden] = ajustados_ordenados

resultado["p_MK_BH"] = np.nan
resultado.loc[validos, "p_MK_BH"] = ajustados

resultado["Tendencia_MK_BH"] = np.where(
    (resultado["p_MK_BH"] < 0.05) & (resultado["Tau_MK"] > 0),
    "Creciente",
    np.where(
        (resultado["p_MK_BH"] < 0.05) & (resultado["Tau_MK"] < 0),
        "Decreciente",
        "Sin tendencia"
    )
)

analisis_original = pd.read_excel(
    RUTA_ANALISIS,
    sheet_name="MatrizAnalisis"
)

analisis_original["IdProducto"] = (
    analisis_original["IdProducto"]
    .astype(str)
    .str.strip()
)

resultado = resultado.merge(
    analisis_original[
        [
            "IdProducto",
            "Pendiente",
            "p_valor",
            "Tendencia"
        ]
    ],
    on="IdProducto",
    how="left"
)

resultado["Coincide_OLS_MK_BH"] = (
    resultado["Tendencia"]
    == resultado["Tendencia_MK_BH"]
)

resumen = (
    resultado.groupby(
        "Tendencia_MK_BH",
        as_index=False
    )
    .agg(
        Productos=("IdProducto", "count"),
        DemandaTotal=("DemandaTotal", "sum")
    )
)

resumen["PctProductos"] = (
    resumen["Productos"]
    / resultado["IdProducto"].nunique()
    * 100
)

resumen["PctDemanda"] = (
    resumen["DemandaTotal"]
    / resultado["DemandaTotal"].sum()
    * 100
)

ols_decrecientes = set(
    resultado.loc[
        resultado["Tendencia"] == "Decreciente",
        "IdProducto"
    ]
)

mk_decrecientes = set(
    resultado.loc[
        resultado["Tendencia_MK_BH"] == "Decreciente",
        "IdProducto"
    ]
)

comparacion = pd.DataFrame({
    "Indicador": [
        "Productos decrecientes según OLS original",
        "Productos decrecientes según Mann-Kendall + BH",
        "Productos decrecientes coincidentes en ambos métodos",
        "Productos sin tendencia según Mann-Kendall + BH",
        "Productos crecientes según Mann-Kendall + BH"
    ],
    "Valor": [
        len(ols_decrecientes),
        len(mk_decrecientes),
        len(ols_decrecientes & mk_decrecientes),
        int((resultado["Tendencia_MK_BH"] == "Sin tendencia").sum()),
        int((resultado["Tendencia_MK_BH"] == "Creciente").sum())
    ]
})

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(
    RUTA_SALIDA,
    engine="openpyxl"
) as writer:
    resultado.to_excel(
        writer,
        sheet_name="Detalle",
        index=False
    )
    resumen.to_excel(
        writer,
        sheet_name="ResumenMK",
        index=False
    )
    comparacion.to_excel(
        writer,
        sheet_name="ComparacionOLS",
        index=False
    )

print(resumen.to_string(index=False))
print()
print(comparacion.to_string(index=False))
print()
print("Archivo generado:", RUTA_SALIDA)
