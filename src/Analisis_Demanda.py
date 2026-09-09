import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_ENTRADA = BASE_DIR / "data" / "processed" / "Demanda_completa.xlsx"
RUTA_SALIDA = BASE_DIR / "data" / "processed" / "matriz_analisis_demanda.xlsx"

df = pd.read_excel(RUTA_ENTRADA)


df = df.rename(columns={
    "Año": "Anio",
    "Cantidad": "Demanda"
})

df["Anio"] = pd.to_numeric(df["Anio"], errors="coerce")
df["Semana"] = pd.to_numeric(df["Semana"], errors="coerce")
df["Demanda"] = pd.to_numeric(df["Demanda"], errors="coerce")

df = df.dropna(subset=["IdProducto", "Anio", "Semana", "Demanda"])


df["Fecha"] = pd.to_datetime(
    df["Anio"].astype(int).astype(str) +
    df["Semana"].astype(int).astype(str).str.zfill(2) +
    "1",
    format="%G%V%u",
    errors="coerce"
)

df = df.dropna(subset=["Fecha"])


demanda_producto = (
    df.groupby(["IdProducto", "Fecha"], as_index=False)["Demanda"]
      .sum()
      .sort_values(["IdProducto", "Fecha"])
)


fecha_min = demanda_producto["Fecha"].min()
fecha_max = demanda_producto["Fecha"].max()

rango_semanal = pd.date_range(
    start=fecha_min,
    end=fecha_max,
    freq="W-MON"
)

productos = demanda_producto["IdProducto"].unique()

series = []

for prod in productos:
    temp = demanda_producto[demanda_producto["IdProducto"] == prod].copy()
    temp = temp.set_index("Fecha").reindex(rango_semanal, fill_value=0.0)
    temp["IdProducto"] = prod
    temp.index.name = "Fecha"
    temp = temp.reset_index()
    series.append(temp)

demanda_producto = pd.concat(series, ignore_index=True)
demanda_producto = demanda_producto[["IdProducto", "Fecha", "Demanda"]]


demanda_producto["AnioISO"] = demanda_producto["Fecha"].dt.isocalendar().year.astype(int)
demanda_producto["SemanaISO"] = demanda_producto["Fecha"].dt.isocalendar().week.astype(int)


media_producto = (
    demanda_producto
    .groupby("IdProducto", as_index=False)["Demanda"]
    .mean()
    .rename(columns={"Demanda": "DemandaMediaTotalProducto"})
)


media_producto_semana = (
    demanda_producto
    .groupby(["IdProducto", "SemanaISO"], as_index=False)["Demanda"]
    .mean()
    .rename(columns={"Demanda": "DemandaMediaSemana"})
)


indice_estacional = media_producto_semana.merge(
    media_producto,
    on="IdProducto",
    how="left"
)


indice_estacional["IndiceEstacional"] = np.where(
    indice_estacional["DemandaMediaTotalProducto"] > 0,
    indice_estacional["DemandaMediaSemana"] / indice_estacional["DemandaMediaTotalProducto"],
    1.0
)


indice_estacional["IndiceEstacional"] = (
    indice_estacional["IndiceEstacional"]
    .clip(0.3, 2.5)
)

indice_estacional = indice_estacional.sort_values(["IdProducto", "SemanaISO"])


def clasificacion_sbc(serie):
    y = serie.values.astype(float)
    positivas = y[y > 0]

    pct_ceros = np.mean(y == 0)

    if len(positivas) == 0:
        return np.nan, np.nan, pct_ceros, "Sin demanda"

    adi = len(y) / len(positivas)

    media = np.mean(positivas)
    std = np.std(positivas, ddof=1) if len(positivas) > 1 else 0.0
    cv2 = (std / media) ** 2 if media > 0 else np.nan

    if adi <= 1.32 and cv2 <= 0.49:
        clase = "Smooth"
    elif adi <= 1.32 and cv2 > 0.49:
        clase = "Erratic"
    elif adi > 1.32 and cv2 <= 0.49:
        clase = "Intermittent"
    else:
        clase = "Lumpy"

    return adi, cv2, pct_ceros, clase


def detectar_tendencia(serie):
    y = serie.values
    x = np.arange(len(y))

    if len(y) < 2:
        return 0, np.nan, "Sin tendencia"

    slope, _, _, pvalue, _ = linregress(x, y)

    if pvalue < 0.05:
        if slope > 0:
            t = "Creciente"
        elif slope < 0:
            t = "Decreciente"
        else:
            t = "Sin tendencia"
    else:
        t = "Sin tendencia"

    return slope, pvalue, t


def detectar_estacionalidad(
    serie,
    lag=52,
    umbral_autocorr=0.30,
    min_demandas_positivas=5
):

    num_semanas = len(serie)
    num_positivos = (serie > 0).sum()
    desv_std = serie.std()
    valores_distintos = serie.nunique()

    if num_semanas <= lag:
        return np.nan, "No evaluable", "Serie demasiado corta para lag 52"

    if valores_distintos <= 1 or desv_std == 0:
        return np.nan, "No evaluable", "Serie constante o sin variabilidad"

    autocorr = pd.Series(serie).autocorr(lag=lag)

    if pd.isna(autocorr):
        return np.nan, "No evaluable", "Autocorrelación no calculable"

    if num_positivos < min_demandas_positivas:
        return autocorr, "No", "Pocas demandas positivas"

    if autocorr >= umbral_autocorr:
        return autocorr, "Sí", "Autocorr_52 >= 0.30 y suficientes demandas positivas"
    else:
        return autocorr, "No", "Autocorr_52 < 0.30"


def analizar_producto(df_prod):
    serie = df_prod.sort_values("Fecha")["Demanda"]

    adi, cv2, pct_ceros, clase = clasificacion_sbc(serie)
    slope, pval, tendencia = detectar_tendencia(serie)
    autocorr, estacionalidad, motivo_estacionalidad = detectar_estacionalidad(serie)


    media_total = serie.mean()


    serie_positiva = serie[serie > 0]
    media = serie_positiva.mean() if len(serie_positiva) > 0 else np.nan


    std = serie.std()


    cv = std / media_total if media_total > 0 else np.nan

    return pd.Series({
        "DemandaTotal": serie.sum(),
        "DemandaMediaTotal": media_total,
        "DemandaMedia": media,
        "DesvStd": std,
        "CV_total": cv,
        "ADI": adi,
        "CV2": cv2,
        "PctCeros": pct_ceros,
        "ClaseSBC": clase,
        "Pendiente": slope,
        "p_valor": pval,
        "Tendencia": tendencia,
        "Autocorr_52": autocorr,
        "Estacionalidad": estacionalidad,
        "MotivoEstacionalidad": motivo_estacionalidad
    })


resultado = (
    demanda_producto
    .groupby("IdProducto")
    .apply(analizar_producto)
    .reset_index()
)


resultado = resultado.sort_values("DemandaTotal", ascending=False)

resultado["Acumulado"] = (
    resultado["DemandaTotal"].cumsum() / resultado["DemandaTotal"].sum()
)

def clasificar_abc(x):
    if x <= 0.8:
        return "A"
    elif x <= 0.95:
        return "B"
    else:
        return "C"

resultado["ABC"] = resultado["Acumulado"].apply(clasificar_abc)


RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(RUTA_SALIDA, engine="openpyxl") as writer:
    resultado.to_excel(writer, sheet_name="MatrizAnalisis", index=False)
    indice_estacional.to_excel(writer, sheet_name="IndiceEstacional", index=False)

print("Archivo generado:", RUTA_SALIDA)
print("Hojas creadas:")
print("- MatrizAnalisis")
print("- IndiceEstacional")


print("\nTipos de demanda:")
print(resultado["ClaseSBC"].value_counts())

print("\nTendencia:")
print(resultado["Tendencia"].value_counts())

print("\nEstacionalidad:")
print(resultado["Estacionalidad"].value_counts())

print("\nMotivos de estacionalidad:")
print(resultado["MotivoEstacionalidad"].value_counts())

print("\nEjemplo de índice estacional:")
print(indice_estacional.head(20))


def graficar_producto(producto_id):
    dfp = demanda_producto[demanda_producto["IdProducto"] == producto_id]

    if dfp.empty:
        print("Producto no encontrado")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(dfp["Fecha"], dfp["Demanda"])
    plt.title(f"Demanda semanal - Producto {producto_id}")
    plt.xlabel("Fecha")
    plt.ylabel("Demanda")
    plt.grid()
    plt.show()


def graficar_indice_estacional(producto_id):
    dfp = indice_estacional[indice_estacional["IdProducto"] == producto_id]

    if dfp.empty:
        print("Producto no encontrado en índice estacional")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(dfp["SemanaISO"], dfp["IndiceEstacional"], marker="o")
    plt.axhline(1, linestyle="--")
    plt.title(f"Índice estacional semanal - Producto {producto_id}")
    plt.xlabel("Semana ISO")
    plt.ylabel("Índice estacional")
    plt.grid()
    plt.show()


def graficar_si_es_estacional(producto_id):
    fila = resultado[resultado["IdProducto"] == producto_id]

    if fila.empty:
        print("Producto no encontrado")
        return

    estacionalidad = fila["Estacionalidad"].iloc[0]
    autocorr = fila["Autocorr_52"].iloc[0]
    motivo = fila["MotivoEstacionalidad"].iloc[0]

    print(f"Producto: {producto_id}")
    print(f"Estacionalidad: {estacionalidad}")
    print(f"Autocorr_52: {autocorr}")
    print(f"Motivo: {motivo}")

    if estacionalidad == "Sí":
        graficar_indice_estacional(producto_id)
    else:
        print("Este producto no ha sido clasificado como estacional.")


def diagnosticar_autocorr(producto_id):
    dfp = demanda_producto[
        demanda_producto["IdProducto"] == producto_id
    ].sort_values("Fecha")

    if dfp.empty:
        print("Producto no encontrado")
        return

    serie = dfp["Demanda"]

    print("Producto:", producto_id)
    print("Número de semanas:", len(serie))
    print("Demandas positivas:", (serie > 0).sum())
    print("Demanda total:", serie.sum())
    print("Demanda media total:", serie.mean())
    print("Desviación estándar:", serie.std())
    print("Valores distintos:", serie.nunique())

    if len(serie) <= 52:
        print("Motivo: serie demasiado corta para calcular autocorrelación a 52 semanas.")
    elif serie.nunique() <= 1 or serie.std() == 0:
        print("Motivo: serie constante o sin variabilidad.")
    else:
        print("Autocorr_52:", pd.Series(serie).autocorr(lag=52))


graficar_producto("100A")
graficar_indice_estacional("100A")
graficar_si_es_estacional("100A")
diagnosticar_autocorr("100A")
