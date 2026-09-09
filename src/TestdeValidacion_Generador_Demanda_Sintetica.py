import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import ks_2samp
from scipy.stats import levene
from scipy.stats import wasserstein_distance


BASE_DIR = Path(__file__).resolve().parents[1]

RUTA_HISTORICO = BASE_DIR / "data" / "processed" / "Demanda_completa.xlsx"

RUTA_SINTETICO = BASE_DIR / "data" / "generated" / "demanda_sintetica.xlsx"

RUTA_SALIDA = BASE_DIR / "results" / "validacion_generador.xlsx"

HOJA_SINTETICO = "DemandaSintetica"


TOP_PRODUCTOS_TEST = 10


ALPHA = 0.05


MIN_OBSERVACIONES_POSITIVAS = 3


def normalizar_id_producto(serie):

    return (
        serie
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def clasificar_sbc(adi, cv2):

    if pd.isna(adi) or pd.isna(cv2):
        return "Sin demanda"

    if adi <= 1.32 and cv2 <= 0.49:
        return "Smooth"

    if adi <= 1.32 and cv2 > 0.49:
        return "Erratic"

    if adi > 1.32 and cv2 <= 0.49:
        return "Intermittent"

    return "Lumpy"


def calcular_metricas_serie(serie):

    serie = pd.to_numeric(
        pd.Series(serie),
        errors="coerce"
    ).dropna()

    n_semanas = len(serie)
    positivas = serie[serie > 0]
    n_positivas = len(positivas)

    if n_semanas == 0:
        return {
            "NumeroSemanas": 0,
            "SemanasConDemanda": 0,
            "DemandaTotal": np.nan,
            "DemandaMediaTotal": np.nan,
            "DemandaMediaPositiva": np.nan,
            "DesvStdPositiva": np.nan,
            "PctCeros": np.nan,
            "ADI": np.nan,
            "CV2": np.nan,
            "ClaseSBC": "Sin datos"
        }

    demanda_total = serie.sum()
    media_total = serie.mean()
    pct_ceros = np.mean(serie == 0)

    if n_positivas == 0:

        return {
            "NumeroSemanas": n_semanas,
            "SemanasConDemanda": 0,
            "DemandaTotal": demanda_total,
            "DemandaMediaTotal": media_total,
            "DemandaMediaPositiva": np.nan,
            "DesvStdPositiva": np.nan,
            "PctCeros": pct_ceros,
            "ADI": np.nan,
            "CV2": np.nan,
            "ClaseSBC": "Sin demanda"
        }

    media_positiva = positivas.mean()

    if n_positivas > 1:
        desv_positiva = positivas.std(ddof=1)
    else:
        desv_positiva = 0.0

    adi = n_semanas / n_positivas

    if media_positiva > 0:
        cv2 = (desv_positiva / media_positiva) ** 2
    else:
        cv2 = np.nan

    clase = clasificar_sbc(
        adi=adi,
        cv2=cv2
    )

    return {
        "NumeroSemanas": n_semanas,
        "SemanasConDemanda": n_positivas,
        "DemandaTotal": demanda_total,
        "DemandaMediaTotal": media_total,
        "DemandaMediaPositiva": media_positiva,
        "DesvStdPositiva": desv_positiva,
        "PctCeros": pct_ceros,
        "ADI": adi,
        "CV2": cv2,
        "ClaseSBC": clase
    }


def preparar_historico(ruta):

    df = pd.read_excel(ruta)

    df = df.rename(
        columns={
            "Año": "Anio",
            "Cantidad": "Demanda"
        }
    )

    columnas_necesarias = [
        "IdProducto",
        "Anio",
        "Semana",
        "Demanda"
    ]

    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas en el histórico: "
            + ", ".join(faltantes)
        )

    df["IdProducto"] = normalizar_id_producto(
        df["IdProducto"]
    )

    df["Anio"] = pd.to_numeric(
        df["Anio"],
        errors="coerce"
    )

    df["Semana"] = pd.to_numeric(
        df["Semana"],
        errors="coerce"
    )

    df["Demanda"] = pd.to_numeric(
        df["Demanda"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "IdProducto",
            "Anio",
            "Semana",
            "Demanda"
        ]
    )

    df["Fecha"] = pd.to_datetime(
        df["Anio"].astype(int).astype(str)
        + df["Semana"].astype(int).astype(str).str.zfill(2)
        + "1",
        format="%G%V%u",
        errors="coerce"
    )

    df = df.dropna(
        subset=["Fecha"]
    )


    semanal = (
        df
        .groupby(
            ["IdProducto", "Fecha"],
            as_index=False
        )["Demanda"]
        .sum()
    )

    fecha_min = semanal["Fecha"].min()
    fecha_max = semanal["Fecha"].max()

    rango_fechas = pd.date_range(
        start=fecha_min,
        end=fecha_max,
        freq="W-MON"
    )

    productos = semanal["IdProducto"].unique()
    registros = []

    for producto in productos:

        datos_producto = (
            semanal[
                semanal["IdProducto"] == producto
            ]
            .set_index("Fecha")
            .reindex(
                rango_fechas,
                fill_value=0.0
            )
        )

        datos_producto["IdProducto"] = producto
        datos_producto.index.name = "Fecha"

        registros.append(
            datos_producto.reset_index()
        )

    historico_completo = pd.concat(
        registros,
        ignore_index=True
    )

    return historico_completo[
        ["IdProducto", "Fecha", "Demanda"]
    ]


def preparar_sintetico(ruta, hoja):

    df = pd.read_excel(
        ruta,
        sheet_name=hoja
    )

    columnas_necesarias = [
        "Escenario",
        "IdProducto",
        "Demanda"
    ]

    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas en la demanda sintética: "
            + ", ".join(faltantes)
        )

    df["IdProducto"] = normalizar_id_producto(
        df["IdProducto"]
    )

    df["Escenario"] = pd.to_numeric(
        df["Escenario"],
        errors="coerce"
    )

    df["Demanda"] = pd.to_numeric(
        df["Demanda"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Escenario",
            "IdProducto",
            "Demanda"
        ]
    )

    df["Escenario"] = df["Escenario"].astype(int)

    return df


def crear_metricas_historicas(historico):

    registros = []

    for producto, grupo in historico.groupby(
        "IdProducto"
    ):

        metricas = calcular_metricas_serie(
            grupo.sort_values("Fecha")["Demanda"]
        )

        metricas["IdProducto"] = producto
        registros.append(metricas)

    resultado = pd.DataFrame(registros)

    orden_columnas = [
        "IdProducto",
        "NumeroSemanas",
        "SemanasConDemanda",
        "DemandaTotal",
        "DemandaMediaTotal",
        "DemandaMediaPositiva",
        "DesvStdPositiva",
        "PctCeros",
        "ADI",
        "CV2",
        "ClaseSBC"
    ]

    return resultado[orden_columnas]


def crear_metricas_sinteticas(sintetico):

    registros = []

    for (
        escenario,
        producto
    ), grupo in sintetico.groupby(
        ["Escenario", "IdProducto"]
    ):

        metricas = calcular_metricas_serie(
            grupo["Demanda"]
        )

        metricas["Escenario"] = escenario
        metricas["IdProducto"] = producto

        registros.append(metricas)

    resultado = pd.DataFrame(registros)

    orden_columnas = [
        "Escenario",
        "IdProducto",
        "NumeroSemanas",
        "SemanasConDemanda",
        "DemandaTotal",
        "DemandaMediaTotal",
        "DemandaMediaPositiva",
        "DesvStdPositiva",
        "PctCeros",
        "ADI",
        "CV2",
        "ClaseSBC"
    ]

    return resultado[orden_columnas]


def crear_comparacion_media(
    metricas_historicas,
    metricas_sinteticas
):


    resumen_sintetico = (
        metricas_sinteticas
        .groupby(
            "IdProducto",
            as_index=False
        )
        .agg(
            DemandaTotalSinteticaMedia=(
                "DemandaTotal",
                "mean"
            ),

            DemandaTotalSinteticaStd=(
                "DemandaTotal",
                "std"
            ),

            PctCerosSinteticoMedio=(
                "PctCeros",
                "mean"
            ),

            CV2SinteticoMedio=(
                "CV2",
                "mean"
            ),

            DemandaMediaPositivaSintetica=(
                "DemandaMediaPositiva",
                "mean"
            ),


            NumeroSemanasSinteticas=(
                "NumeroSemanas",
                "sum"
            ),


            SemanasConDemandaSinteticas=(
                "SemanasConDemanda",
                "sum"
            )
        )
    )


    resumen_sintetico["ADISinteticoMedio"] = np.where(
        resumen_sintetico[
            "SemanasConDemandaSinteticas"
        ] > 0,
        (
            resumen_sintetico[
                "NumeroSemanasSinteticas"
            ]
            /
            resumen_sintetico[
                "SemanasConDemandaSinteticas"
            ]
        ),
        np.nan
    )


    historico = metricas_historicas.rename(
        columns={
            "DemandaTotal":
                "DemandaTotalHistorica",

            "PctCeros":
                "PctCerosHistorico",

            "ADI":
                "ADIHistorico",

            "CV2":
                "CV2Historico",

            "DemandaMediaPositiva":
                "DemandaMediaPositivaHistorica",

            "ClaseSBC":
                "ClaseSBCHistorica"
        }
    )

    columnas_historicas = [
        "IdProducto",
        "DemandaTotalHistorica",
        "DemandaMediaPositivaHistorica",
        "PctCerosHistorico",
        "ADIHistorico",
        "CV2Historico",
        "ClaseSBCHistorica"
    ]


    comparacion = resumen_sintetico.merge(
        historico[
            columnas_historicas
        ],
        on="IdProducto",
        how="left"
    )


    comparacion["RatioTotal"] = (
        comparacion[
            "DemandaTotalSinteticaMedia"
        ]
        /
        comparacion[
            "DemandaTotalHistorica"
        ]
    )

    comparacion["ErrorPorcentualTotal"] = (
        comparacion["RatioTotal"] - 1
    ) * 100


    comparacion["ErrorPctCeros"] = (
        comparacion[
            "PctCerosSinteticoMedio"
        ]
        -
        comparacion[
            "PctCerosHistorico"
        ]
    )


    comparacion["ErrorADI"] = (
        comparacion[
            "ADISinteticoMedio"
        ]
        -
        comparacion[
            "ADIHistorico"
        ]
    )


    comparacion["ErrorCV2"] = (
        comparacion[
            "CV2SinteticoMedio"
        ]
        -
        comparacion[
            "CV2Historico"
        ]
    )


    comparacion["ErrorMediaPositivaPct"] = (
        (
            comparacion[
                "DemandaMediaPositivaSintetica"
            ]
            /
            comparacion[
                "DemandaMediaPositivaHistorica"
            ]
        )
        - 1
    ) * 100


    return comparacion.sort_values(
        "DemandaTotalHistorica",
        ascending=False
    )


def calcular_coincidencia_sbc(
    metricas_historicas,
    metricas_sinteticas
):

    clases_historicas = metricas_historicas[
        ["IdProducto", "ClaseSBC"]
    ].rename(
        columns={
            "ClaseSBC": "ClaseSBCHistorica"
        }
    )

    comparacion = metricas_sinteticas.merge(
        clases_historicas,
        on="IdProducto",
        how="left"
    )

    comparacion = comparacion.rename(
        columns={
            "ClaseSBC": "ClaseSBCSintetica"
        }
    )

    comparacion["CoincideClase"] = (
        comparacion["ClaseSBCSintetica"]
        == comparacion["ClaseSBCHistorica"]
    )

    resumen_escenario = (
        comparacion
        .groupby(
            "Escenario",
            as_index=False
        )
        .agg(
            NumeroProductos=(
                "IdProducto",
                "nunique"
            ),
            ProductosCoincidentes=(
                "CoincideClase",
                "sum"
            ),
            PorcentajeCoincidencia=(
                "CoincideClase",
                "mean"
            )
        )
    )

    resumen_escenario["PorcentajeCoincidencia"] *= 100

    resumen_producto = (
        comparacion
        .groupby(
            [
                "IdProducto",
                "ClaseSBCHistorica"
            ],
            as_index=False
        )
        .agg(
            ProporcionEscenariosCoincidentes=(
                "CoincideClase",
                "mean"
            )
        )
    )

    resumen_producto[
        "ProporcionEscenariosCoincidentes"
    ] *= 100

    return (
        comparacion,
        resumen_escenario,
        resumen_producto
    )


def realizar_tests_productos(
    historico,
    sintetico,
    metricas_historicas,
    top_n=10,
    alpha=0.05,
    minimo_observaciones=3
):

    productos_top = (
        metricas_historicas
        .sort_values(
            "DemandaTotal",
            ascending=False
        )
        .head(top_n)["IdProducto"]
        .tolist()
    )

    registros = []

    for producto in productos_top:

        muestra_historica = (
            historico.loc[
                (
                    historico["IdProducto"] == producto
                )
                &
                (
                    historico["Demanda"] > 0
                ),
                "Demanda"
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        demanda_total_historica = (
            metricas_historicas.loc[
                metricas_historicas[
                    "IdProducto"
                ] == producto,
                "DemandaTotal"
            ]
            .iloc[0]
        )

        clase_historica = (
            metricas_historicas.loc[
                metricas_historicas[
                    "IdProducto"
                ] == producto,
                "ClaseSBC"
            ]
            .iloc[0]
        )

        escenarios_producto = (
            sintetico.loc[
                sintetico["IdProducto"] == producto,
                "Escenario"
            ]
            .unique()
        )

        for escenario in escenarios_producto:

            muestra_sintetica = (
                sintetico.loc[
                    (
                        sintetico["IdProducto"]
                        == producto
                    )
                    &
                    (
                        sintetico["Escenario"]
                        == escenario
                    )
                    &
                    (
                        sintetico["Demanda"] > 0
                    ),
                    "Demanda"
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            registro = {
                "IdProducto": producto,
                "ClaseSBCHistorica": clase_historica,
                "DemandaTotalHistorica":
                    demanda_total_historica,
                "Escenario": escenario,
                "NPositivasHistoricas":
                    len(muestra_historica),
                "NPositivasSinteticas":
                    len(muestra_sintetica),
                "KS_Estadistico": np.nan,
                "KS_pvalor": np.nan,
                "KS_NoRechazaH0": np.nan,
                "BF_Estadistico": np.nan,
                "BF_pvalor": np.nan,
                "BF_NoRechazaH0": np.nan,
                "Wasserstein": np.nan,
                "WassersteinNormalizada": np.nan,
                "Observacion": ""
            }

            if (
                len(muestra_historica)
                < minimo_observaciones
                or len(muestra_sintetica)
                < minimo_observaciones
            ):

                registro["Observacion"] = (
                    "Muestras positivas insuficientes"
                )

                registros.append(registro)
                continue


            resultado_ks = ks_2samp(
                muestra_historica,
                muestra_sintetica,
                alternative="two-sided",
                method="auto"
            )

            registro["KS_Estadistico"] = (
                resultado_ks.statistic
            )

            registro["KS_pvalor"] = (
                resultado_ks.pvalue
            )

            registro["KS_NoRechazaH0"] = (
                resultado_ks.pvalue >= alpha
            )


            resultado_bf = levene(
                muestra_historica,
                muestra_sintetica,
                center="median"
            )

            registro["BF_Estadistico"] = (
                resultado_bf.statistic
            )

            registro["BF_pvalor"] = (
                resultado_bf.pvalue
            )

            registro["BF_NoRechazaH0"] = (
                resultado_bf.pvalue >= alpha
            )


            distancia_w = wasserstein_distance(
                muestra_historica,
                muestra_sintetica
            )

            media_historica = (
                np.mean(muestra_historica)
            )

            if media_historica > 0:
                distancia_w_normalizada = (
                    distancia_w
                    / media_historica
                )
            else:
                distancia_w_normalizada = np.nan

            registro["Wasserstein"] = distancia_w

            registro[
                "WassersteinNormalizada"
            ] = distancia_w_normalizada

            registros.append(registro)

    return pd.DataFrame(registros)


def resumir_tests_por_producto(resultados_tests):

    datos_validos = resultados_tests[
        resultados_tests["Observacion"] == ""
    ].copy()

    if datos_validos.empty:
        return pd.DataFrame()


    datos_validos["KS_NoRechazaH0"] = (
        datos_validos["KS_NoRechazaH0"]
        .astype(bool)
    )

    datos_validos["BF_NoRechazaH0"] = (
        datos_validos["BF_NoRechazaH0"]
        .astype(bool)
    )


    resumen = (
        datos_validos
        .groupby(
            [
                "IdProducto",
                "ClaseSBCHistorica",
                "DemandaTotalHistorica"
            ],
            as_index=False
        )
        .agg(
            EscenariosEvaluados=(
                "Escenario",
                "count"
            ),

            KS_EstadisticoMedio=(
                "KS_Estadistico",
                "mean"
            ),

            KS_pvalorMediano=(
                "KS_pvalor",
                "median"
            ),

            PorcentajeKS_NoRechaza=(
                "KS_NoRechazaH0",
                "mean"
            ),

            BF_EstadisticoMedio=(
                "BF_Estadistico",
                "mean"
            ),

            BF_pvalorMediano=(
                "BF_pvalor",
                "median"
            ),

            PorcentajeBF_NoRechaza=(
                "BF_NoRechazaH0",
                "mean"
            ),

            WassersteinMedia=(
                "Wasserstein",
                "mean"
            ),

            WassersteinNormalizadaMedia=(
                "WassersteinNormalizada",
                "mean"
            ),

            WassersteinNormalizadaMediana=(
                "WassersteinNormalizada",
                "median"
            )
        )
    )


    resumen["PorcentajeKS_NoRechaza"] *= 100
    resumen["PorcentajeBF_NoRechaza"] *= 100


    resumen = resumen.sort_values(
        "DemandaTotalHistorica",
        ascending=False
    )

    return resumen


def crear_comparacion_total_global(
    historico,
    sintetico
):

    total_historico = historico["Demanda"].sum()

    totales_sinteticos = (
        sintetico
        .groupby(
            "Escenario",
            as_index=False
        )["Demanda"]
        .sum()
        .rename(
            columns={
                "Demanda": "DemandaTotalSintetica"
            }
        )
    )

    totales_sinteticos[
        "DemandaTotalHistorica"
    ] = total_historico

    totales_sinteticos[
        "RatioSinteticoHistorico"
    ] = (
        totales_sinteticos[
            "DemandaTotalSintetica"
        ]
        / total_historico
    )

    totales_sinteticos[
        "ErrorPorcentual"
    ] = (
        totales_sinteticos[
            "RatioSinteticoHistorico"
        ]
        - 1
    ) * 100

    resumen = pd.DataFrame({
        "Indicador": [
            "Demanda total histórica",
            "Demanda sintética media",
            "Demanda sintética desviación",
            "Demanda sintética mínima",
            "Demanda sintética máxima",
            "Percentil sintético 2,5 %",
            "Percentil sintético 97,5 %",
            "Ratio medio sintético/histórico",
            "Error porcentual medio"
        ],
        "Valor": [
            total_historico,
            totales_sinteticos[
                "DemandaTotalSintetica"
            ].mean(),
            totales_sinteticos[
                "DemandaTotalSintetica"
            ].std(),
            totales_sinteticos[
                "DemandaTotalSintetica"
            ].min(),
            totales_sinteticos[
                "DemandaTotalSintetica"
            ].max(),
            totales_sinteticos[
                "DemandaTotalSintetica"
            ].quantile(0.025),
            totales_sinteticos[
                "DemandaTotalSintetica"
            ].quantile(0.975),
            totales_sinteticos[
                "RatioSinteticoHistorico"
            ].mean(),
            totales_sinteticos[
                "ErrorPorcentual"
            ].mean()
        ]
    })

    return (
        totales_sinteticos,
        resumen
    )


def crear_resumen_indicadores(
    comparacion_media,
    coincidencia_escenarios,
    resumen_tests
):

    error_total_global = (
        comparacion_media[
            "DemandaTotalSinteticaMedia"
        ].sum()
        / comparacion_media[
            "DemandaTotalHistorica"
        ].sum()
        - 1
    ) * 100

    mae_pct_ceros = (
        comparacion_media[
            "ErrorPctCeros"
        ]
        .abs()
        .mean()
    )

    mae_adi = (
        comparacion_media[
            "ErrorADI"
        ]
        .abs()
        .mean()
    )

    mae_cv2 = (
        comparacion_media[
            "ErrorCV2"
        ]
        .abs()
        .mean()
    )

    coincidencia_media = (
        coincidencia_escenarios[
            "PorcentajeCoincidencia"
        ].mean()
    )

    if resumen_tests.empty:

        ks_no_rechaza = np.nan
        bf_no_rechaza = np.nan
        wasserstein_norm = np.nan

    else:

        ks_no_rechaza = (
            resumen_tests[
                "PorcentajeKS_NoRechaza"
            ].mean()
        )

        bf_no_rechaza = (
            resumen_tests[
                "PorcentajeBF_NoRechaza"
            ].mean()
        )

        wasserstein_norm = (
            resumen_tests[
                "WassersteinNormalizadaMedia"
            ].mean()
        )

    return pd.DataFrame({
        "Indicador": [
            "Error porcentual global de demanda",
            "Error absoluto medio de PctCeros",
            "Error absoluto medio de ADI",
            "Error absoluto medio de CV2",
            "Coincidencia media de clase SBC (%)",
            "Escenarios KS sin rechazo (%)",
            "Escenarios Brown-Forsythe sin rechazo (%)",
            "Wasserstein normalizada media"
        ],
        "Valor": [
            error_total_global,
            mae_pct_ceros,
            mae_adi,
            mae_cv2,
            coincidencia_media,
            ks_no_rechaza,
            bf_no_rechaza,
            wasserstein_norm
        ]
    })


if not RUTA_HISTORICO.exists():
    raise FileNotFoundError(
        f"No se encuentra el histórico:\n{RUTA_HISTORICO}"
    )

if not RUTA_SINTETICO.exists():
    raise FileNotFoundError(
        f"No se encuentra el sintético:\n{RUTA_SINTETICO}"
    )

RUTA_SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("Leyendo demanda histórica...")

historico = preparar_historico(
    RUTA_HISTORICO
)

print("Leyendo demanda sintética...")

sintetico = preparar_sintetico(
    RUTA_SINTETICO,
    HOJA_SINTETICO
)

print("Calculando métricas históricas...")

metricas_historicas = (
    crear_metricas_historicas(
        historico
    )
)

print("Calculando métricas sintéticas...")

metricas_sinteticas = (
    crear_metricas_sinteticas(
        sintetico
    )
)

print("Creando comparación media...")

comparacion_media = (
    crear_comparacion_media(
        metricas_historicas,
        metricas_sinteticas
    )
)

print("Calculando coincidencia SBC...")

(
    detalle_clases,
    coincidencia_escenarios,
    coincidencia_productos
) = calcular_coincidencia_sbc(
    metricas_historicas,
    metricas_sinteticas
)

print("Ejecutando tests estadísticos...")

resultados_tests = realizar_tests_productos(
    historico=historico,
    sintetico=sintetico,
    metricas_historicas=metricas_historicas,
    top_n=TOP_PRODUCTOS_TEST,
    alpha=ALPHA,
    minimo_observaciones=(
        MIN_OBSERVACIONES_POSITIVAS
    )
)

resumen_tests = resumir_tests_por_producto(
    resultados_tests
)

print("Comparando demanda global...")

(
    comparacion_total_escenarios,
    resumen_total_global
) = crear_comparacion_total_global(
    historico,
    sintetico
)

resumen_indicadores = crear_resumen_indicadores(
    comparacion_media=comparacion_media,
    coincidencia_escenarios=(
        coincidencia_escenarios
    ),
    resumen_tests=resumen_tests
)


with pd.ExcelWriter(
    RUTA_SALIDA,
    engine="openpyxl"
) as writer:

    resumen_indicadores.to_excel(
        writer,
        sheet_name="ResumenValidacion",
        index=False
    )

    resumen_total_global.to_excel(
        writer,
        sheet_name="ResumenTotal",
        index=False
    )

    comparacion_total_escenarios.to_excel(
        writer,
        sheet_name="TotalEscenarios",
        index=False
    )

    comparacion_media.to_excel(
        writer,
        sheet_name="ComparacionProductos",
        index=False
    )

    metricas_historicas.to_excel(
        writer,
        sheet_name="MetricasHistoricas",
        index=False
    )

    metricas_sinteticas.to_excel(
        writer,
        sheet_name="MetricasSinteticas",
        index=False
    )

    coincidencia_escenarios.to_excel(
        writer,
        sheet_name="CoincidenciaSBC",
        index=False
    )

    coincidencia_productos.to_excel(
        writer,
        sheet_name="CoincidenciaPorProducto",
        index=False
    )

    resumen_tests.to_excel(
        writer,
        sheet_name="ResumenTests",
        index=False
    )

    resultados_tests.to_excel(
        writer,
        sheet_name="DetalleTests",
        index=False
    )


print("\n" + "=" * 70)
print("VALIDACIÓN FINALIZADA")
print("=" * 70)

print("\nArchivo generado:")
print(RUTA_SALIDA)

print("\nResumen general:")
print(
    resumen_indicadores.to_string(
        index=False
    )
)

print("\nProductos sometidos a tests:")
print(
    resumen_tests[
        [
            "IdProducto",
            "ClaseSBCHistorica",
            "DemandaTotalHistorica"
        ]
    ].to_string(index=False)
)

print("\nInterpretación de los tests:")

print(
    f"- KS p >= {ALPHA}: no se detectan diferencias "
    "significativas en la distribución."
)

print(
    f"- Brown-Forsythe p >= {ALPHA}: no se detectan "
    "diferencias significativas en la dispersión."
)

print(
    "- Wasserstein normalizada: cuanto más próxima "
    "a cero, mayor similitud práctica."
)
