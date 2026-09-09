import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

RUTA_ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "matriz_analisis_demanda.xlsx"

RUTA_ARCHIVO_SALIDA = BASE_DIR / "data" / "generated" / "demanda_sintetica.xlsx"


NUMERO_SEMANAS = 104


NUMERO_ESCENARIOS = 8


FECHA_INICIO = "2026-01-05"


SEMILLA = 123


DEMANDA_ENTERA = True


DEMANDA_MINIMA_POSITIVA = 1


APLICAR_SOLO_SI_ESTACIONAL = True


PESO_ESTACIONAL_APARICION = 0.50


SUAVIZACION_TENDENCIA = 0.25


FACTOR_TENDENCIA_MINIMO = 0.50
FACTOR_TENDENCIA_MAXIMO = 2.00


def convertir_numerico(valor, valor_defecto=np.nan):

    try:
        valor = float(valor)

        if np.isfinite(valor):
            return valor

        return valor_defecto

    except (TypeError, ValueError):
        return valor_defecto


def normalizar_id_producto(serie):

    return (
        serie
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def media_ponderada(valores, pesos):

    valores = np.asarray(valores, dtype=float)
    pesos = np.asarray(pesos, dtype=float)

    suma_pesos = pesos.sum()

    if suma_pesos <= 0:
        return valores.mean()

    return np.average(valores, weights=pesos)


def calcular_parametros_lognormal(media_positiva, cv2):

    media_positiva = convertir_numerico(
        media_positiva,
        np.nan
    )

    cv2 = convertir_numerico(
        cv2,
        0.0
    )

    if pd.isna(media_positiva) or media_positiva <= 0:
        return np.nan, np.nan

    if pd.isna(cv2) or cv2 < 0:
        cv2 = 0.0

    sigma2 = np.log1p(cv2)
    sigma = np.sqrt(sigma2)

    mu = (
        np.log(media_positiva)
        - sigma2 / 2
    )

    return mu, sigma


def calcular_probabilidad_base(fila):

    pct_ceros = convertir_numerico(
        fila.get("PctCeros", np.nan),
        np.nan
    )

    adi = convertir_numerico(
        fila.get("ADI", np.nan),
        np.nan
    )

    if not pd.isna(pct_ceros):
        probabilidad = 1.0 - pct_ceros

    elif not pd.isna(adi) and adi > 0:
        probabilidad = 1.0 / adi

    else:
        probabilidad = 0.0

    return float(
        np.clip(
            probabilidad,
            0.0,
            1.0
        )
    )


def preparar_indices_estacionales(indice_estacional):

    datos = indice_estacional.copy()

    datos["IdProducto"] = normalizar_id_producto(
        datos["IdProducto"]
    )

    datos["SemanaISO"] = pd.to_numeric(
        datos["SemanaISO"],
        errors="coerce"
    )

    datos["IndiceEstacional"] = pd.to_numeric(
        datos["IndiceEstacional"],
        errors="coerce"
    )

    datos = datos.dropna(
        subset=[
            "IdProducto",
            "SemanaISO",
            "IndiceEstacional"
        ]
    )

    datos = datos[
        datos["IndiceEstacional"] > 0
    ].copy()

    datos["SemanaISO"] = (
        datos["SemanaISO"]
        .astype(int)
    )

    datos = datos.rename(
        columns={
            "IndiceEstacional":
            "IndiceEstacionalOriginal"
        }
    )

    datos["MediaIndiceOriginal"] = (
        datos
        .groupby("IdProducto")[
            "IndiceEstacionalOriginal"
        ]
        .transform("mean")
    )

    datos["IndiceEstacionalNormalizado"] = np.where(
        datos["MediaIndiceOriginal"] > 0,
        (
            datos["IndiceEstacionalOriginal"]
            / datos["MediaIndiceOriginal"]
        ),
        1.0
    )

    datos["MediaIndiceNormalizado"] = (
        datos
        .groupby("IdProducto")[
            "IndiceEstacionalNormalizado"
        ]
        .transform("mean")
    )

    diccionario_indices = {}

    for producto, grupo in datos.groupby("IdProducto"):

        diccionario_indices[producto] = (
            grupo
            .set_index("SemanaISO")[
                "IndiceEstacionalNormalizado"
            ]
            .to_dict()
        )

    return diccionario_indices, datos


def obtener_indices_horizonte(
    producto,
    estacionalidad,
    semanas_iso,
    diccionario_indices,
    aplicar_solo_si_estacional=True
):

    semanas_iso = np.asarray(
        semanas_iso,
        dtype=int
    )

    if (
        aplicar_solo_si_estacional
        and str(estacionalidad).strip() != "Sí"
    ):
        return np.ones(
            len(semanas_iso),
            dtype=float
        )

    indices_producto = diccionario_indices.get(
        str(producto),
        {}
    )

    indices = np.array(
        [
            convertir_numerico(
                indices_producto.get(
                    int(semana),
                    1.0
                ),
                1.0
            )
            for semana in semanas_iso
        ],
        dtype=float
    )

    indices[
        ~np.isfinite(indices)
    ] = 1.0

    indices[
        indices <= 0
    ] = 1.0

    media_horizonte = indices.mean()

    if media_horizonte > 0:
        indices = (
            indices
            / media_horizonte
        )
    else:
        indices = np.ones_like(indices)

    return indices


def dividir_efecto_estacional(
    indices_estacionales,
    peso_aparicion=0.50
):

    peso_aparicion = float(
        np.clip(
            peso_aparicion,
            0.0,
            1.0
        )
    )

    indices_estacionales = np.asarray(
        indices_estacionales,
        dtype=float
    )

    factor_aparicion = (
        indices_estacionales
        ** peso_aparicion
    )

    factor_cantidad = (
        indices_estacionales
        ** (1.0 - peso_aparicion)
    )

    return (
        factor_aparicion,
        factor_cantidad
    )


def calibrar_probabilidades_semanales(
    probabilidad_base,
    factor_aparicion,
    tolerancia=1e-12,
    max_iteraciones=200
):

    probabilidad_base = float(
        np.clip(
            probabilidad_base,
            0.0,
            1.0
        )
    )

    factor_aparicion = np.asarray(
        factor_aparicion,
        dtype=float
    )

    if probabilidad_base <= 0:
        return np.zeros_like(
            factor_aparicion
        )

    if probabilidad_base >= 1:
        return np.ones_like(
            factor_aparicion
        )

    def calcular_media(escala):
        probabilidades = np.clip(
            probabilidad_base
            * escala
            * factor_aparicion,
            0.0,
            1.0
        )

        return probabilidades.mean()

    limite_inferior = 0.0
    limite_superior = 1.0

    while (
        calcular_media(limite_superior)
        < probabilidad_base
        and limite_superior < 1e12
    ):
        limite_superior *= 2.0

    for _ in range(max_iteraciones):

        punto_medio = (
            limite_inferior
            + limite_superior
        ) / 2.0

        media_actual = calcular_media(
            punto_medio
        )

        if (
            abs(
                media_actual
                - probabilidad_base
            )
            <= tolerancia
        ):
            break

        if media_actual < probabilidad_base:
            limite_inferior = punto_medio
        else:
            limite_superior = punto_medio

    escala_final = (
        limite_inferior
        + limite_superior
    ) / 2.0

    probabilidades = np.clip(
        probabilidad_base
        * escala_final
        * factor_aparicion,
        0.0,
        1.0
    )

    return probabilidades


def normalizar_factor_cantidad(
    factor_cantidad,
    probabilidades_semanales
):

    factor_cantidad = np.asarray(
        factor_cantidad,
        dtype=float
    )

    probabilidades_semanales = np.asarray(
        probabilidades_semanales,
        dtype=float
    )

    media_factor = media_ponderada(
        factor_cantidad,
        probabilidades_semanales
    )

    if (
        not np.isfinite(media_factor)
        or media_factor <= 0
    ):
        return np.ones_like(
            factor_cantidad
        )

    return (
        factor_cantidad
        / media_factor
    )


def calcular_factores_tendencia(
    fila,
    numero_semanas,
    pesos_normalizacion,
    suavizacion_tendencia=0.25,
    factor_minimo=0.50,
    factor_maximo=2.00
):

    tendencia = str(
        fila.get(
            "Tendencia",
            "Sin tendencia"
        )
    ).strip()

    pendiente = convertir_numerico(
        fila.get(
            "Pendiente",
            0.0
        ),
        0.0
    )

    media_total = convertir_numerico(
        fila.get(
            "DemandaMediaTotal",
            np.nan
        ),
        np.nan
    )

    if (
        tendencia not in [
            "Creciente",
            "Decreciente"
        ]
        or pd.isna(media_total)
        or media_total <= 0
    ):
        return np.ones(
            numero_semanas,
            dtype=float
        )

    suavizacion_tendencia = float(
        np.clip(
            suavizacion_tendencia,
            0.0,
            1.0
        )
    )

    tasa_semanal = (
        pendiente
        / media_total
    )

    semanas = np.arange(
        numero_semanas,
        dtype=float
    )

    centro_horizonte = (
        numero_semanas - 1
    ) / 2.0

    semanas_centradas = (
        semanas
        - centro_horizonte
    )

    factores = (
        1.0
        + suavizacion_tendencia
        * tasa_semanal
        * semanas_centradas
    )

    factores = np.clip(
        factores,
        factor_minimo,
        factor_maximo
    )

    pesos_normalizacion = np.asarray(
        pesos_normalizacion,
        dtype=float
    )


    media_factor = media_ponderada(
        factores,
        pesos_normalizacion
    )

    if (
        np.isfinite(media_factor)
        and media_factor > 0
    ):
        factores = (
            factores
            / media_factor
        )

    return factores


def preparar_perfil_producto(
    fila,
    fechas,
    diccionario_indices,
    aplicar_solo_si_estacional=True,
    peso_estacional_aparicion=0.50,
    suavizacion_tendencia=0.25,
    factor_tendencia_minimo=0.50,
    factor_tendencia_maximo=2.00
):

    producto = str(
        fila["IdProducto"]
    )

    estacionalidad = str(
        fila.get(
            "Estacionalidad",
            "No"
        )
    ).strip()

    probabilidad_base = (
        calcular_probabilidad_base(fila)
    )

    semanas_iso = np.array(
        [
            int(fecha.isocalendar().week)
            for fecha in fechas
        ],
        dtype=int
    )

    indices_horizonte = (
        obtener_indices_horizonte(
            producto=producto,
            estacionalidad=estacionalidad,
            semanas_iso=semanas_iso,
            diccionario_indices=(
                diccionario_indices
            ),
            aplicar_solo_si_estacional=(
                aplicar_solo_si_estacional
            )
        )
    )

    (
        factor_aparicion,
        factor_cantidad
    ) = dividir_efecto_estacional(
        indices_estacionales=(
            indices_horizonte
        ),
        peso_aparicion=(
            peso_estacional_aparicion
        )
    )

    probabilidades_semanales = (
        calibrar_probabilidades_semanales(
            probabilidad_base=(
                probabilidad_base
            ),
            factor_aparicion=(
                factor_aparicion
            )
        )
    )

    factor_cantidad = (
        normalizar_factor_cantidad(
            factor_cantidad=(
                factor_cantidad
            ),
            probabilidades_semanales=(
                probabilidades_semanales
            )
        )
    )


    pesos_tendencia = (
        probabilidades_semanales
        * factor_cantidad
    )

    factores_tendencia = (
        calcular_factores_tendencia(
            fila=fila,
            numero_semanas=len(fechas),
            pesos_normalizacion=(
                pesos_tendencia
            ),
            suavizacion_tendencia=(
                suavizacion_tendencia
            ),
            factor_minimo=(
                factor_tendencia_minimo
            ),
            factor_maximo=(
                factor_tendencia_maximo
            )
        )
    )

    return {
        "probabilidad_base":
            probabilidad_base,

        "indice_estacional":
            indices_horizonte,

        "factor_aparicion":
            factor_aparicion,

        "factor_cantidad":
            factor_cantidad,

        "probabilidades_semanales":
            probabilidades_semanales,

        "factores_tendencia":
            factores_tendencia
    }


def obtener_media_positiva(fila):

    if "DemandaMediaPositiva" in fila.index:

        return convertir_numerico(
            fila.get(
                "DemandaMediaPositiva",
                np.nan
            ),
            np.nan
        )

    return convertir_numerico(
        fila.get(
            "DemandaMedia",
            np.nan
        ),
        np.nan
    )


def generar_cantidad_positiva(
    fila,
    rng,
    factor_cantidad,
    factor_tendencia,
    demanda_entera=True,
    demanda_minima=1
):

    media_positiva = (
        obtener_media_positiva(fila)
    )

    cv2 = convertir_numerico(
        fila.get(
            "CV2",
            0.0
        ),
        0.0
    )

    if (
        pd.isna(media_positiva)
        or media_positiva <= 0
    ):
        return 0

    mu, sigma = (
        calcular_parametros_lognormal(
            media_positiva,
            cv2
        )
    )

    if (
        pd.isna(mu)
        or pd.isna(sigma)
    ):
        return 0

    if sigma < 1e-12:
        cantidad_base = media_positiva

    else:
        cantidad_base = rng.lognormal(
            mean=mu,
            sigma=sigma
        )

    cantidad = (
        cantidad_base
        * factor_cantidad
        * factor_tendencia
    )

    if demanda_entera:

        cantidad = int(
            np.rint(cantidad)
        )

        if cantidad < demanda_minima:
            cantidad = demanda_minima

    else:

        cantidad = max(
            float(cantidad),
            float(demanda_minima)
        )

    return cantidad


def generar_demanda_sintetica(
    matriz_analisis,
    indice_estacional,
    fecha_inicio,
    numero_semanas,
    numero_escenarios=1,
    semilla=123,
    demanda_entera=True,
    demanda_minima_positiva=1,
    aplicar_solo_si_estacional=True,
    peso_estacional_aparicion=0.50,
    suavizacion_tendencia=0.25,
    factor_tendencia_minimo=0.50,
    factor_tendencia_maximo=2.00
):

    if numero_semanas <= 0:
        raise ValueError(
            "NUMERO_SEMANAS debe ser mayor que cero."
        )

    if numero_escenarios <= 0:
        raise ValueError(
            "NUMERO_ESCENARIOS debe ser mayor que cero."
        )

    fecha_inicio = pd.to_datetime(
        fecha_inicio,
        errors="raise"
    )


    fecha_inicio = (
        fecha_inicio
        - pd.to_timedelta(
            fecha_inicio.weekday(),
            unit="D"
        )
    )

    fechas = pd.date_range(
        start=fecha_inicio,
        periods=numero_semanas,
        freq="W-MON"
    )

    matriz_analisis = (
        matriz_analisis.copy()
    )

    indice_estacional = (
        indice_estacional.copy()
    )

    matriz_analisis["IdProducto"] = (
        normalizar_id_producto(
            matriz_analisis["IdProducto"]
        )
    )

    indice_estacional["IdProducto"] = (
        normalizar_id_producto(
            indice_estacional["IdProducto"]
        )
    )

    (
        diccionario_indices,
        indices_normalizados
    ) = preparar_indices_estacionales(
        indice_estacional
    )


    perfiles_productos = {}

    for _, fila in matriz_analisis.iterrows():

        producto = str(
            fila["IdProducto"]
        )

        perfiles_productos[producto] = (
            preparar_perfil_producto(
                fila=fila,
                fechas=fechas,
                diccionario_indices=(
                    diccionario_indices
                ),
                aplicar_solo_si_estacional=(
                    aplicar_solo_si_estacional
                ),
                peso_estacional_aparicion=(
                    peso_estacional_aparicion
                ),
                suavizacion_tendencia=(
                    suavizacion_tendencia
                ),
                factor_tendencia_minimo=(
                    factor_tendencia_minimo
                ),
                factor_tendencia_maximo=(
                    factor_tendencia_maximo
                )
            )
        )

    registros = []

    secuencia_semillas = (
        np.random.SeedSequence(
            semilla
        )
    )

    semillas_escenarios = (
        secuencia_semillas.spawn(
            numero_escenarios
        )
    )

    for escenario in range(
        1,
        numero_escenarios + 1
    ):

        rng = np.random.default_rng(
            semillas_escenarios[
                escenario - 1
            ]
        )

        for _, fila in matriz_analisis.iterrows():

            producto = str(
                fila["IdProducto"]
            )

            clase_sbc = str(
                fila.get(
                    "ClaseSBC",
                    ""
                )
            ).strip()

            estacionalidad = str(
                fila.get(
                    "Estacionalidad",
                    "No"
                )
            ).strip()

            media_positiva = (
                obtener_media_positiva(
                    fila
                )
            )

            perfil = (
                perfiles_productos[
                    producto
                ]
            )

            probabilidad_base = (
                perfil[
                    "probabilidad_base"
                ]
            )

            producto_sin_demanda = (
                clase_sbc == "Sin demanda"
                or probabilidad_base <= 0
                or pd.isna(media_positiva)
                or media_positiva <= 0
            )

            for numero_semana, fecha in enumerate(
                fechas
            ):

                calendario_iso = (
                    fecha.isocalendar()
                )

                anio_iso = int(
                    calendario_iso.year
                )

                semana_iso = int(
                    calendario_iso.week
                )

                indice_estacional_semana = (
                    perfil[
                        "indice_estacional"
                    ][numero_semana]
                )

                factor_aparicion = (
                    perfil[
                        "factor_aparicion"
                    ][numero_semana]
                )

                factor_cantidad = (
                    perfil[
                        "factor_cantidad"
                    ][numero_semana]
                )

                probabilidad_semana = (
                    perfil[
                        "probabilidades_semanales"
                    ][numero_semana]
                )

                factor_tendencia = (
                    perfil[
                        "factores_tendencia"
                    ][numero_semana]
                )

                if producto_sin_demanda:

                    aparece_demanda = False
                    demanda = 0

                else:

                    aparece_demanda = (
                        rng.random()
                        < probabilidad_semana
                    )

                    if aparece_demanda:

                        demanda = (
                            generar_cantidad_positiva(
                                fila=fila,
                                rng=rng,
                                factor_cantidad=(
                                    factor_cantidad
                                ),
                                factor_tendencia=(
                                    factor_tendencia
                                ),
                                demanda_entera=(
                                    demanda_entera
                                ),
                                demanda_minima=(
                                    demanda_minima_positiva
                                )
                            )
                        )

                    else:
                        demanda = 0

                registros.append({
                    "Escenario":
                        escenario,

                    "NumeroSemanaSimulada":
                        numero_semana + 1,

                    "Fecha":
                        fecha,

                    "AnioISO":
                        anio_iso,

                    "SemanaISO":
                        semana_iso,

                    "IdProducto":
                        producto,

                    "Demanda":
                        demanda,

                    "ApareceDemanda":
                        int(aparece_demanda),

                    "ProbabilidadBase":
                        probabilidad_base,

                    "ProbabilidadSemana":
                        probabilidad_semana,

                    "IndiceEstacional":
                        indice_estacional_semana,

                    "FactorEstacionalAparicion":
                        factor_aparicion,

                    "FactorEstacionalCantidad":
                        factor_cantidad,

                    "FactorTendencia":
                        factor_tendencia,

                    "ClaseSBC_Historica":
                        clase_sbc,

                    "EstacionalidadHistorica":
                        estacionalidad
                })

    demanda_sintetica = pd.DataFrame(
        registros
    )

    demanda_sintetica = (
        demanda_sintetica
        .sort_values(
            [
                "Escenario",
                "Fecha",
                "IdProducto"
            ]
        )
        .reset_index(drop=True)
    )

    return (
        demanda_sintetica,
        indices_normalizados,
        perfiles_productos
    )


def clasificar_sbc_sintetica(serie):

    y = np.asarray(
        serie,
        dtype=float
    )

    positivas = y[
        y > 0
    ]

    if len(y) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            "Sin datos"
        )

    pct_ceros = np.mean(
        y == 0
    )

    if len(positivas) == 0:
        return (
            np.nan,
            np.nan,
            pct_ceros,
            "Sin demanda"
        )

    adi = (
        len(y)
        / len(positivas)
    )

    media_positiva = (
        positivas.mean()
    )

    if len(positivas) > 1:

        desviacion_positiva = (
            positivas.std(
                ddof=1
            )
        )

    else:
        desviacion_positiva = 0.0

    if media_positiva > 0:

        cv2 = (
            desviacion_positiva
            / media_positiva
        ) ** 2

    else:
        cv2 = np.nan

    if (
        adi <= 1.32
        and cv2 <= 0.49
    ):
        clase = "Smooth"

    elif (
        adi <= 1.32
        and cv2 > 0.49
    ):
        clase = "Erratic"

    elif (
        adi > 1.32
        and cv2 <= 0.49
    ):
        clase = "Intermittent"

    else:
        clase = "Lumpy"

    return (
        adi,
        cv2,
        pct_ceros,
        clase
    )


def crear_validacion_productos(
    demanda_sintetica,
    matriz_analisis
):

    registros = []

    for (
        escenario,
        producto
    ), grupo in demanda_sintetica.groupby(
        [
            "Escenario",
            "IdProducto"
        ]
    ):

        serie = (
            grupo
            .sort_values("Fecha")[
                "Demanda"
            ]
        )

        positivas = serie[
            serie > 0
        ]

        (
            adi,
            cv2,
            pct_ceros,
            clase
        ) = clasificar_sbc_sintetica(
            serie
        )

        registros.append({
            "Escenario":
                escenario,

            "IdProducto":
                producto,

            "DemandaTotalSintetica":
                serie.sum(),

            "DemandaMediaTotalSintetica":
                serie.mean(),

            "DemandaMediaPositivaSintetica":
                (
                    positivas.mean()
                    if len(positivas) > 0
                    else np.nan
                ),

            "DesvStdPositivaSintetica":
                (
                    positivas.std(ddof=1)
                    if len(positivas) > 1
                    else 0.0
                ),

            "SemanasConDemandaSinteticas":
                int(
                    (serie > 0).sum()
                ),

            "PctCerosSintetico":
                pct_ceros,

            "ADISintetico":
                adi,

            "CV2Sintetico":
                cv2,

            "ClaseSBC_Sintetica":
                clase,

            "ProbabilidadBase":
                grupo[
                    "ProbabilidadBase"
                ].iloc[0],

            "ProbabilidadSemanaMedia":
                grupo[
                    "ProbabilidadSemana"
                ].mean(),

            "IndiceEstacionalMedio":
                grupo[
                    "IndiceEstacional"
                ].mean(),

            "FactorCantidadMedio":
                grupo[
                    "FactorEstacionalCantidad"
                ].mean(),

            "FactorTendenciaMedio":
                grupo[
                    "FactorTendencia"
                ].mean()
        })

    resumen = pd.DataFrame(
        registros
    )

    columnas_historicas = [
        "IdProducto",
        "DemandaTotal",
        "DemandaMediaTotal",
        "DemandaMedia",
        "DemandaMediaPositiva",
        "ADI",
        "CV2",
        "PctCeros",
        "ClaseSBC",
        "Tendencia",
        "Pendiente",
        "Estacionalidad"
    ]

    columnas_disponibles = [
        columna
        for columna in columnas_historicas
        if columna in matriz_analisis.columns
    ]

    historico = (
        matriz_analisis[
            columnas_disponibles
        ]
        .copy()
    )

    historico["IdProducto"] = (
        normalizar_id_producto(
            historico["IdProducto"]
        )
    )

    historico = historico.rename(
        columns={
            "DemandaTotal":
                "DemandaTotalHistorica",

            "DemandaMediaTotal":
                "DemandaMediaTotalHistorica",

            "DemandaMedia":
                "DemandaMediaPositivaHistorica",

            "DemandaMediaPositiva":
                "DemandaMediaPositivaHistorica",

            "ADI":
                "ADIHistorico",

            "CV2":
                "CV2Historico",

            "PctCeros":
                "PctCerosHistorico",

            "ClaseSBC":
                "ClaseSBC_Historica",

            "Tendencia":
                "TendenciaHistorica",

            "Pendiente":
                "PendienteHistorica",

            "Estacionalidad":
                "EstacionalidadHistorica"
        }
    )

    historico = historico.loc[
        :,
        ~historico.columns.duplicated()
    ]

    resumen = resumen.merge(
        historico,
        on="IdProducto",
        how="left"
    )

    resumen["RatioTotalSinteticoHistorico"] = (
        resumen["DemandaTotalSintetica"]
        / resumen["DemandaTotalHistorica"]
    )

    resumen["DiferenciaPctCeros"] = (
        resumen["PctCerosSintetico"]
        - resumen["PctCerosHistorico"]
    )

    resumen["DiferenciaADI"] = (
        resumen["ADISintetico"]
        - resumen["ADIHistorico"]
    )

    resumen["DiferenciaCV2"] = (
        resumen["CV2Sintetico"]
        - resumen["CV2Historico"]
    )

    return resumen


def crear_validacion_media(
    validacion_productos,
    matriz_analisis
):

    validacion_media = (
        validacion_productos
        .groupby(
            "IdProducto",
            as_index=False
        )
        .agg(
            DemandaTotalSinteticaMedia=(
                "DemandaTotalSintetica",
                "mean"
            ),

            DemandaTotalSinteticaStd=(
                "DemandaTotalSintetica",
                "std"
            ),

            DemandaMediaPositivaSinteticaMedia=(
                "DemandaMediaPositivaSintetica",
                "mean"
            ),

            PctCerosSinteticoMedio=(
                "PctCerosSintetico",
                "mean"
            ),

            ADISinteticoMedio=(
                "ADISintetico",
                "mean"
            ),

            CV2SinteticoMedio=(
                "CV2Sintetico",
                "mean"
            )
        )
    )

    columnas_historicas = [
        "IdProducto",
        "DemandaTotal",
        "DemandaMediaTotal",
        "DemandaMedia",
        "DemandaMediaPositiva",
        "PctCeros",
        "ADI",
        "CV2",
        "ClaseSBC"
    ]

    columnas_disponibles = [
        columna
        for columna in columnas_historicas
        if columna in matriz_analisis.columns
    ]

    historico = (
        matriz_analisis[
            columnas_disponibles
        ]
        .copy()
    )

    historico["IdProducto"] = (
        normalizar_id_producto(
            historico["IdProducto"]
        )
    )

    historico = historico.rename(
        columns={
            "DemandaTotal":
                "DemandaTotalHistorica",

            "DemandaMediaTotal":
                "DemandaMediaTotalHistorica",

            "DemandaMedia":
                "DemandaMediaPositivaHistorica",

            "DemandaMediaPositiva":
                "DemandaMediaPositivaHistorica",

            "PctCeros":
                "PctCerosHistorico",

            "ADI":
                "ADIHistorico",

            "CV2":
                "CV2Historico",

            "ClaseSBC":
                "ClaseSBCHistorica"
        }
    )

    historico = historico.loc[
        :,
        ~historico.columns.duplicated()
    ]

    validacion_media = (
        validacion_media.merge(
            historico,
            on="IdProducto",
            how="left"
        )
    )

    validacion_media[
        "RatioTotalMedio"
    ] = (
        validacion_media[
            "DemandaTotalSinteticaMedia"
        ]
        / validacion_media[
            "DemandaTotalHistorica"
        ]
    )

    validacion_media[
        "ErrorPorcentualTotal"
    ] = (
        validacion_media[
            "RatioTotalMedio"
        ]
        - 1.0
    ) * 100

    validacion_media[
        "ErrorPctCeros"
    ] = (
        validacion_media[
            "PctCerosSinteticoMedio"
        ]
        - validacion_media[
            "PctCerosHistorico"
        ]
    )

    validacion_media[
        "ErrorADI"
    ] = (
        validacion_media[
            "ADISinteticoMedio"
        ]
        - validacion_media[
            "ADIHistorico"
        ]
    )

    validacion_media[
        "ErrorCV2"
    ] = (
        validacion_media[
            "CV2SinteticoMedio"
        ]
        - validacion_media[
            "CV2Historico"
        ]
    )

    return validacion_media


def crear_resumen_semanal(
    demanda_sintetica
):

    return (
        demanda_sintetica
        .groupby(
            [
                "Escenario",
                "NumeroSemanaSimulada",
                "Fecha",
                "AnioISO",
                "SemanaISO"
            ],
            as_index=False
        )
        .agg(
            DemandaTotalSemana=(
                "Demanda",
                "sum"
            ),

            ProductosConDemanda=(
                "ApareceDemanda",
                "sum"
            ),

            NumeroProductos=(
                "IdProducto",
                "nunique"
            ),

            ProbabilidadMediaSemana=(
                "ProbabilidadSemana",
                "mean"
            ),

            FactorTendenciaMedio=(
                "FactorTendencia",
                "mean"
            )
        )
    )


if not RUTA_ARCHIVO_ENTRADA.exists():

    raise FileNotFoundError(
        f"No se ha encontrado el archivo:\n"
        f"{RUTA_ARCHIVO_ENTRADA}"
    )

RUTA_ARCHIVO_SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

matriz_analisis = pd.read_excel(
    RUTA_ARCHIVO_ENTRADA,
    sheet_name="MatrizAnalisis"
)

indice_estacional = pd.read_excel(
    RUTA_ARCHIVO_ENTRADA,
    sheet_name="IndiceEstacional"
)


columnas_obligatorias_matriz = [
    "IdProducto",
    "DemandaTotal",
    "DemandaMediaTotal",
    "ADI",
    "CV2",
    "PctCeros",
    "ClaseSBC",
    "Pendiente",
    "Tendencia",
    "Estacionalidad"
]

faltantes_matriz = [
    columna
    for columna in columnas_obligatorias_matriz
    if columna not in matriz_analisis.columns
]

if faltantes_matriz:

    raise ValueError(
        "Faltan columnas en MatrizAnalisis:\n"
        + ", ".join(faltantes_matriz)
    )

if (
    "DemandaMedia" not in matriz_analisis.columns
    and
    "DemandaMediaPositiva"
    not in matriz_analisis.columns
):

    raise ValueError(
        "Debe existir la columna DemandaMedia "
        "o DemandaMediaPositiva."
    )

columnas_obligatorias_indice = [
    "IdProducto",
    "SemanaISO",
    "IndiceEstacional"
]

faltantes_indice = [
    columna
    for columna in columnas_obligatorias_indice
    if columna not in indice_estacional.columns
]

if faltantes_indice:

    raise ValueError(
        "Faltan columnas en IndiceEstacional:\n"
        + ", ".join(faltantes_indice)
    )


(
    demanda_sintetica,
    indices_normalizados,
    perfiles_productos
) = generar_demanda_sintetica(
    matriz_analisis=matriz_analisis,
    indice_estacional=indice_estacional,
    fecha_inicio=FECHA_INICIO,
    numero_semanas=NUMERO_SEMANAS,
    numero_escenarios=NUMERO_ESCENARIOS,
    semilla=SEMILLA,
    demanda_entera=DEMANDA_ENTERA,
    demanda_minima_positiva=(
        DEMANDA_MINIMA_POSITIVA
    ),
    aplicar_solo_si_estacional=(
        APLICAR_SOLO_SI_ESTACIONAL
    ),
    peso_estacional_aparicion=(
        PESO_ESTACIONAL_APARICION
    ),
    suavizacion_tendencia=(
        SUAVIZACION_TENDENCIA
    ),
    factor_tendencia_minimo=(
        FACTOR_TENDENCIA_MINIMO
    ),
    factor_tendencia_maximo=(
        FACTOR_TENDENCIA_MAXIMO
    )
)


validacion_productos = (
    crear_validacion_productos(
        demanda_sintetica=(
            demanda_sintetica
        ),
        matriz_analisis=(
            matriz_analisis
        )
    )
)

validacion_media = crear_validacion_media(
    validacion_productos=(
        validacion_productos
    ),
    matriz_analisis=(
        matriz_analisis
    )
)

resumen_semanal = crear_resumen_semanal(
    demanda_sintetica
)

demanda_optimizador = (
    demanda_sintetica[
        [
            "Escenario",
            "NumeroSemanaSimulada",
            "Fecha",
            "AnioISO",
            "SemanaISO",
            "IdProducto",
            "Demanda"
        ]
    ]
    .copy()
)


total_historico = pd.to_numeric(
    matriz_analisis["DemandaTotal"],
    errors="coerce"
).sum()

comparacion_totales = (
    demanda_sintetica
    .groupby(
        "Escenario",
        as_index=False
    )["Demanda"]
    .sum()
    .rename(
        columns={
            "Demanda":
                "DemandaTotalSintetica"
        }
    )
)

comparacion_totales[
    "DemandaTotalHistorica"
] = total_historico

comparacion_totales[
    "RatioSinteticoHistorico"
] = (
    comparacion_totales[
        "DemandaTotalSintetica"
    ]
    / comparacion_totales[
        "DemandaTotalHistorica"
    ]
)

comparacion_totales[
    "DiferenciaPorcentual"
] = (
    comparacion_totales[
        "RatioSinteticoHistorico"
    ]
    - 1.0
) * 100


registros_perfiles = []

for producto, perfil in perfiles_productos.items():

    for numero_semana in range(
        NUMERO_SEMANAS
    ):

        registros_perfiles.append({
            "IdProducto":
                producto,

            "NumeroSemanaSimulada":
                numero_semana + 1,

            "ProbabilidadBase":
                perfil[
                    "probabilidad_base"
                ],

            "IndiceEstacional":
                perfil[
                    "indice_estacional"
                ][numero_semana],

            "FactorEstacionalAparicion":
                perfil[
                    "factor_aparicion"
                ][numero_semana],

            "FactorEstacionalCantidad":
                perfil[
                    "factor_cantidad"
                ][numero_semana],

            "ProbabilidadSemana":
                perfil[
                    "probabilidades_semanales"
                ][numero_semana],

            "FactorTendencia":
                perfil[
                    "factores_tendencia"
                ][numero_semana]
        })

perfiles_semanales = pd.DataFrame(
    registros_perfiles
)


with pd.ExcelWriter(
    RUTA_ARCHIVO_SALIDA,
    engine="openpyxl"
) as writer:

    demanda_optimizador.to_excel(
        writer,
        sheet_name="DemandaSintetica",
        index=False
    )

    demanda_sintetica.to_excel(
        writer,
        sheet_name="DetalleGeneracion",
        index=False
    )

    validacion_productos.to_excel(
        writer,
        sheet_name="ValidacionProductos",
        index=False
    )

    validacion_media.to_excel(
        writer,
        sheet_name="ValidacionMedia",
        index=False
    )

    resumen_semanal.to_excel(
        writer,
        sheet_name="ResumenSemanal",
        index=False
    )

    comparacion_totales.to_excel(
        writer,
        sheet_name="ComparacionTotales",
        index=False
    )

    indices_normalizados.to_excel(
        writer,
        sheet_name="IndicesNormalizados",
        index=False
    )

    perfiles_semanales.to_excel(
        writer,
        sheet_name="PerfilesSemanales",
        index=False
    )

    matriz_analisis.to_excel(
        writer,
        sheet_name="ParametrosHistoricos",
        index=False
    )


print("=" * 75)
print("GENERACIÓN DE DEMANDA SINTÉTICA FINALIZADA")
print("=" * 75)

print("\nArchivo generado:")
print(RUTA_ARCHIVO_SALIDA)

print("\nNúmero de productos:")
print(
    matriz_analisis[
        "IdProducto"
    ].nunique()
)

print("\nNúmero de semanas por escenario:")
print(NUMERO_SEMANAS)

print("\nNúmero de escenarios:")
print(NUMERO_ESCENARIOS)

print("\nPeso estacional sobre aparición:")
print(PESO_ESTACIONAL_APARICION)

print("\nSuavización de tendencia:")
print(SUAVIZACION_TENDENCIA)


media_total_sintetica = (
    comparacion_totales[
        "DemandaTotalSintetica"
    ].mean()
)

std_total_sintetica = (
    comparacion_totales[
        "DemandaTotalSintetica"
    ].std()
)

ratio_global = (
    media_total_sintetica
    / total_historico
)

error_global = (
    ratio_global - 1.0
) * 100

print("\n" + "-" * 75)
print("VALIDACIÓN GLOBAL")
print("-" * 75)

print("\nDemanda total histórica:")
print(total_historico)

print("\nDemanda total sintética media:")
print(media_total_sintetica)

print("\nDesviación total entre escenarios:")
print(std_total_sintetica)

print("\nRatio sintético/histórico:")
print(ratio_global)

print("\nError porcentual global:")
print(error_global, "%")

print("\nDemanda sintética mínima:")
print(
    comparacion_totales[
        "DemandaTotalSintetica"
    ].min()
)

print("\nDemanda sintética máxima:")
print(
    comparacion_totales[
        "DemandaTotalSintetica"
    ].max()
)


print("\n" + "-" * 75)
print("VALIDACIÓN DE APARICIÓN DE DEMANDA")
print("-" * 75)

print("\nError absoluto medio de PctCeros:")

print(
    validacion_media[
        "ErrorPctCeros"
    ]
    .abs()
    .mean()
)

print("\nError absoluto medio de ADI:")

print(
    validacion_media[
        "ErrorADI"
    ]
    .abs()
    .mean()
)


print("\n" + "-" * 75)
print("VALIDACIÓN DE CANTIDADES POSITIVAS")
print("-" * 75)

print("\nError absoluto medio de CV²:")

print(
    validacion_media[
        "ErrorCV2"
    ]
    .abs()
    .mean()
)

print("\nProductos con total sintético medio entre 90 % y 110 %:")

proporcion_productos_aceptables = (
    validacion_media[
        "RatioTotalMedio"
    ]
    .between(
        0.90,
        1.10
    )
    .mean()
)

print(
    proporcion_productos_aceptables
    * 100,
    "%"
)


print("\n" + "-" * 75)
print("COMPROBACIÓN DE NORMALIZACIONES")
print("-" * 75)

comprobacion_perfiles = (
    perfiles_semanales
    .groupby(
        "IdProducto",
        as_index=False
    )
    .agg(
        ProbabilidadBase=(
            "ProbabilidadBase",
            "first"
        ),

        ProbabilidadSemanaMedia=(
            "ProbabilidadSemana",
            "mean"
        ),

        IndiceEstacionalMedio=(
            "IndiceEstacional",
            "mean"
        ),

        FactorTendenciaMedio=(
            "FactorTendencia",
            "mean"
        )
    )
)

comprobacion_perfiles[
    "ErrorProbabilidad"
] = (
    comprobacion_perfiles[
        "ProbabilidadSemanaMedia"
    ]
    - comprobacion_perfiles[
        "ProbabilidadBase"
    ]
)

print("\nMáximo error de probabilidad media:")

print(
    comprobacion_perfiles[
        "ErrorProbabilidad"
    ]
    .abs()
    .max()
)

print("\nMedia de los índices estacionales:")

print(
    comprobacion_perfiles[
        "IndiceEstacionalMedio"
    ]
    .describe()
)

print("\nMedia de los factores de tendencia:")

print(
    comprobacion_perfiles[
        "FactorTendenciaMedio"
    ]
    .describe()
)


print("\n" + "-" * 75)
print("HOJAS GENERADAS")
print("-" * 75)

print("- DemandaSintetica")
print("- DetalleGeneracion")
print("- ValidacionProductos")
print("- ValidacionMedia")
print("- ResumenSemanal")
print("- ComparacionTotales")
print("- IndicesNormalizados")
print("- PerfilesSemanales")
print("- ParametrosHistoricos")
