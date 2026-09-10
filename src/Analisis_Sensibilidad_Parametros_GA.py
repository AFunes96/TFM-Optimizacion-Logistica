from pathlib import Path

import numpy as np
import pandas as pd

import TFM_Optimizacion_Validacion as tfm

BASE_DIR = Path(__file__).resolve().parents[1]

RUTA_PRECIOS = BASE_DIR / "data" / "processed" / "enllaç-v2_precios_medios.xlsx"
RUTA_DEMANDA = BASE_DIR / "data" / "generated" / "demanda_sintetica.xlsx"
RUTA_DISTANCIAS = BASE_DIR / "data" / "processed" / "matriz_distancias_120.xlsx"

CARPETA_SALIDA = BASE_DIR / "results" / "reference"
RUTA_DETALLE = CARPETA_SALIDA / "Sensibilidad_Parametros_GA_Detalle.csv"
RUTA_RESUMEN = CARPETA_SALIDA / "Sensibilidad_Parametros_GA_Resumen.csv"
RUTA_SEMANAS = CARPETA_SALIDA / "Sensibilidad_Parametros_GA_Semanas.csv"

SEMILLA_SELECCION = 123
SEMILLA_GA = 1

PARAMETROS_BASE = {
    "TAMANO_POBLACION": 200,
    "NUMERO_GENERACIONES": 500,
    "PROBABILIDAD_CRUCE": 0.80,
    "PROBABILIDAD_MUTACION": 0.25,
    "PACIENCIA": 100
}

EXPERIMENTOS = {
    "Tamaño de población": (
        "TAMANO_POBLACION",
        [100, 200, 300]
    ),
    "Generaciones máximas": (
        "NUMERO_GENERACIONES",
        [250, 500, 750]
    ),
    "Probabilidad de cruce": (
        "PROBABILIDAD_CRUCE",
        [0.60, 0.80, 1.00]
    ),
    "Probabilidad de mutación": (
        "PROBABILIDAD_MUTACION",
        [0.10, 0.25, 0.40]
    )
}


def configurar_rutas():
    tfm.RUTA_PRECIOS = RUTA_PRECIOS
    tfm.RUTA_DEMANDA = RUTA_DEMANDA
    tfm.RUTA_DISTANCIAS = RUTA_DISTANCIAS
    tfm.CARPETA_SALIDA = CARPETA_SALIDA
    tfm.GENERAR_GRAFICAS = False
    tfm.GENERAR_MAPAS = False
    tfm.EJECUTAR_MILP = False
    tfm.EJECUTAR_SENSIBILIDAD = False
    tfm.N_SEMANAS_BAJA = 10
    tfm.N_SEMANAS_MEDIA = 10
    tfm.N_SEMANAS_ALTA = 10


def seleccionar_cinco_semanas(maestros):
    resumen = tfm.crear_resumen_semanal(
        maestros.demanda_df
    )

    semanas = tfm.seleccionar_semanas_experimento(
        resumen,
        SEMILLA_SELECCION
    )

    semanas = (
        semanas
        .sort_values("DemandaTotalSemana")
        .reset_index(drop=True)
    )

    posiciones = np.rint(
        np.linspace(
            0,
            len(semanas) - 1,
            7
        )[1:-1]
    ).astype(int)

    return (
        semanas
        .iloc[posiciones]
        .copy()
        .reset_index(drop=True)
    )


def restaurar_configuracion_base():
    for parametro, valor in PARAMETROS_BASE.items():
        setattr(
            tfm,
            parametro,
            valor
        )


def ejecutar_estudio(maestros, semanas):
    registros = []

    for nombre, (
        parametro,
        valores
    ) in EXPERIMENTOS.items():

        for valor in valores:
            restaurar_configuracion_base()

            setattr(
                tfm,
                parametro,
                valor
            )

            for _, semana in semanas.iterrows():
                datos = tfm.construir_problema_semana(
                    maestros,
                    semana
                )

                resultado = tfm.ejecutar_ga(
                    datos,
                    semilla_ga=SEMILLA_GA,
                    verbose=False
                )

                registros.append({
                    "Parametro": nombre,
                    "Atributo": parametro,
                    "Valor": valor,
                    "SemillaGA": SEMILLA_GA,
                    "Escenario": int(
                        semana["Escenario"]
                    ),
                    "NumeroSemanaSimulada": int(
                        semana["NumeroSemanaSimulada"]
                    ),
                    "NivelDemanda": str(
                        semana["NivelDemanda"]
                    ),
                    "DemandaTotalSemana": float(
                        semana["DemandaTotalSemana"]
                    ),
                    "CosteTotal": float(
                        resultado.mejor.coste_total
                    ),
                    "TiempoSegundos": float(
                        resultado.tiempo_total_s
                    ),
                    "GeneracionMejor": int(
                        resultado.generacion_mejor
                    ),
                    "GeneracionesEjecutadas": int(
                        resultado.generaciones_ejecutadas
                    ),
                    "Factible": bool(
                        resultado.mejor.factible
                    )
                })

    return pd.DataFrame(registros)


def crear_resumen(detalle):
    claves = [
        "Parametro",
        "Escenario",
        "NumeroSemanaSimulada"
    ]

    detalle = detalle.copy()

    detalle[
        "MejorCosteSemanaParametro"
    ] = (
        detalle
        .groupby(claves)["CosteTotal"]
        .transform("min")
    )

    detalle[
        "DesviacionMejorPct"
    ] = (
        detalle["CosteTotal"]
        / detalle["MejorCosteSemanaParametro"]
        - 1.0
    ) * 100.0

    resumen = (
        detalle
        .groupby(
            [
                "Parametro",
                "Valor"
            ],
            as_index=False
        )
        .agg(
            CosteMedio=(
                "CosteTotal",
                "mean"
            ),
            DesviacionMediaMejorPct=(
                "DesviacionMejorPct",
                "mean"
            ),
            DesviacionMaximaMejorPct=(
                "DesviacionMejorPct",
                "max"
            ),
            TiempoMedioSegundos=(
                "TiempoSegundos",
                "mean"
            ),
            GeneracionesMedias=(
                "GeneracionesEjecutadas",
                "mean"
            ),
            FactibilidadPct=(
                "Factible",
                "mean"
            )
        )
    )

    resumen["FactibilidadPct"] *= 100.0

    return detalle, resumen


def main():
    configurar_rutas()

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    maestros = tfm.cargar_datos_maestros()

    semanas = seleccionar_cinco_semanas(
        maestros
    )

    detalle = ejecutar_estudio(
        maestros,
        semanas
    )

    detalle, resumen = crear_resumen(
        detalle
    )

    detalle.to_csv(
        RUTA_DETALLE,
        index=False,
        encoding="utf-8-sig"
    )

    resumen.to_csv(
        RUTA_RESUMEN,
        index=False,
        encoding="utf-8-sig"
    )

    semanas.to_csv(
        RUTA_SEMANAS,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Semanas utilizadas:")
    print(
        semanas[
            [
                "Escenario",
                "NumeroSemanaSimulada",
                "NivelDemanda",
                "DemandaTotalSemana"
            ]
        ].to_string(index=False)
    )

    print()
    print("Resumen de sensibilidad:")
    print(
        resumen.to_string(index=False)
    )

    print()
    print("Resultados guardados en:")
    print(RUTA_RESUMEN)


if __name__ == "__main__":
    main()
