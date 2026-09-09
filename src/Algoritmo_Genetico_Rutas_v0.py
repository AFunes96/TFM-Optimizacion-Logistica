import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RUTA_PRECIOS = BASE_DIR / "data" / "processed" / "enllaç-v2_precios_medios.xlsx"

RUTA_DEMANDA = BASE_DIR / "data" / "generated" / "demanda_sintetica.xlsx"

RUTA_DISTANCIAS = BASE_DIR / "data" / "processed" / "matriz_distancias_120.xlsx"

RUTA_SALIDA = BASE_DIR / "results" / "resultado_GA.xlsx"


ESCENARIO_OBJETIVO = None
SEMANA_OBJETIVO = None


SEMILLA = 123


TAMANO_POBLACION = 1000
NUMERO_GENERACIONES = 500

PROBABILIDAD_CRUCE = 0.80
PROBABILIDAD_MUTACION = 0.25


PORCENTAJE_ELITISMO = 0.05


PACIENCIA = 100


MAX_EXTRAS_INICIALES = 3


NUMERO_MUTACIONES = [1, 2, 3, 4]
PROB_NUMERO_MUTACIONES = [0.50, 0.30, 0.15, 0.05]


TIPOS_MUTACION = [
    "sustitucion",
    "inversion",
    "swap",
    "insercion",
    "anadir",
    "eliminar",
]

PROB_TIPOS_MUTACION = [
    0.30,
    0.25,
    0.15,
    0.10,
    0.10,
    0.10,
]


COSTE_POR_KM = 1.2747


PENALIZACION_PRODUCTO_NO_CUBIERTO = 1_000_000_000.0
PENALIZACION_UNIDAD_NO_CUBIERTA = 1_000_000.0


HUB_ID = None


@dataclass
class Individuo:

    orden: list
    k: int

    coste_total: float = math.inf
    coste_compra: float = math.inf
    coste_transporte: float = math.inf
    distancia_km: float = math.inf
    factible: bool = False
    demanda_no_cubierta: float = 0.0
    asignacion: list = field(default_factory=list)

    def activos(self):
        return self.orden[:self.k]


@dataclass
class DatosProblema:
    demanda: dict
    productos_demandados: set
    productores: list
    hub: str
    distancias: pd.DataFrame
    ofertas_por_producto: dict
    productos_por_productor: dict
    escenario: int
    semana: int
    fecha: object
    demanda_semana_df: pd.DataFrame


def normalizar_id(valor):

    texto = str(valor).strip()

    if texto.endswith(".0"):
        texto = texto[:-2]

    return texto


def cargar_semana_demanda(ruta, rng):

    df = pd.read_excel(
        ruta,
        sheet_name="DemandaSintetica"
    )

    columnas_necesarias = {
        "Escenario",
        "NumeroSemanaSimulada",
        "IdProducto",
        "Demanda",
    }

    faltan = columnas_necesarias - set(df.columns)

    if faltan:
        raise ValueError(
            "Faltan columnas en DemandaSintetica: "
            + ", ".join(sorted(faltan))
        )

    df["IdProducto"] = df["IdProducto"].map(normalizar_id)
    df["Demanda"] = pd.to_numeric(df["Demanda"], errors="coerce").fillna(0.0)
    df["Escenario"] = pd.to_numeric(df["Escenario"], errors="raise").astype(int)
    df["NumeroSemanaSimulada"] = pd.to_numeric(
        df["NumeroSemanaSimulada"],
        errors="raise"
    ).astype(int)

    pares = (
        df[["Escenario", "NumeroSemanaSimulada"]]
        .drop_duplicates()
        .sort_values(["Escenario", "NumeroSemanaSimulada"])
    )

    if ESCENARIO_OBJETIVO is not None:
        pares = pares[pares["Escenario"] == int(ESCENARIO_OBJETIVO)]

    if SEMANA_OBJETIVO is not None:
        pares = pares[
            pares["NumeroSemanaSimulada"] == int(SEMANA_OBJETIVO)
        ]

    if pares.empty:
        raise ValueError(
            "No existe ninguna pareja Escenario/Semana compatible "
            "con la selección indicada."
        )

    opciones = list(
        pares[["Escenario", "NumeroSemanaSimulada"]]
        .itertuples(index=False, name=None)
    )

    escenario, semana = rng.choice(opciones)

    semana_df = df[
        (df["Escenario"] == escenario)
        & (df["NumeroSemanaSimulada"] == semana)
    ].copy()


    columnas_agrupacion = ["IdProducto"]

    if "Fecha" in semana_df.columns:
        fecha = semana_df["Fecha"].iloc[0]
    else:
        fecha = None

    demanda_agrupada = (
        semana_df
        .groupby(columnas_agrupacion, as_index=False)["Demanda"]
        .sum()
    )

    demanda_positiva = demanda_agrupada[demanda_agrupada["Demanda"] > 0].copy()

    demanda = dict(
        zip(
            demanda_positiva["IdProducto"],
            demanda_positiva["Demanda"].astype(float)
        )
    )

    if not demanda:
        raise ValueError(
            "La semana seleccionada tiene demanda total igual a cero. "
            "Selecciona otra semana."
        )

    return demanda, escenario, semana, fecha, semana_df


def cargar_precios(ruta):

    df = pd.read_excel(
        ruta,
        sheet_name="Productos"
    )

    columnas_necesarias = {
        "IdProducto",
        "IdProductor",
        "PrecioPorUnidad",
    }

    faltan = columnas_necesarias - set(df.columns)

    if faltan:
        raise ValueError(
            "Faltan columnas en la hoja Productos: "
            + ", ".join(sorted(faltan))
        )

    df["IdProducto"] = df["IdProducto"].map(normalizar_id)
    df["IdProductor"] = df["IdProductor"].map(normalizar_id)
    df["PrecioPorUnidad"] = pd.to_numeric(
        df["PrecioPorUnidad"],
        errors="coerce"
    )

    df = df.dropna(subset=["PrecioPorUnidad"])
    df = df[df["PrecioPorUnidad"] >= 0].copy()


    df = (
        df
        .groupby(["IdProducto", "IdProductor"], as_index=False)
        .agg(PrecioPorUnidad=("PrecioPorUnidad", "mean"))
    )

    return df


def cargar_matriz_distancias(ruta):

    matriz = pd.read_excel(
        ruta,
        index_col=0
    )

    matriz.index = [normalizar_id(x) for x in matriz.index]
    matriz.columns = [normalizar_id(x) for x in matriz.columns]

    if matriz.index.duplicated().any():
        raise ValueError("Hay IDs duplicados en las filas de la matriz.")

    if pd.Index(matriz.columns).duplicated().any():
        raise ValueError("Hay IDs duplicados en las columnas de la matriz.")

    if set(matriz.index) != set(matriz.columns):
        raise ValueError(
            "Los IDs de filas y columnas de la matriz de distancias "
            "no coinciden."
        )


    matriz = matriz.loc[matriz.index, matriz.index]
    matriz = matriz.apply(pd.to_numeric, errors="coerce")

    return matriz


def preparar_datos():

    rng_semana = random.Random(SEMILLA)

    (
        demanda,
        escenario,
        semana,
        fecha,
        demanda_semana_df,
    ) = cargar_semana_demanda(
        RUTA_DEMANDA,
        rng_semana
    )

    precios = cargar_precios(RUTA_PRECIOS)
    distancias = cargar_matriz_distancias(RUTA_DISTANCIAS)

    if HUB_ID is None:
        hub = normalizar_id(distancias.index[-1])
    else:
        hub = normalizar_id(HUB_ID)

    if hub not in distancias.index:
        raise ValueError(
            f"El HUB '{hub}' no aparece en la matriz de distancias."
        )

    productos_demandados = set(demanda.keys())


    precios = precios[
        precios["IdProducto"].isin(productos_demandados)
    ].copy()

    productores_con_oferta = set(precios["IdProductor"])
    nodos_matriz = set(distancias.index)

    fuera_matriz = productores_con_oferta - nodos_matriz

    if fuera_matriz:
        print(
            "AVISO: se ignoran productores con precio que no aparecen "
            "en la matriz de distancias:"
        )
        print(sorted(fuera_matriz))


    productores = [
        nodo
        for nodo in distancias.index
        if nodo != hub and nodo in productores_con_oferta
    ]

    if not productores:
        raise ValueError(
            "No hay productores utilizables después de cruzar precios "
            "y matriz de distancias."
        )

    precios = precios[precios["IdProductor"].isin(productores)].copy()

    ofertas_por_producto = {}

    for fila in precios.itertuples(index=False):
        ofertas_por_producto.setdefault(
            fila.IdProducto,
            {}
        )[fila.IdProductor] = float(fila.PrecioPorUnidad)

    productos_sin_oferta = sorted(
        p
        for p in productos_demandados
        if p not in ofertas_por_producto
        or len(ofertas_por_producto[p]) == 0
    )

    if productos_sin_oferta:
        raise ValueError(
            "La demanda de la semana no puede satisfacerse porque no "
            "hay productor con precio y distancia para estos productos:\n"
            + ", ".join(productos_sin_oferta)
        )

    productos_por_productor = {
        productor: set()
        for productor in productores
    }

    for producto, ofertas in ofertas_por_producto.items():
        for productor in ofertas:
            productos_por_productor[productor].add(producto)

    return DatosProblema(
        demanda=demanda,
        productos_demandados=productos_demandados,
        productores=productores,
        hub=hub,
        distancias=distancias,
        ofertas_por_producto=ofertas_por_producto,
        productos_por_productor=productos_por_productor,
        escenario=escenario,
        semana=semana,
        fecha=fecha,
        demanda_semana_df=demanda_semana_df,
    )


def productos_cubiertos(productores_activos, datos):

    cubiertos = set()

    for productor in productores_activos:
        cubiertos.update(
            datos.productos_por_productor.get(productor, set())
        )

    return cubiertos & datos.productos_demandados


def productos_faltantes(individuo, datos):

    cubiertos = productos_cubiertos(
        individuo.activos(),
        datos
    )

    return datos.productos_demandados - cubiertos


def reparar_individuo(individuo, datos, rng):

    individuo.k = max(1, min(individuo.k, len(individuo.orden)))

    faltantes = productos_faltantes(individuo, datos)

    while faltantes:

        candidatos = [
            productor
            for productor in individuo.orden[individuo.k:]
            if datos.productos_por_productor.get(productor, set())
            & faltantes
        ]

        if not candidatos:

            return individuo

        elegido = rng.choice(candidatos)


        posicion_antigua = individuo.orden.index(elegido)
        individuo.orden.pop(posicion_antigua)


        posicion_nueva = rng.randint(0, individuo.k)
        individuo.orden.insert(posicion_nueva, elegido)
        individuo.k += 1

        faltantes = productos_faltantes(individuo, datos)

    return individuo


def generar_individuo_inicial(datos, rng):

    faltantes = set(datos.productos_demandados)
    disponibles = list(datos.productores)
    activos = []

    while faltantes:

        candidatos = [
            productor
            for productor in disponibles
            if datos.productos_por_productor.get(productor, set())
            & faltantes
        ]

        if not candidatos:
            raise RuntimeError(
                "No se puede construir una solución factible con los "
                "productores disponibles."
            )

        elegido = rng.choice(candidatos)
        disponibles.remove(elegido)


        posicion = rng.randint(0, len(activos))
        activos.insert(posicion, elegido)

        faltantes -= datos.productos_por_productor[elegido]


    max_extras = min(
        MAX_EXTRAS_INICIALES,
        len(disponibles)
    )

    numero_extras = rng.randint(0, max_extras)

    if numero_extras > 0:
        extras = rng.sample(disponibles, numero_extras)

        for productor in extras:
            disponibles.remove(productor)
            posicion = rng.randint(0, len(activos))
            activos.insert(posicion, productor)

    rng.shuffle(disponibles)

    orden = activos + disponibles

    return Individuo(
        orden=orden,
        k=len(activos)
    )


def crear_poblacion_inicial(datos, rng):

    poblacion = []

    for _ in range(TAMANO_POBLACION):
        individuo = generar_individuo_inicial(datos, rng)
        individuo = reparar_individuo(individuo, datos, rng)
        evaluar_individuo(individuo, datos)

        if not individuo.factible:
            raise RuntimeError(
                "Se ha generado un individuo inicial no factible."
            )

        poblacion.append(individuo)

    return poblacion


def distancia_ruta_metros(productores_activos, datos):

    ruta = [datos.hub] + list(productores_activos) + [datos.hub]
    distancia = 0.0

    for origen, destino in zip(ruta[:-1], ruta[1:]):
        valor = datos.distancias.loc[origen, destino]

        if pd.isna(valor) or not np.isfinite(valor):
            return math.inf

        distancia += float(valor)

    return distancia


def evaluar_individuo(individuo, datos):

    activos = individuo.activos()

    coste_compra = 0.0
    demanda_no_cubierta = 0.0
    productos_no_cubiertos = 0
    asignacion = []

    for producto, cantidad in datos.demanda.items():

        ofertas = datos.ofertas_por_producto.get(producto, {})

        candidatos = [
            productor
            for productor in activos
            if productor in ofertas
        ]

        if not candidatos:
            productos_no_cubiertos += 1
            demanda_no_cubierta += float(cantidad)
            continue


        mejor_productor = min(
            candidatos,
            key=lambda productor: ofertas[productor]
        )

        precio = float(ofertas[mejor_productor])
        coste_producto = float(cantidad) * precio

        coste_compra += coste_producto

        asignacion.append({
            "IdProducto": producto,
            "Demanda": float(cantidad),
            "IdProductor": mejor_productor,
            "PrecioUnidad": precio,
            "CosteProducto": coste_producto,
        })

    distancia_m = distancia_ruta_metros(activos, datos)

    if math.isinf(distancia_m):
        coste_transporte = PENALIZACION_PRODUCTO_NO_CUBIERTO
        distancia_km = math.inf
    else:
        distancia_km = distancia_m / 1000.0
        coste_transporte = distancia_km * COSTE_POR_KM

    penalizacion = (
        productos_no_cubiertos * PENALIZACION_PRODUCTO_NO_CUBIERTO
        + demanda_no_cubierta * PENALIZACION_UNIDAD_NO_CUBIERTA
    )

    individuo.coste_compra = coste_compra
    individuo.coste_transporte = coste_transporte
    individuo.distancia_km = distancia_km
    individuo.demanda_no_cubierta = demanda_no_cubierta
    individuo.factible = (
        productos_no_cubiertos == 0
        and not math.isinf(distancia_m)
    )
    individuo.asignacion = asignacion
    individuo.coste_total = (
        coste_compra
        + coste_transporte
        + penalizacion
    )

    return individuo.coste_total


def seleccionar_ruleta(poblacion, rng):

    ordenados = sorted(
        poblacion,
        key=lambda ind: ind.coste_total
    )

    n = len(ordenados)
    pesos = list(range(n, 0, -1))

    return rng.choices(
        ordenados,
        weights=pesos,
        k=1
    )[0]


def ox_una_direccion(orden_a, orden_b, rng):

    n = len(orden_a)

    if n < 2:
        return orden_a.copy()

    corte_1, corte_2 = sorted(rng.sample(range(n), 2))

    hijo = [None] * n
    hijo[corte_1:corte_2 + 1] = orden_a[corte_1:corte_2 + 1]

    genes_segmento = set(hijo[corte_1:corte_2 + 1])


    recorrido_b = (
        orden_b[corte_2 + 1:]
        + orden_b[:corte_2 + 1]
    )

    genes_para_insertar = [
        gen
        for gen in recorrido_b
        if gen not in genes_segmento
    ]

    posiciones_vacias = (
        list(range(corte_2 + 1, n))
        + list(range(0, corte_1))
    )

    for posicion, gen in zip(posiciones_vacias, genes_para_insertar):
        hijo[posicion] = gen

    if any(gen is None for gen in hijo):
        raise RuntimeError("OX ha generado una permutación incompleta.")

    return hijo


def cruzar_ox(padre_1, padre_2, datos, rng):

    orden_hijo_1 = ox_una_direccion(
        padre_1.orden,
        padre_2.orden,
        rng
    )

    orden_hijo_2 = ox_una_direccion(
        padre_2.orden,
        padre_1.orden,
        rng
    )

    k_min = min(padre_1.k, padre_2.k)
    k_max = max(padre_1.k, padre_2.k)

    k_hijo_1 = rng.randint(k_min, k_max)
    k_hijo_2 = rng.randint(k_min, k_max)

    hijo_1 = Individuo(orden_hijo_1, k_hijo_1)
    hijo_2 = Individuo(orden_hijo_2, k_hijo_2)

    reparar_individuo(hijo_1, datos, rng)
    reparar_individuo(hijo_2, datos, rng)

    return hijo_1, hijo_2


def mutacion_swap(individuo, rng):

    if individuo.k < 2:
        return

    i, j = rng.sample(range(individuo.k), 2)
    individuo.orden[i], individuo.orden[j] = (
        individuo.orden[j],
        individuo.orden[i],
    )


def mutacion_inversion(individuo, rng):

    if individuo.k < 2:
        return

    i, j = sorted(rng.sample(range(individuo.k), 2))

    individuo.orden[i:j + 1] = reversed(
        individuo.orden[i:j + 1]
    )


def mutacion_insercion(individuo, rng):

    if individuo.k < 2:
        return

    origen = rng.randrange(individuo.k)
    gen = individuo.orden.pop(origen)


    destino = rng.randrange(individuo.k)
    individuo.orden.insert(destino, gen)


def mutacion_sustitucion(individuo, datos, rng):

    if individuo.k >= len(individuo.orden):
        return

    posicion_activa = rng.randrange(individuo.k)
    saliente = individuo.orden[posicion_activa]

    productos_saliente = (
        datos.productos_por_productor.get(saliente, set())
        & datos.productos_demandados
    )

    candidatos = [
        productor
        for productor in individuo.orden[individuo.k:]
        if datos.productos_por_productor.get(productor, set())
        & productos_saliente
    ]

    if not candidatos:
        candidatos = list(individuo.orden[individuo.k:])

    if not candidatos:
        return

    entrante = rng.choice(candidatos)
    posicion_inactiva = individuo.orden.index(entrante)

    individuo.orden[posicion_activa], individuo.orden[posicion_inactiva] = (
        individuo.orden[posicion_inactiva],
        individuo.orden[posicion_activa],
    )


def mutacion_anadir(individuo, datos, rng):

    if individuo.k >= len(individuo.orden):
        return

    candidatos = [
        productor
        for productor in individuo.orden[individuo.k:]
        if datos.productos_por_productor.get(productor, set())
        & datos.productos_demandados
    ]

    if not candidatos:
        return

    elegido = rng.choice(candidatos)
    posicion_antigua = individuo.orden.index(elegido)
    individuo.orden.pop(posicion_antigua)

    posicion_nueva = rng.randint(0, individuo.k)
    individuo.orden.insert(posicion_nueva, elegido)
    individuo.k += 1


def mutacion_eliminar(individuo, rng):

    if individuo.k <= 1:
        return

    posicion = rng.randrange(individuo.k)
    eliminado = individuo.orden.pop(posicion)

    individuo.k -= 1


    posicion_inactiva = rng.randint(
        individuo.k,
        len(individuo.orden)
    )
    individuo.orden.insert(posicion_inactiva, eliminado)


def mutar(individuo, datos, rng):

    numero_operaciones = rng.choices(
        NUMERO_MUTACIONES,
        weights=PROB_NUMERO_MUTACIONES,
        k=1
    )[0]

    for _ in range(numero_operaciones):

        tipo = rng.choices(
            TIPOS_MUTACION,
            weights=PROB_TIPOS_MUTACION,
            k=1
        )[0]

        if tipo == "sustitucion":
            mutacion_sustitucion(individuo, datos, rng)

        elif tipo == "inversion":
            mutacion_inversion(individuo, rng)

        elif tipo == "swap":
            mutacion_swap(individuo, rng)

        elif tipo == "insercion":
            mutacion_insercion(individuo, rng)

        elif tipo == "anadir":
            mutacion_anadir(individuo, datos, rng)

        elif tipo == "eliminar":
            mutacion_eliminar(individuo, rng)

    reparar_individuo(individuo, datos, rng)

    return individuo


def clonar(individuo):
    return copy.deepcopy(individuo)


def resumen_generacion(poblacion, generacion):
    costes = np.array(
        [ind.coste_total for ind in poblacion],
        dtype=float
    )

    mejor = min(
        poblacion,
        key=lambda ind: ind.coste_total
    )

    return {
        "Generacion": generacion,
        "MejorCosteTotal": mejor.coste_total,
        "CosteMedio": float(np.mean(costes)),
        "CosteMediano": float(np.median(costes)),
        "MejorCosteCompra": mejor.coste_compra,
        "MejorCosteTransporte": mejor.coste_transporte,
        "MejorDistanciaKm": mejor.distancia_km,
        "MejorNumeroProductores": mejor.k,
    }


def ejecutar_ga(datos):
    rng = random.Random(SEMILLA)

    print("\nCreando generación inicial válida...")
    poblacion = crear_poblacion_inicial(datos, rng)

    mejor_global = clonar(
        min(poblacion, key=lambda ind: ind.coste_total)
    )

    historial = [
        resumen_generacion(poblacion, 0)
    ]

    sin_mejora = 0

    print(
        f"Generación 0 | "
        f"Coste={mejor_global.coste_total:.2f} | "
        f"Compra={mejor_global.coste_compra:.2f} | "
        f"Distancia={mejor_global.distancia_km:.2f} km | "
        f"Productores={mejor_global.k}"
    )

    for generacion in range(1, NUMERO_GENERACIONES + 1):

        poblacion_ordenada = sorted(
            poblacion,
            key=lambda ind: ind.coste_total
        )

        numero_elites = max(
            1,
            int(round(TAMANO_POBLACION * PORCENTAJE_ELITISMO))
        )

        nueva_poblacion = [
            clonar(ind)
            for ind in poblacion_ordenada[:numero_elites]
        ]

        while len(nueva_poblacion) < TAMANO_POBLACION:

            padre_1 = seleccionar_ruleta(poblacion, rng)
            padre_2 = seleccionar_ruleta(poblacion, rng)

            if rng.random() < PROBABILIDAD_CRUCE:
                hijo_1, hijo_2 = cruzar_ox(
                    padre_1,
                    padre_2,
                    datos,
                    rng
                )
            else:
                hijo_1 = Individuo(
                    orden=padre_1.orden.copy(),
                    k=padre_1.k
                )
                hijo_2 = Individuo(
                    orden=padre_2.orden.copy(),
                    k=padre_2.k
                )

            hijos = [hijo_1, hijo_2]

            for hijo in hijos:

                if rng.random() < PROBABILIDAD_MUTACION:
                    mutar(hijo, datos, rng)
                else:
                    reparar_individuo(hijo, datos, rng)

                evaluar_individuo(hijo, datos)

                nueva_poblacion.append(hijo)

                if len(nueva_poblacion) >= TAMANO_POBLACION:
                    break

        poblacion = nueva_poblacion

        mejor_generacion = min(
            poblacion,
            key=lambda ind: ind.coste_total
        )

        if mejor_generacion.coste_total < mejor_global.coste_total - 1e-9:
            mejor_global = clonar(mejor_generacion)
            sin_mejora = 0
        else:
            sin_mejora += 1

        historial.append(
            resumen_generacion(poblacion, generacion)
        )

        if generacion == 1 or generacion % 10 == 0:
            print(
                f"Generación {generacion} | "
                f"Coste={mejor_global.coste_total:.2f} | "
                f"Compra={mejor_global.coste_compra:.2f} | "
                f"Distancia={mejor_global.distancia_km:.2f} km | "
                f"Productores={mejor_global.k}"
            )

        if PACIENCIA is not None and sin_mejora >= PACIENCIA:
            print(
                f"\nParada anticipada: {PACIENCIA} generaciones "
                "sin mejorar el mejor coste."
            )
            break

    return mejor_global, pd.DataFrame(historial)


def crear_tabla_ruta(mejor, datos):
    ruta = [datos.hub] + mejor.activos() + [datos.hub]

    filas = []
    acumulada = 0.0

    for numero_tramo, (origen, destino) in enumerate(
        zip(ruta[:-1], ruta[1:]),
        start=1
    ):
        distancia_m = float(
            datos.distancias.loc[origen, destino]
        )
        distancia_km = distancia_m / 1000.0
        acumulada += distancia_km

        filas.append({
            "Tramo": numero_tramo,
            "Origen": origen,
            "Destino": destino,
            "DistanciaKm": distancia_km,
            "DistanciaAcumuladaKm": acumulada,
        })

    return pd.DataFrame(filas)


def crear_tabla_productores(mejor):
    compras = pd.DataFrame(mejor.asignacion)
    activos = mejor.activos()

    if compras.empty:
        resumen_compras = pd.DataFrame(
            columns=[
                "IdProductor",
                "NumeroProductosAsignados",
                "CantidadTotalAsignada",
                "CosteCompraProductor",
            ]
        )
    else:
        resumen_compras = (
            compras
            .groupby("IdProductor", as_index=False)
            .agg(
                NumeroProductosAsignados=("IdProducto", "nunique"),
                CantidadTotalAsignada=("Demanda", "sum"),
                CosteCompraProductor=("CosteProducto", "sum"),
            )
        )

    filas = []

    for orden_visita, productor in enumerate(activos, start=1):

        fila = resumen_compras[
            resumen_compras["IdProductor"] == productor
        ]

        if fila.empty:
            numero_productos = 0
            cantidad = 0.0
            coste = 0.0
        else:
            numero_productos = int(
                fila["NumeroProductosAsignados"].iloc[0]
            )
            cantidad = float(
                fila["CantidadTotalAsignada"].iloc[0]
            )
            coste = float(
                fila["CosteCompraProductor"].iloc[0]
            )

        filas.append({
            "OrdenVisita": orden_visita,
            "IdProductor": productor,
            "NumeroProductosAsignados": numero_productos,
            "CantidadTotalAsignada": cantidad,
            "CosteCompraProductor": coste,
        })

    return pd.DataFrame(filas)


def guardar_resultados(mejor, historial, datos):
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    resumen = pd.DataFrame({
        "Indicador": [
            "Escenario",
            "NumeroSemanaSimulada",
            "Fecha",
            "CostePorKm",
            "CosteTotal",
            "CosteCompra",
            "CosteTransporte",
            "DistanciaTotalKm",
            "NumeroProductoresVisitados",
            "DemandaTotalSemana",
            "NumeroProductosConDemanda",
            "SolucionFactible",
        ],
        "Valor": [
            datos.escenario,
            datos.semana,
            datos.fecha,
            COSTE_POR_KM,
            mejor.coste_total,
            mejor.coste_compra,
            mejor.coste_transporte,
            mejor.distancia_km,
            mejor.k,
            sum(datos.demanda.values()),
            len(datos.productos_demandados),
            mejor.factible,
        ]
    })

    compras = pd.DataFrame(mejor.asignacion)

    if not compras.empty:
        compras = compras.sort_values("IdProducto")

    ruta = crear_tabla_ruta(mejor, datos)
    productores = crear_tabla_productores(mejor)

    demanda_semana = (
        datos.demanda_semana_df[
            [
                columna
                for columna in [
                    "Escenario",
                    "NumeroSemanaSimulada",
                    "Fecha",
                    "AnioISO",
                    "SemanaISO",
                    "IdProducto",
                    "Demanda",
                ]
                if columna in datos.demanda_semana_df.columns
            ]
        ]
        .copy()
    )

    with pd.ExcelWriter(
        RUTA_SALIDA,
        engine="openpyxl"
    ) as writer:

        resumen.to_excel(
            writer,
            sheet_name="Resumen",
            index=False
        )

        compras.to_excel(
            writer,
            sheet_name="Compras",
            index=False
        )

        productores.to_excel(
            writer,
            sheet_name="ProductoresVisitados",
            index=False
        )

        ruta.to_excel(
            writer,
            sheet_name="Ruta",
            index=False
        )

        historial.to_excel(
            writer,
            sheet_name="Evolucion",
            index=False
        )

        demanda_semana.to_excel(
            writer,
            sheet_name="DemandaSemana",
            index=False
        )


def main():

    print("=========================================================")
    print(" ALGORITMO GENÉTICO - PRODUCTORES + RUTA")
    print("=========================================================")

    datos = preparar_datos()

    print("\nSemana seleccionada:")
    print("Escenario:", datos.escenario)
    print("Semana simulada:", datos.semana)
    print("Fecha:", datos.fecha)
    print("Productos con demanda positiva:", len(datos.productos_demandados))
    print("Demanda total:", round(sum(datos.demanda.values()), 2))
    print("Productores utilizables:", len(datos.productores))
    print("HUB:", datos.hub)

    mejor, historial = ejecutar_ga(datos)


    evaluar_individuo(mejor, datos)

    print("\n=========================================================")
    print(" MEJOR SOLUCIÓN")
    print("=========================================================")
    print("Factible:", mejor.factible)
    print("Productores visitados:", mejor.k)
    print("Orden:")
    print("HUB -> " + " -> ".join(mejor.activos()) + " -> HUB")
    print("Coste de compra:", round(mejor.coste_compra, 2))
    print("Distancia total (km):", round(mejor.distancia_km, 2))
    print("Coste de transporte:", round(mejor.coste_transporte, 2))
    print("Coste total:", round(mejor.coste_total, 2))

    guardar_resultados(
        mejor,
        historial,
        datos
    )

    print("\nResultados guardados en:")
    print(RUTA_SALIDA)


if __name__ == "__main__":
    main()
