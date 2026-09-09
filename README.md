# TFM - Optimización Logística

Repositorio asociado al Trabajo Fin de Máster sobre análisis y generación de demanda sintética y optimización de una red logística de aprovisionamiento mediante algoritmos genéticos y comparación con un modelo MILP.

## Objetivo

El proyecto implementa un flujo completo formado por:

1. limpieza y preparación de la demanda histórica;
2. análisis estadístico de la demanda;
3. generación de escenarios de demanda sintética;
4. validación estadística del generador;
5. preparación de precios y matrices de distancias;
6. optimización de la selección de productores y del orden de visita mediante un algoritmo genético;
7. comparación del GA con un baseline y con una formulación MILP resuelta mediante HiGHS.

## Estructura del repositorio

```text
TFM-Optimizacion-Logistica/
├── README.md
├── requirements.txt
├── seeds.json
├── .gitignore
├── src/
├── data/
│   ├── raw/
│   ├── processed/
│   └── generated/
└── results/
    └── reference/
```

### `src/`

Contiene el código fuente del proyecto:

- `Demanda_Cleaner.py`: completa el histórico semanal y rellena semanas sin demanda.
- `Analisis_Demanda.py`: calcula métricas de demanda, clasificación SBC, tendencia, estacionalidad y clasificación ABC.
- `Generador_Demanda.py`: genera escenarios semanales de demanda sintética.
- `TestdeValidacion_Generador_Demanda_Sintetica.py`: compara demanda histórica y sintética mediante métricas y tests estadísticos.
- `Precios_Productor.py`: obtiene precios medios por producto y productor.
- `DataLoader.py`: utilidades de carga y transformación geográfica.
- `Generador_Matriz_Costes_v0.py`: genera matrices de tiempos y distancias mediante OSRM local.
- `Mapa_Productores_HUB_clientes_v0.py`: genera un mapa de productores, HUB y clientes.
- `Algoritmo_Genetico_Rutas_v0.py`: versión base del algoritmo genético.
- `TFM_Optimizacion_Validacion.py`: experimento final de optimización y validación, incluyendo GA, baseline y MILP.

## Entorno

El proyecto se ha desarrollado con Python 3.13.2.

Instalación de dependencias:

```bash
python -m pip install -r requirements.txt
```

## Reproducibilidad

Las semillas utilizadas se encuentran centralizadas y documentadas en `seeds.json`.

Configuración principal de los experimentos finales:

- generación de demanda sintética: semilla `123`;
- número de escenarios sintéticos: `8`;
- horizonte generado: `104` semanas;
- selección de semanas del experimento: semilla `123`;
- ejecuciones del GA: semillas `1` a `10`;
- semilla del MILP/HiGHS: `123`;
- análisis de sensibilidad: semillas `1`, `2` y `3`.

Estas semillas coinciden con las constantes definidas en los scripts utilizados para los experimentos.

## Datos incluidos

Se incluyen los datos procesados e intermedios necesarios para reproducir el flujo principal:

- `data/processed/Demanda_completa.xlsx`
- `data/processed/matriz_analisis_demanda.xlsx`
- `data/processed/enllaç-v2_precios_medios.xlsx`
- `data/processed/matriz_distancias_120.xlsx`
- `data/processed/matriz_tiempos_120.xlsx`
- `data/generated/demanda_sintetica.xlsx`

Los datos brutos `Demanda.xlsx` y `enllaç-v2.xlsx` no están incluidos en este paquete. Los scripts que dependen de ellos se conservan para documentar el preprocesamiento original. En `data/raw/README.md` se detallan estos requisitos.

## Ejecución para reproducir el flujo principal

Desde la carpeta raíz del repositorio:

```bash
python src/Analisis_Demanda.py
python src/Generador_Demanda.py
python src/TestdeValidacion_Generador_Demanda_Sintetica.py
python src/TFM_Optimizacion_Validacion.py
```

`Analisis_Demanda.py` parte de `data/processed/Demanda_completa.xlsx`, por lo que no es necesario disponer del fichero bruto `Demanda.xlsx` para repetir el análisis.

El generador produce:

```text
data/generated/demanda_sintetica.xlsx
```

La validación del generador produce:

```text
results/validacion_generador.xlsx
```

El experimento final produce sus resultados en:

```text
results/Resultados_Validacion_GA/
```

## Resultados de referencia

La carpeta `results/reference/` contiene las salidas utilizadas como referencia en la preparación del repositorio:

- `validacion_generador.xlsx`
- `Resultados_Validacion_GA.xlsx`

Permiten comparar una nueva ejecución con los resultados previamente obtenidos.

## OSRM

La optimización final utiliza directamente la matriz de distancias incluida en el repositorio y no necesita un servidor OSRM activo.

Solo es necesario OSRM si se desea regenerar las matrices desde las coordenadas originales. En ese caso, `Generador_Matriz_Costes_v0.py` espera un servidor OSRM local en:

```text
http://localhost:5000
```

## Rutas de archivos

Los scripts del repositorio utilizan rutas relativas calculadas a partir de la ubicación del propio repositorio. No contienen dependencias de rutas locales del tipo `C:\Users\...`.

## Versión utilizada en el TFM

La memoria debe citar tanto la URL pública de este repositorio como el identificador del commit correspondiente a la versión utilizada para generar los resultados.

Repositorio:

```text
https://github.com/AFunes96/TFM-Optimizacion-Logistica
```

Después de realizar la subida definitiva, el hash del commit puede obtenerse desde GitHub o mediante:

```bash
git rev-parse HEAD
```

Ese hash debe incorporarse a la memoria para fijar de forma inequívoca la versión reproducible del código.
