# Datos brutos

Esta carpeta está reservada para los ficheros de origen:

- `Demanda.xlsx`
- `enllaç-v2.xlsx`

Estos dos ficheros no forman parte de este paquete porque no estaban entre los archivos facilitados para preparar el repositorio.

Los resultados principales del TFM pueden reproducirse a partir de los datos procesados incluidos en `data/processed/` y `data/generated/`.

Los scripts que requieren los datos brutos son:

- `Demanda_Cleaner.py`
- `Precios_Productor.py`
- `Generador_Matriz_Costes_v0.py`
- `Mapa_Productores_HUB_clientes_v0.py`

Para regenerar las matrices de distancias y tiempos, `Generador_Matriz_Costes_v0.py` requiere además un servidor OSRM local disponible en `http://localhost:5000`.
