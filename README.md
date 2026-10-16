# Proyecto RL para inversión

Scaffold reproducible para descargar, validar y explorar velas horarias
BTC/USDT de Binance.

## Estructura

```text
project/
├── data/
├── src/
│   ├── environment.py
│   ├── agent.py
│   ├── train.py
│   ├── evaluate.py
│   ├── baselines.py
│   ├── data.py
│   ├── download_data.py
│   └── prepare_data.py
├── notebooks/
│   └── eda.ipynb
├── configs/
│   └── project.toml
├── pyproject.toml
└── README.md
```

Los módulos de entorno y agente, entrenamiento, evaluación y baselines son
scaffolds explícitos para una etapa posterior y aún no implementan RL.

## Instalación

Requiere Python 3.11, 3.12 o 3.13:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Las versiones directas están fijadas en `pyproject.toml`.

## Dataset

La fuente única es BTC/USDT spot de Binance, timeframe `1h`, desde el
2020-01-01 hasta el 2026-10-01 (fecha final exclusiva). La fuente de descarga
es [Gym Trading Environment](https://gym-trading-env.readthedocs.io/en/latest/download.html).
El dataset normalizado se guarda en
`data/binance-BTCUSDT-1h.pkl`, con las columnas `timestamp, open, high, low,
close, volume`.

Se usa una ventana de casi siete años para observar distintos regímenes del
mercado reciente y contar con suficientes datos para reservar un año completo
para validación y otro para held-out, manteniendo al menos 70 % de las filas
en entrenamiento. En esta primera formulación se limita el análisis a un
activo y una fuente (BTC/USDT de Binance): así se reduce la heterogeneidad
entre instrumentos y exchanges; no se elimina el sesgo específico de BTC ni
el riesgo propio de Binance.

Para descargarlo de nuevo según `configs/project.toml`:

```bash
python -m src.download_data
```

El script descarga en un directorio temporal, valida el archivo, limita las
filas al rango solicitado y guarda un solo pickle de salida. Si la API no
genera el archivo esperado o no hay datos dentro del período, informa el error
y no crea un resultado vacío. Los pickles solo deben abrirse si provienen de
una fuente confiable.

## Validación y particiones

`src/data.py` convierte timestamps a UTC y comprueba columnas requeridas,
orden cronológico, duplicados, nulos, velas faltantes y consistencia de OHLCV.
Elimina duplicados de forma determinista y filas con campos obligatorios
nulos. Los valores OHLC inválidos causan un error explícito. El dataset
descargado se mantiene sin modificar; las ausencias se informan por separado.

Los cortes semiabiertos están fijados en `configs/project.toml` y no se
seleccionan mirando el desempeño del agente:

| Partición | Intervalo UTC |
|---|---|
| Entrenamiento | 2020-01-01 a 2024-10-01 |
| Validación | 2024-10-01 a 2025-10-01 |
| Held-out | 2025-10-01 a 2026-10-01 |

Validación y held-out cubren un año calendario cada una. Con la descarga
actual hay 41.602 filas de entrenamiento de 59.122 observadas, equivalente a
70,37 %. Los cortes están versionados en `configs/project.toml`;
`src.data.split_by_dates` los aplica como intervalos `[inicio, fin)` sin
solapamiento. La última celda del EDA comprueba la proporción y la duración
mínima de las particiones.

No use validación ni held-out para ajustar el entrenamiento; held-out es la
evaluación final.

## Análisis exploratorio

Descargue los datos y abra `notebooks/eda.ipynb`. El notebook analiza un solo
activo y fuente:

- lista todos los timestamps ausentes del calendario completo configurado,
  incluidos los que preceden a la primera vela observada;
- agrupa timestamps consecutivos en huecos, resume su duración y muestra los
  cierres observados antes y después y el movimiento entre esos extremos para
  explicar la decisión de tratamiento;
- presenta evolución del precio, retornos, histograma y colas, asimetría y
  curtosis, volatilidad alta/baja, tendencias, drawdown, volumen e indicadores;
- compara retorno medio, volatilidad, tendencia y drawdown entre entrenamiento,
  validación y held-out para discutir no-estacionariedad;
- incluye espacios Markdown debajo de cada sección para redactar la
  interpretación de tablas y gráficos.

Se muestran SMA de 24 y 168 horas, RSI de 14 horas y volatilidad móvil de 24
horas. Medias, RSI y volatilidad se desplazan una vela. Los retornos que
cruzan un timestamp faltante se excluyen de los resúmenes para no tratarlos
como retornos horarios. El EDA es descriptivo y no usa held-out para escoger
parámetros.

### Tratamiento de timestamps faltantes

En la descarga actual faltan 38 timestamps (0,06423 % del calendario), de
los cuales seis preceden a la primera vela observada (2020-01-01 06:00 UTC) y
32 son huecos internos distribuidos en 15 bloques de hasta cinco horas. Los
huecos iniciales quedan sin imputar: no hay un cierre anterior y copiar el
primer cierre conocido hacia atrás introduciría información futura.

Para los huecos internos de hasta cinco horas se usa **forward-fill causal
plano**: cada vela sintética recibe `open = high = low = close` igual al último
cierre observado y `volume = 0`. El indicador booleano
`is_interpolated=True` permite distinguirlas de las velas reales. Esta
aproximación evita consultar el cierre futuro, conserva una cuadrícula horaria
regular y es sencilla de auditar; no recupera la trayectoria que realmente
ocurrió. En particular, subestima la volatilidad durante el hueco y puede
concentrar el movimiento no observado en la vela real de reanudación. Por eso
se excluyen de los resúmenes los retornos que cruzan interrupciones, y el
dataset original permanece intacto.

Para crear el dataset derivado
`data/binance-BTCUSDT-1h-interpolated.pkl` después de descargar el original:

```bash
python -m src.prepare_data
```

El máximo de cinco horas está versionado como
`preprocessing.max_gap_hours` en `configs/project.toml`, acorde con la mayor
interrupción interna encontrada. Si una nueva descarga deja huecos mayores,
el proceso falla para que se revise la estrategia en lugar de inventar datos
sin autorización. No use velas sintéticas como observaciones reales al evaluar
riesgo o rendimiento financiero.

## Entrenamiento, evaluación y baselines

Los puntos de entrada previstos para una etapa futura son:

```bash
python -m src.train
python -m src.evaluate
python -m src.baselines
```

Por ahora los módulos reportan explícitamente que esa funcionalidad aún no
está implementada.

## Reproducibilidad

La semilla global `42`, el activo, la frecuencia, el rango y los cortes están
en `configs/project.toml`. El downloader y el notebook leen esa configuración.
Conserve el archivo de datos descargado y las versiones fijadas de
`pyproject.toml` para reproducir el EDA.

### Replicar lo realizado hasta ahora

Los comandos siguientes reproducen la preparación de datos, la partición y el
análisis exploratorio actual. Ejecútelos desde la raíz del proyecto.

1. **Crear el entorno e instalar las dependencias fijadas:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

2. **Obtener el dataset.** Para reproducir exactamente el EDA registrado,
   conserva `data/binance-BTCUSDT-1h.pkl`. Si todavía no existe o quieres
   volver a solicitar los datos a Binance, ejecuta:

   ```bash
   python -m src.download_data
   ```

   La descarga depende de la disponibilidad del API; los datos podrían cambiar
   o contener diferente cobertura con el tiempo. Por eso, para reproducibilidad
   estricta, guarda una copia del archivo usado y no dependas de volver a
   consultar el API.

3. **Crear el derivado imputado** (no sobrescribe el dataset original):

   ```bash
   python -m src.prepare_data
   ```

4. **Revisar carga, calidad y particiones** (opcional, desde una sesión Python):

   ```bash
   python - <<'PY'
   from src.data import load_ohlcv, load_split_config, split_by_dates

   data, quality = load_ohlcv("data/binance-BTCUSDT-1h.pkl")
   print(quality.to_dict())

   partitions = split_by_dates(data, load_split_config())
   for name, period in partitions.items():
       print(name, len(period), period["timestamp"].min(), period["timestamp"].max())
   PY
   ```

   En el dataset actual, las proporciones por filas son 70,37 % para
   entrenamiento y 14,82 % para cada partición restante.

5. **Reproducir el EDA y la auditoría de huecos:**

   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

   Selecciona el kernel del entorno `.venv` y ejecuta **Restart Kernel and Run
   All Cells**. El notebook vuelve a calcular la lista completa de timestamps
   faltantes, los bloques de huecos, las métricas descriptivas y las gráficas.
   Aplica la política causal en memoria para visualizar el efecto, pero no
   modifica el dataset original. El archivo derivado persistente se genera con
   `python -m src.prepare_data`.

Los cortes temporales y el período solicitado se leen de
`configs/project.toml`; la semilla declarada es `42`. En el EDA actual la
semilla se muestra como parte de la configuración, pero no interviene en los
cálculos deterministas presentados.

Esta receta cubre únicamente descarga/preparación, validación, partición y
EDA. El agente RL, el entrenamiento, la evaluación de políticas y los
baselines todavía no están implementados; sus comandos indicados en este
README son puntos de entrada futuros, no resultados reproducibles de esta
etapa.

Ejemplo de uso del loader:

```python
from src.data import load_ohlcv, load_split_config, split_by_dates

data, quality = load_ohlcv("data/binance-BTCUSDT-1h.pkl")
print(quality.to_dict())
partitions = split_by_dates(data, load_split_config())
train = partitions["train"]
validation = partitions["validation"]
held_out = partitions["held_out"]
```
