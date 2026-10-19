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

El entorno (`src/environment.py`), el agente DQN (`src/agent.py`) y el
entrenamiento (`src/train.py`, `configs/dqn.toml`) están implementados. La
evaluación final y los baselines (`src/evaluate.py`, `src/baselines.py`) aún
son scaffolds.

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

## Entrenamiento del agente DQN

El agente es un DQN de Stable-Baselines3 (`src/agent.py`). Todos los
hiperparámetros, la recompensa a entrenar, las semillas y el criterio de
parada están en `configs/dqn.toml`; `src/train.py` no tiene hiperparámetros en
el código.

Requisitos: haber generado `data/binance-BTCUSDT-1h-interpolated.pkl`
(`python -m src.prepare_data`).

```bash
# Entrena la recompensa configurada con las semillas 42, 43 y 44
python -m src.train

# Entrena otra versión de la recompensa registrada en src/rewards.py
python -m src.train --reward v2_nombre

# Corrida corta para probar el pipeline
python -m src.train --seeds 42 --timesteps 10000 --eval-freq 5000 --output-dir runs_prueba
```

Protocolo:

1. `train.py` construye solo los entornos de entrenamiento y validación; el
   held-out se descarta explícitamente y nunca se carga.
2. El agente entrena con episodios de 1.024 horas que empiezan en una vela
   aleatoria del periodo de entrenamiento (reproducible con la semilla).
3. Cada `eval_freq` pasos se recorre la validación completa con la política
   greedy; si mejora la métrica de validación se guarda `best_model.zip`.
4. Se detiene al llegar a `total_timesteps` o tras `patience` evaluaciones
   sin mejora.
5. Se repite con cada semilla; `summary.csv` reporta cada semilla y la media
   y desviación entre semillas.

Salidas en `runs/<recompensa>/`: `summary.csv`, `learning_curves.png` y una
carpeta por semilla con `best_model.zip`, `last_model.zip`, `monitor.csv`,
`validation_log.csv`, `config.json` y `metadata.json` (commit, versiones,
hash del dataset, tiempo de entrenamiento).

Cuando el equipo elija el agente final usando **solo validación**, se congela
y se hace commit antes de la evaluación en held-out:

```bash
python -m src.train --freeze runs/v1_net_return/seed_42
git add models/final && git commit -m "Agente final congelado"
```

El agente congelado se usa con la interfaz del enunciado:

```python
from src.agent import DQNAgent
agent = DQNAgent.load("models/final/model.zip")
action = agent.act(obs)
```

Sensibilidad a hiperparámetros (cambia uno a la vez respecto a la
configuración inicial, con la recompensa v1, las mismas semillas y midiendo
solo en validación). El learning rate de `configs/dqn.toml` (3e-4) se eligió
con este estudio:

```bash
python -m src.sensitivity run         # ~50 min en CPU (5 variantes × 3 semillas)
python -m src.sensitivity summarize   # runs/sensitivity/sensitivity.md y .png
```

Para probar un valor puntual sin editar `configs/dqn.toml`:
`python -m src.train --set agent.gamma=0.95 --output-dir runs/prueba`.

La justificación del algoritmo, la tabla completa de hiperparámetros, el
protocolo de entrenamiento y los resultados de la recompensa v1 están en
[`docs/informe_persona4.md`](docs/informe_persona4.md).

## Iterar la recompensa (Persona 5)

Todo el ciclo se hace sin modificar el entorno, el agente ni el script de
entrenamiento. Todo se evalúa en **validación**; `src/analysis.py` no permite
usar el held-out.

1. **Definir la nueva versión** en `src/rewards.py` y registrarla en
   `REWARDS` con un nombre nuevo (el archivo trae la lista de variables
   disponibles en `info` y un ejemplo con parámetro).
2. **Entrenarla** con los mismos hiperparámetros y semillas que la v1
   (~10 minutos en CPU para las 3 semillas):

   ```bash
   python -m src.train --reward v2_nombre
   ```

3. **Ver qué hace el agente** (posiciones en el tiempo, curva de capital
   frente a comprar y mantener, operaciones acumuladas y valores Q en cada
   hora):

   ```bash
   python -m src.analysis inspect runs/v2_nombre/seed_42
   # acercar a un periodo para documentar una anomalía
   python -m src.analysis inspect runs/v2_nombre/seed_42 --start 2025-02-15 --end 2025-04-15
   ```

   Archivos en `runs/v2_nombre/seed_42/analysis/`: `trajectory.csv`,
   `behavior.png` y `stats.json`.
4. **Comparar versiones** (media y desviación entre semillas):

   ```bash
   python -m src.analysis compare v1_net_return v2_nombre
   ```

   Genera `runs/comparison/reward_comparison.md` (tabla para el informe),
   `.csv` y `.png`.

Entrene cada versión nueva con `configs/dqn.toml` sin cambios, para que la
diferencia se deba solo a la recompensa. El diagnóstico de la v1 está en
`docs/informe_persona4.md`, sección 4: con la configuración final el agente
pasa ~97 % del tiempo invertido, no supera a comprar y mantener y concentra
sus operaciones en las caídas de alta volatilidad (febrero–abril de 2025).

## Evaluación y baselines

```bash
python -m src.evaluate
python -m src.baselines
```

Estos módulos están a cargo de la Persona 6 y aún no están implementados.

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

Esta receta cubre descarga/preparación, validación, partición y EDA.

6. **Correr las pruebas:**

   ```bash
   python -m pytest
   ```

7. **Reproducir el entrenamiento** (ver "Entrenamiento del agente DQN"):

   ```bash
   python -m src.train
   ```

   Las semillas de entrenamiento (42, 43 y 44) están en `configs/dqn.toml`.
   `src.agent.set_global_seed` fija Python, NumPy y PyTorch (un hilo y
   algoritmos deterministas), Stable-Baselines3 recibe la misma semilla para la
   exploración y el entorno la usa para los inicios aleatorios de episodio. En
   la misma máquina y con las versiones fijadas, la misma semilla produce
   exactamente los mismos pesos (`tests/test_agent.py`). Entre sistemas
   operativos o CPU distintos pueden aparecer diferencias en los últimos
   decimales.

La evaluación final y los baselines (Persona 6) todavía no están
implementados.

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
