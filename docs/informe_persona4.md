# Persona 4 — Algoritmo, hiperparámetros y protocolo de entrenamiento

> Borrador de las secciones del informe a cargo de la Persona 4. Los números
> de la sección 4 corresponden a la recompensa v1 (`v1_net_return`) y se deben
> actualizar cuando la Persona 5 entregue la v2.

## 1. Algoritmo utilizado

Se utilizó **Deep Q-Network (DQN)** (Mnih et al., 2015) en la implementación de
Stable-Baselines3. DQN aproxima la función de valor de acción Q(s, a) con una
red neuronal y actúa de forma greedy respecto a ella; la política no se
representa explícitamente sino que se deriva de Q:

π(s) = argmax_a Q(s, a; θ)

La red se entrena minimizando el error de diferencia temporal (TD) contra un
objetivo calculado con una copia congelada de la red (red objetivo θ⁻):

y = r + γ · max_a' Q(s', a'; θ⁻)

Es la extensión con aproximación de funciones del Q-learning tabular
(Sutton y Barto, 2018, cap. 6.5 y 16.5). Las dos piezas que la estabilizan son
el *replay buffer* (rompe la correlación temporal entre transiciones
consecutivas, que en una serie de precios es muy fuerte) y la red objetivo
(evita que el objetivo se mueva en cada actualización).

**Por qué DQN y no PPO.** El espacio de acciones es discreto y pequeño
(A = {0 %, 50 %, 100 %}), que es el caso para el que DQN fue diseñado: produce
un valor Q por acción y elegir la mejor es un argmax. Además DQN es
*off-policy*: reutiliza muchas veces cada transición guardada en el buffer, lo
que aprovecha mejor una serie histórica de tamaño fijo. PPO
(Schulman et al., 2017) también es válido con acciones discretas y suele ser
más estable, pero es *on-policy* (descarta los datos tras cada actualización) y
no aporta una ventaja clara en un problema con solo tres acciones. Queda como
alternativa si DQN resulta inestable entre semillas.

## 2. Arquitectura e hiperparámetros

Todos los valores están en `configs/dqn.toml`; `src/train.py` no contiene
hiperparámetros en el código y copia la configuración efectiva a cada corrida
(`config.json`).

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Algoritmo | DQN (Stable-Baselines3 2.9.0) | Acciones discretas; off-policy |
| Arquitectura de la red | MLP 7 → 64 → 64 → 3, ReLU | 6 indicadores + posición actual de entrada; un valor Q por acción. Red pequeña porque la entrada es de baja dimensión y la señal es ruidosa (una red grande sobreajusta) |
| Learning rate | 1 × 10⁻⁴ (Adam) | Valor estándar de DQN; recompensas del orden de 10⁻³ por paso hacen inestables tasas mayores |
| Batch size | 64 | Estándar; compromiso entre ruido del gradiente y costo |
| Factor de descuento γ | 0,99 | Horizonte efectivo 1/(1−γ) ≈ 100 horas (~4 días): suficiente para que el agente "vea" el efecto de pagar un costo hoy frente al movimiento de los días siguientes |
| Replay buffer | 100.000 transiciones | ≈ 2,4 veces el periodo de entrenamiento (41.467 horas): cubre varios regímenes de mercado |
| Pasos antes de aprender | 5.000 | El buffer se llena primero con experiencia aleatoria |
| Frecuencia de entrenamiento | 1 actualización cada 4 pasos | Estándar de DQN |
| Actualización de la red objetivo | cada 2.000 pasos (copia dura) | Estabiliza el objetivo TD |
| Recorte del gradiente | norma máxima 10 | Valor por defecto de SB3 |
| Exploración | ε-greedy con decaimiento lineal | |
| ε inicial / final | 1,0 / 0,05 | |
| Fracción de decaimiento de ε | 30 % del entrenamiento (90.000 pasos) | |
| Pasos de entrenamiento | 300.000 como máximo | ≈ 7 recorridos del periodo de entrenamiento |
| Longitud de episodio (entrenamiento) | 1.024 horas (~43 días), inicio aleatorio | Episodios cortos y variados en vez de recorrer siempre la misma serie desde el inicio |
| Criterio de parada | 300.000 pasos o 6 evaluaciones de validación seguidas sin mejora (early stopping) | |
| Selección del checkpoint | mejor retorno acumulado en **validación** | Ver sección 3 |
| Semillas | 42, 43, 44 | Al menos tres para medir la variabilidad |
| Capital inicial / costo | 100.000 / 10 bps por unidad negociada | Acuerdos del equipo; iguales a los baselines |

## 3. Protocolo de entrenamiento

1. **Separación temporal.** Entrenamiento: 2020-01-08 a 2024-09-30
   (41.467 horas útiles tras el calentamiento de los indicadores). Validación:
   2024-10-01 a 2025-09-30 (8.760 horas). El held-out (2025-10-01 a
   2026-09-30) **no se construye en `train.py`**: la función que arma los
   entornos lo descarta explícitamente y una prueba automática
   (`tests/test_agent.py`) verifica que el último dato visto durante el
   entrenamiento es anterior al inicio del held-out.
2. **Entrenamiento.** Episodios de 1.024 horas que empiezan en una vela
   aleatoria del periodo de entrenamiento. El agente siempre arranca en
   efectivo (posición 0).
3. **Evaluación periódica en validación.** Cada 20.000 pasos se recorre la
   validación completa con la política greedy (ε = 0) y se registran retorno
   acumulado, máximo drawdown, costos, número de operaciones y porcentaje de
   tiempo en cada posición (`validation_log.csv`). Si el retorno de validación
   mejora, se guarda el checkpoint (`best_model.zip`).
4. **Parada.** Al llegar a 300.000 pasos o tras 6 evaluaciones sin mejora.
5. **Repetición con 3 semillas** y reporte de media y desviación estándar
   entre semillas (`summary.csv`), no solo la mejor corrida.
6. **Congelamiento.** El agente final se elige con validación y se congela
   (`python -m src.train --freeze ...`), lo que guarda el modelo, su
   configuración, el commit y un hash del archivo del modelo en
   `models/final/`. Desde ese momento no se reentrena ni se ajusta nada, y la
   Persona 6 lo evalúa una sola vez en el held-out.

**Reproducibilidad.** `set_global_seed` fija las semillas de Python, NumPy y
PyTorch, limita PyTorch a un hilo y activa algoritmos deterministas;
Stable-Baselines3 recibe la misma semilla para la exploración y el entorno la
usa para los inicios aleatorios. Se verificó que dos corridas con la misma
semilla producen pesos idénticos. Cada corrida guarda el commit de git, las
versiones de las librerías y el hash SHA-256 del dataset (`metadata.json`).

**Limitación del protocolo.** Seleccionar el checkpoint con la validación
convierte a la validación en parte del proceso de ajuste: el retorno de
validación del mejor checkpoint es optimista. Por eso la estimación honesta del
desempeño es la del held-out, que se usa una sola vez.

## 4. Resultados del entrenamiento con la recompensa v1

Se entrenó con la recompensa v1 (retorno neto de costos) y tres semillas,
300.000 pasos cada una (~3,5 minutos por semilla en CPU). Ninguna semilla
activó el early stopping.

![Curvas de aprendizaje v1](figuras/curvas_aprendizaje_v1.png)

**Tabla 1.** Mejor checkpoint de cada semilla en **validación**
(2024-10-01 a 2025-09-30, capital inicial 100.000, costo 10 bps).

| Semilla | Paso del mejor checkpoint | Retorno acumulado | Máximo drawdown | Costos pagados | Operaciones | % tiempo en 0 / 50 / 100 % |
|---|---|---|---|---|---|---|
| 42 | 200.000 | +7,4 % | −42,9 % | 66.112 | 1.084 | 1 / 10 / 89 |
| 43 | 300.000 | −33,1 % | −51,4 % | 93.326 | 1.912 | 3 / 38 / 60 |
| 44 | 260.000 | +24,3 % | −36,8 % | 40.534 | 521 | 1 / 5 / 94 |
| **Media ± desv.** | | **−0,5 % ± 29,5 %** | **−43,7 % ± 7,3 %** | **66.657 ± 26.400** | **1.172 ± 700** | |
| *Ref.: comprar y mantener* | | *+79,3 %* | *−30,9 %* | *100* | *1* | *0 / 0 / 100* |

**Interpretación.**

- **El agente aprende algo en entrenamiento.** La recompensa por episodio
  sube de forma consistente en las tres semillas, de ≈ −0,45 a ≈ −0,08, y
  todavía no se estabiliza a los 300.000 pasos. Aun así sigue siendo negativa:
  incluso en los datos que conoce, el agente pierde dinero por episodio.
- **El problema principal es operar en exceso.** Al inicio del entrenamiento
  la política greedy hace entre 3.800 y 5.300 operaciones en un año de
  validación (8.760 horas), es decir, cambia de posición en casi la mitad de
  las horas. Al final baja a 1.300–1.900 operaciones, pero eso sigue
  significando pagar en costos entre el 40 % y el 93 % del capital inicial.
  Con 10 bps por cambio completo y retornos horarios con una desviación de
  ≈ 0,7 % en entrenamiento y ≈ 0,5 % en validación, cada operación cuesta una fracción considerable del movimiento que
  intenta capturar. **Este es el diagnóstico central de la recompensa v1**
  para la Persona 5: aunque la recompensa sí descuenta el costo, el costo de
  un cambio de posición es pequeño frente al ruido del retorno horario y el
  agente no aprende a evitarlo.
- **Gran variabilidad entre semillas.** El retorno de validación del mejor
  checkpoint va de −33 % a +24 % según la semilla, y dentro de una misma
  corrida salta de una evaluación a otra (por ejemplo, la semilla 44 pasa de
  −67 % a +24 % y de vuelta a −51 % en tres evaluaciones consecutivas). Los
  mejores checkpoints son picos aislados, no mesetas, por lo que la selección
  por validación tiene mucho de suerte. Esto se debe reportar como
  limitación y es la razón para usar varias semillas.
- **Las políticas que mejor funcionan se parecen a comprar y mantener.** Los
  mejores checkpoints (semillas 42 y 44) pasan entre el 89 % y el 94 % del
  tiempo 100 % invertidos. Validación fue un año alcista (+79 % para comprar y
  mantener), así que estar invertido es lo que más paga; las diferencias
  frente a comprar y mantener vienen casi completamente de los costos.
- **Ninguna versión supera a comprar y mantener en validación**, ni en retorno
  ni en drawdown.

**Recomendaciones para la siguiente iteración** (a decidir con la Persona 5,
usando solo validación):

1. Recompensa v2 con penalización adicional por cambiar de posición
   (por ejemplo, `r = retorno_neto − λ · turnover`) o con un costo
   amplificado solo durante el entrenamiento, para que el agente aprenda que
   operar tiene consecuencias.
2. Evaluar si conviene más presupuesto de entrenamiento, porque la curva de
   entrenamiento no se ha estabilizado.
3. Mantener los mismos hiperparámetros al comparar v1 contra v2, para que la
   diferencia se pueda atribuir a la recompensa y no a otro cambio.

Archivos de respaldo: `runs/v1_net_return/summary.csv`,
`runs/v1_net_return/seed_*/validation_log.csv` y `metadata.json`.

## Referencias

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Raffin, A., et al. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *JMLR*, 22(268), 1–8.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Sutton, R. S., y Barto, A. G. (2018). *Reinforcement learning: An introduction* (2.ª ed.). MIT Press.
