# Revisión y organización del agente inversor

Verificación de requisitos y plan corregido por actividades

Proyecto final de aprendizaje por refuerzo    |    20 de octubre de 2026

El proyecto dispone de datos, análisis exploratorio, formulación, entorno y cinco versiones entrenadas de la recompensa. La principal ausencia era la evaluación final comparable contra estrategias simples. La entrega incorpora esa evaluación, sus métricas y gráficas, y organiza la documentación por actividades.

## Criterio de revisión

La descripción de la actividad es la referencia académica. El plan de trabajo amplía esa referencia con validación, tres semillas y una ejecución final registrada. Estas ampliaciones mejoran el protocolo, pero no deben presentarse como exigencias textuales del enunciado.

El plan menciona cinco criterios de un punto y una presentación ante un panel. La descripción suministrada enumera requisitos y entregables, pero no permite confirmar esa ponderación ni que la presentación sea obligatoria. Por ello, esta revisión evalúa cobertura y evidencia, sin prometer una calificación.

| Actividad | Estado de la entrega |
| --- | --- |
| Datos y análisis exploratorio | Implementada con límites metodológicos documentados |
| Estado y acciones | Implementada y prueba causal reforzada |
| Entorno de inversión | Implementado y verificado; simplificaciones declaradas |
| Agente y entrenamiento | Modelos y registros disponibles |
| Diseño de recompensas | Cinco versiones comparadas en validación |
| Evaluación e integración | Implementada con resultados finales y trazabilidad |
| Reflexiones individuales | Una propuesta desarrollada; faltan cinco aportes propios |


Los estados siguientes distinguen lo ya existente, las correcciones incorporadas y los asuntos que requieren una decisión o un aporte del equipo. La prueba final conserva la selección recomendada antes de observar sus resultados.

Base revisada: commit 23c07d0017d581ab7a475d4df1c6515ace3156d1. Las modificaciones y la ejecución quedan identificadas por sus huellas de archivo en el paquete.

## Datos y análisis exploratorio

Apartados 6, 7 y 16 del enunciado

Preparar una serie temporal verificable y comprender sus características antes de diseñar el agente.

1. Carga y calidad. Cumplida. src/data.py valida columnas, orden, nulos, duplicados, huecos y coherencia OHLCV. El origen utilizado es Binance BTC/USDT por hora. Debe confirmarse con la asignatura si se esperaba el dataset común del profesor; el enunciado lo recomienda.

2. División temporal. Cumplida. configs/project.toml fija entrenamiento hasta octubre de 2024, validación hasta octubre de 2025 y prueba hasta octubre de 2026, con finales exclusivos. La validación es una mejora del plan sobre el mínimo de dos períodos.

3. Análisis exploratorio. Cumplida. eda.ipynb contiene precio, retornos, colas, volatilidad, tendencias, caídas, volumen e indicadores con interpretaciones. Se corrigen errores de redacción y se aclara la diferencia entre sus indicadores desplazados y los del agente.

4. Tratamiento de huecos. Cumplida con limitación. Se conservan seis horas iniciales ausentes y se imputan 32 horas internas con el último cierre y volumen cero. Se declara la distorsión posible de volatilidad y no se afirma que una baja proporción elimine el sesgo.

5. Comparación entre períodos. Realizada, con reserva metodológica. El EDA ya describe held-out. Se elimina la afirmación de una prueba totalmente ciega; una evaluación futura deberá reservar otro período antes de explorar sus resultados.

6. Estructura y dependencias. Cumplida. Existen data, src, notebooks, configs y pyproject.toml. La entrega conserva modelos, datos y resultados para que la reproducción no dependa de una nueva descarga.

7. Instrucciones de ejecución. Corregida. README explica instalación, entrenamiento, baselines, evaluación y reproducción de la prueba registrada, y deja de presentar los módulos completados como pendientes.

Entrega organizada: src/data.py, src/download_data.py, src/prepare_data.py, notebooks/eda.ipynb, configs/project.toml y README.md. Depende del acuerdo sobre dataset y fechas; alimenta la formulación y el entorno.

## Formulación del estado y las acciones

Apartados 3, 4 y 5 y preguntas 8 a 15

Definir exactamente qué información se observa y qué decisiones de inversión permite el sistema.

1. Selección de indicadores. Cumplida. La red recibe retorno horario, logaritmo del volumen, distancias a medias de 24 y 168 horas, RSI de 14 horas, volatilidad de 24 horas y posición. El informe justifica cada variable.

2. Ventanas y significado. Cumplida en la integración. Un día y una semana aportan horizontes de tendencia distintos; RSI y volatilidad resumen momentum y dispersión recientes. Las ventanas no prueban capacidad predictiva por sí mismas.

3. Construcción causal. Cumplida para la convención implementada. src/features.py utiliza información hasta el cierre t; la acción se aplica al retorno de t a t+1. El EDA usa un desplazamiento adicional que ahora se identifica expresamente.

4. Escalamiento. Cumplida. env_factory ajusta StandardScaler únicamente con filas útiles de entrenamiento y transforma las otras particiones sin reajuste. La posición se agrega en su escala natural; el patrimonio no forma parte del vector de entrada.

5. Prueba de información futura. Corregida. La prueba original usaba cinco filas y dejaba sin valores válidos las ventanas de 24 y 168 horas. La nueva prueba usa 420 filas, cambia precio y volumen posteriores al corte y compara indicadores ya calculables.

6. Espacio de acciones. Cumplida. Los índices 0, 1 y 2 representan 0 %, 50 % y 100 % del capital. Se explican granularidad, facilidad de interpretación y límites: sin cortos, apalancamiento, varios activos ni pesos intermedios.

7. Alcance de la observación. Limitación añadida. Los indicadores no garantizan un estado plenamente markoviano. La recompensa de drawdown depende de un máximo acumulado que la red no observa directamente. Agregar ese estado requiere un experimento nuevo y reentrenamiento.

Entrega organizada: src/features.py, src/env_factory.py y tests/test_lookahead.py. Recibe datos y fechas; entrega a la simulación el orden exacto de las seis variables de mercado.

## Entorno de simulación

Apartados 8 y 9

Transformar una acción en cambios de patrimonio, costos y recompensa bajo una convención temporal explícita.

1. Interfaz de interacción. Cumplida. TradingEnvironment implementa reset y step con la interfaz de Gymnasium, espacios de observación y acción y finalización del episodio.

2. Valor del portafolio. Cumplida bajo el modelo del repositorio. Primero se descuenta el costo y luego se aplica el retorno de cierre a cierre a la fracción elegida. No se modificó esta dinámica después de entrenar los modelos.

3. Costos. Cumplida. El costo es 0,001 por el cambio absoluto de fracción y por el patrimonio previo. Se registra cada costo y su suma. La penalización de recompensa por operar no se vuelve a descontar del dinero.

4. Recompensa intercambiable. Cumplida. El entorno recibe una función de recompensa y reinicia su estado al empezar cada episodio. Las cinco versiones usan la misma dinámica económica.

5. Tiempo de decisión. Aclarada. Se decide con el cierre t conocido y se simula ejecución a ese cierre. Es una simplificación de ejecución; no incorpora demora, spread ni deslizamiento. La evaluación guarda horas separadas para decidir y valorar.

6. Pruebas de sanidad. Cumplida y ampliada. Se comprueban efectivo, comprar y mantener, costos y causalidad. Se añaden pruebas del cálculo de drawdown desde el capital inicial y de las fechas de valoración.

7. Compatibilidad. Verificada con la batería del proyecto, incluida la comprobación del entorno de Stable-Baselines3. El registro de pruebas acompaña la entrega.

8. Rebalanceo y final del episodio. Limitación declarada. Mantener una fracción de 50 % supone un rebalanceo implícito cuyos costos no se modelan si la acción no cambia. No hay liquidación forzosa final. El corte temporal se distingue de la terminación por pérdida del capital.

Entrega organizada: src/environment.py y tests/test_environment.py. Recibe datos e indicadores; debe mantenerse común al entrenamiento, los baselines y la evaluación.

## Agente y entrenamiento

Apartados 10, 12 y 16

Aprender y seleccionar una política de forma reproducible utilizando entrenamiento y validación.

1. Algoritmo. Cumplida. Se usa DQN de Stable-Baselines3 con tres acciones discretas. Se documenta el papel de la red objetivo y del buffer de experiencias.

2. Prueba del flujo. Cumplida mediante las pruebas de agente y de integración. Las corridas cortas se reservan para comprobar funcionamiento y no sustituyen los resultados del experimento.

3. Configuración. Cumplida. configs/dqn.toml incluye arquitectura, tasa de aprendizaje, lote, descuento, buffer, exploración, pasos, evaluaciones periódicas y criterio de parada.

4. Semillas. Cumplida. Se conservan las semillas 42, 43 y 44 y las medidas de variación entre corridas. La evaluación final de un único modelo no mide la variabilidad de todo el procedimiento de entrenamiento.

5. Selección. Cumplida con límites. Los checkpoints se eligen por retorno de validación. La comparación reiterada de versiones e hiperparámetros sobre un mismo año genera optimismo de selección, que debe declararse.

6. Modelos y registros. Cumplida. runs conserva configuraciones, metadatos, modelos y curvas. Algunos entrenamientos originales registran un árbol de código modificado; su commit por sí solo no reconstruye cada cambio histórico.

7. Entrenamientos por recompensa. Cumplida. Las cinco versiones tienen corridas con el protocolo común. No se requiere repetirlas para completar la evaluación solicitada.

8. Congelamiento. Completado. Se adopta la recomendación previa de v5 y semilla 42, mediana por retorno de validación. El checkpoint corresponde a 200.000 pasos, dentro de una corrida de 300.000. Se guarda un protocolo con huellas antes de la prueba final.

Entrega organizada: src/agent.py, src/train.py, src/freeze_evaluation.py, configs/dqn.toml y docs/algoritmo_y_entrenamiento.md. Entrega un modelo inmutable a la evaluación.

## Diseño e iteración de la recompensa

Apartado 9 y pasos 16 a 22

Vincular cada cambio de recompensa con una conducta observada y evaluar sus efectos en validación.

1. Recompensa inicial. Cumplida. v1 utiliza retorno neto de costos. Sus resultados muestran elevada exposición y una reacción costosa durante episodios de caída.

2. Diagnóstico. Cumplida. Hay trayectorias, posiciones, costos y operaciones. Se diferencia la observación de una explicación causal: comparar tres semillas no identifica por sí solo todos los mecanismos del aprendizaje.

3. Alternativas. Cumplida. v2 penaliza rotación; v3 penaliza profundización de drawdown con coeficiente 1; v4 reduce ese coeficiente a 0,02; v5 combina rotación y drawdown.

4. Comparación. Cumplida. Se presentan retornos, drawdown, costos, operaciones y tiempo en cada posición para tres semillas. Se distinguen mejores checkpoints, último checkpoint y evaluaciones de la segunda mitad del entrenamiento.

5. Anomalía. Cumplida y corregida conceptualmente. v3 permanece casi siempre en efectivo. La escala de la penalización es una explicación plausible; no se demuestra una política globalmente óptima. Un Q positivo para efectivo tampoco prueba sobreestimación, porque incluye decisiones futuras.

6. Ciclo completo. Cumplida. El informe enlaza propuesta, entrenamiento, observación, problema, modificación, nuevo entrenamiento y comparación, en vez de enumerar fórmulas aisladas.

7. Elección final. Cumplida. La recomendación v5 y semilla 42 procede de validación y se conserva para la prueba. Las medidas llamadas robustas reducen la dependencia del mejor pico, pero siguen perteneciendo al mismo año de validación y no son evidencia independiente.

Entrega organizada: src/rewards.py, src/analysis.py, src/reward_report.py, docs/diseno_de_recompensas.md y runs/comparison. Los módulos y documentos cubren el análisis previsto sin requerir un notebook adicional con el mismo contenido.

## Evaluación y comparación de estrategias

Apartados 11 a 14 y resultados 23 a 29

Medir el modelo final frente a referencias simples bajo condiciones idénticas y explicar su comportamiento.

1. Baselines. Implementados. Comprar y mantener elige exposición total; el agente aleatorio selecciona uniformemente entre las tres acciones. Se utilizan veinte semillas fijadas del 42 al 61, sin escoger la de mejor resultado.

2. Métricas. Implementadas. Retorno acumulado, máximo drawdown, costos, operaciones, exposición y tiempo en cada posición. El drawdown incluye el capital inicial y las operaciones cuentan cambios de posición, incluida la entrada.

3. Gráficas. Implementadas. Curvas de capital comparables, banda descriptiva de ejecuciones aleatorias y posiciones del agente con el precio. Cada figura se acompaña de una interpretación.

4. Ensayo en validación. Realizado antes de la prueba final. Se verifica que el modelo congelado reproduce las métricas guardadas de su checkpoint de validación. Los tests verifican costos, fechas y agregación de resultados.

5. Ejecución final. Registrada. Se conserva una evaluación oficial del modelo congelado y el conjunto de baselines. El registro impide una segunda evaluación oficial accidental. Una reproducción explícita conserva el protocolo y los resultados originales.

6. Tabla e interpretación. Completadas. Se diferencia pérdida absoluta, desempeño relativo y riesgo. La dispersión del baseline aleatorio corresponde a semillas de acciones; no representa incertidumbre sobre futuros mercados.

7. Integración de la documentación. Completada. Se unifican formulación, datos, entorno, algoritmo, recompensa, protocolo, resultados, límites y reproducción. La reflexión propuesta responde las cinco preguntas del enunciado.

Entrega organizada: src/baselines.py, src/evaluate.py, src/metrics.py, tests/test_evaluation.py, models/final y results/held_out. Los resultados finales no habilitan ajustes posteriores sobre ese mismo período.

## Integración y cierre académico

### Discusión conjunta

La discusión debe reunir la no estacionariedad del mercado, la escala de la recompensa, el sesgo de selección en validación, las limitaciones del estado y las simplificaciones de ejecución. El README desarrolla estos puntos y relaciona cada uno con la evidencia disponible.

### Reflexiones individuales

La reflexión es individual: el apartado 15 exige responder qué sorprendió, qué decisión tuvo mayor impacto, qué se cambiaría, qué conducta sigue sin explicación y qué diferencia existe entre aprender RL y aplicarlo. Se entrega una propuesta en primera persona basada en los resultados. Los otros cinco integrantes deben aportar sus respuestas propias; no corresponde atribuirles experiencias inventadas.

### Orden de presentación del trabajo

| Orden | Actividad | Resultado que permite avanzar |
| --- | --- | --- |
| 1 | Datos y análisis | Serie validada y cortes temporales fijos |
| 2 | Estado y acciones | Observación causal y decisiones definidas |
| 3 | Entorno | Simulación y costos verificados |
| 4 | Entrenamiento | Corridas reproducibles en varias semillas |
| 5 | Recompensas | Diagnóstico e iteración comparada |
| 6 | Evaluación | Modelo congelado y comparación final |
| 7 | Discusión y reflexión | Interpretación, límites y aportes individuales |
| 8 | Reproducibilidad | Datos, código, modelos y registro ejecutable |


### Pendientes de validación por el equipo

Confirmar la aceptación del dataset de Binance si la asignatura exige uno común; incorporar las cinco reflexiones restantes y validar la propuesta personal; agregar nombres y datos de portada si el formato de entrega los exige. Si se requiere una exposición, usar el orden anterior y ajustar los tiempos al espacio asignado.

No se puede demostrar retrospectivamente que nadie examinó el held-out fuera de los archivos revisados. La evidencia sí permite identificar esta ejecución, su selección previa y sus archivos congelados. Una nueva investigación con decisiones posteriores necesita otro período reservado.

### Fuentes de la revisión

Descripción de la actividad, apartados 1 a 17; plan de división del trabajo, tareas y cronograma; código, notebooks, modelos y registros del repositorio en el commit indicado. Los títulos de las actividades sustituyen las etiquetas numéricas de integrantes.

Repositorio del proyecto y fuente original
