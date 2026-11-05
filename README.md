# Halcón

Detección de fraude en pagos en tiempo real: reglas + machine learning + agente de IA para analistas.

> 🚧 En construcción — Fase 5 (agente de IA para analistas).

## Simulador

Halcón genera sus propios datos: clientes y comercios repartidos en 12 ciudades de Ecuador,
montos en USD y tres escenarios de fraude con huellas distintas.

| Escenario | Qué hace el atacante | Huella |
|---|---|---|
| `card_testing` | Prueba la tarjeta con compras de $0.50–$2 en un comercio digital y luego compra fuerte en línea | Ráfaga de montos mínimos en segundos |
| `account_takeover` | Toma la cuenta y compra en línea 4–12 veces el gasto normal | Montos fuera del perfil del cliente |
| `skimming` | Copia tarjetas en un datáfono comprometido y las usa días después | Compras presenciales grandes, a menudo en otra ciudad |

Cada fraude incluye `fraud_reported_at`: cuándo se entera el banco (reclamo o contracargo,
1 a ~45 días después). Antes de esa fecha la etiqueta no existe, y la evaluación del modelo
lo respeta (*feedback delay*).

```bash
uv run halcon simulate            # 5 000 clientes, 120 días ≈ 1 M transacciones, 0,19 % fraude
```

Salida en `data/raw/` (no se sube a git): `transactions.parquet`, `customers.parquet`,
`merchants.parquet`. Rendimiento medido: 3 M de transacciones en 1,5 s y 460 MB de RAM.

## Modelo

```bash
uv run halcon features   # features con DuckDB, solo con información del pasado
uv run halcon train      # XGBoost (usa la GPU NVIDIA si está disponible)
```

**Validación temporal con *feedback delay*.** El modelo se entrena "el día 90" y solo conoce
los fraudes reportados antes de ese día. Los 14 días previos al corte se descartan porque
la mayoría de sus etiquetas aún no llegan. Los 14 días anteriores a ese hueco son la
validación, que sirve para elegir los umbrales de decisión. El test son los 30 días siguientes.

**Métrica de negocio.** Un equipo de analistas revisa 50 alertas por día. Cada día se alertan
las 50 transacciones más riesgosas y se mide cuánto fraude cae ahí.

| Modelo (seed 42, 50 alertas/día) | PR-AUC | Precisión | Recall | $ atrapado |
|---|---|---|---|---|
| Regla ingenua: monto vs. promedio del cliente | 0.109 | 10.7 % | 33.3 % | 62.4 % |
| **XGBoost** | **0.900** | **30.3 %** | **94.8 %** | **96.8 %** |
| XGBoost "oráculo" (conoce etiquetas futuras) | 0.903 | 30.4 % | 95.0 % | 97.3 % |

Hay unos 16 fraudes por día, así que con 50 alertas la precisión máxima posible ronda 32 %.
Una prueba automática verifica que los features no miran al futuro: si se borran las
transacciones posteriores, los features anteriores no cambian.

> Los datos simulados son más limpios que los reales. Lo que vale aquí es la metodología:
> validación temporal, *feedback delay*, presupuesto de alertas y pruebas anti-fuga.

## Motor de decisión

Cada transacción recibe **APROBAR**, **REVISAR** (cola de analistas) o **RECHAZAR**. Gana la
acción más severa entre las reglas y el puntaje del modelo.

| Código | Regla | Acción |
|---|---|---|
| R001 | 4 o más compras de $2 o menos en 1 hora (card testing) | Rechazar |
| R002 | 5 o más compras en 10 minutos | Revisar |
| R003 | Compra presencial en otra ciudad, 6 veces el promedio y de $200 o más | Revisar |
| R004 | Monto de $2 000 o más | Revisar |

**Umbrales elegidos por dinero, no a ojo.** Se buscan los umbrales que minimizan el costo total
en la validación, sin pasar de 50 revisiones por día. El costo total suma el fraude que se escapa
(más el cargo por contracargo), las horas de analista y los rechazos injustos a clientes honestos.
Supuestos en `CostModel`: revisión $3, el analista detiene el 85 % de lo que revisa, rechazo
injusto $10 + 10 % del monto, contracargo $15.

Resultado en los 30 días de test:

| Política | Costo total | Revisiones/día | Rechazos injustos | Fraude detenido |
|---|---|---|---|---|
| Sin sistema (aprobar todo) | $124 968 | 0 | 0 | 0 % |
| Solo reglas | $82 570 | 5.7 | 1 | 40.6 % |
| Solo modelo | $12 455 | 20.1 | 12 | 90.4 % |
| **Halcón (reglas + modelo)** | **$12 519** | **20.7** | **13** | **90.4 %** |

**Lectura honesta:** en estos datos, el modelo ya aprendió lo que atrapan las reglas, así que
sumarlas casi no cambia el costo. Su valor es operativo: se pueden activar en minutos ante un
ataque nuevo (el modelo tarda días en reentrenarse), rechazan con certeza sin depender de un
puntaje y su código de razón es fácil de explicar a un cliente o a un regulador.

```bash
uv run halcon score --sample card_testing   # también: account_takeover, skimming, legit
```

```
Transacción 900024 · 2026-04-19 16:58 (Ecuador) · digital_goods · $0.82
  Decisión: ⛔ RECHAZAR   puntaje 0.9823   (2.58 ms)
  Pistas que más subieron el riesgo:
    small_tx_count_1h        =       3.00   (+4.30)
    secs_since_prev_tx       =      35.00   (+2.68)
    amount_usd               =       0.82   (+2.62)
```

Las pistas son valores SHAP (`pred_contribs` de XGBoost). Latencia medida por transacción,
con explicación incluida: p50 1,1 ms y p99 2,3 ms.

## Tiempo real

```
simulador ──► Redpanda "transactions" ──► scorer ──► Redpanda "decisions"
              (clave = cliente)             │
                                            └─ Valkey: historial de 30 días por cliente
```

* **Redpanda** (compatible con Kafka, un solo nodo y 1 GB) transporta los pagos. La clave del
  mensaje es el cliente, así que sus compras llegan en orden.
* **Valkey** guarda el historial reciente de cada cliente: una lectura y una escritura por pago.
* **Sin *training-serving skew*:** las features en vivo (`stream/online.py`) se comparan fila por
  fila contra las del SQL de entrenamiento. Dan exactamente lo mismo en las 994 146
  transacciones.
* **Camino rápido:** el 99 % de los pagos se aprueba y no necesita explicación. SHAP solo se
  calcula para revisar o rechazar. El scorer tarda p50 0,18 ms y p99 0,7 ms, sin contar la red.

### Rendimiento medido en una laptop

MSI Katana 17 (i7-14650HX, 16 GB de RAM), WSL2 limitado a 6 GB, Docker Desktop. Datos: los 30
días de test que el modelo nunca vio.

| Configuración | Carga | Procesado | Proceso p50 / p99 | Extremo a extremo p50 | Fraude detenido o en revisión |
|---|---|---|---|---|---|
| 1 scorer | 1 000 pagos/s | 1 025/s | 0,74 / 1,84 ms | ~11 ms | 102 de 111 (91,9 %) |
| **3 scorers** | **2 000 pagos/s** | **2 014/s** (663 + 688 + 663) | 0,91 / 5,4 ms | **~10 ms** | **196 de 207 (94,7 %)** |

*Extremo a extremo* va desde que el productor envía el pago hasta que la decisión sale hacia
Redpanda. Por ventanas de 5 s, el p99 anda entre 33 y 90 ms, con picos ocasionales de ~250 ms
(pausas de la máquina). Con 1 scorer a 1 000/s se mantuvo por debajo de 50 ms. RAM usada:
Redpanda 210 MB y Valkey 21 MB.

**Lo que enseñaron las mediciones:**

1. **Teoría de colas.** Con un scorer al ~85 % de su capacidad, cualquier pausa formaba una fila
   y la latencia saltaba de 16 ms a más de 600 ms. La espera crece como ρ/(1−ρ). La solución no
   fue código más rápido: fue sumar scorers.
2. **Un viaje de red por pago.** La lectura y la escritura en Valkey van en un solo *pipeline*.
   Valkey ejecuta los comandos en orden, así que la lectura ve el estado previo a la compra actual.
3. **Las particiones mandan.** Si un consumidor se suscribe antes de que exista el topic, Redpanda
   (modo desarrollo) lo crea con 1 partición y solo un scorer trabaja. `halcon stream init` ahora
   verifica y agrega las particiones que falten.

```bash
docker compose up -d                 # Redpanda + Valkey (≈1,9 GB máx.)
uv run halcon stream init            # crea topics y precarga 30 días de historial en Valkey
uv run halcon stream score           # en 1 a 3 terminales: cada una toma particiones
uv run halcon stream produce --limit 120000 --rate 2000  # otra terminal: 2 000 pagos/s
uv run halcon stream summary         # cruza las decisiones con la verdad del simulador
docker compose down                  # libera la RAM al terminar
```

## Agente investigador (humano en el bucle)

Cada pago que el motor manda a **REVISAR** o **RECHAZAR** se vuelve un caso en una cola
(`data/cases.duckdb`). Un agente con un modelo local (Ollama, `qwen2.5:7b`, en la GPU) le
prepara al analista un resumen con evidencia citada y una recomendación. **La decisión final
es siempre del humano.**

```
caso ─► evidencia de solo lectura (E1…E6) ─► LLM con esquema JSON ─► barandas ─► analista
        transacción · decisión y reglas ·                            citas válidas
        pistas SHAP · perfil 30 días ·                               números verificados
        últimas 24 h · comercio
```

**Decisiones de diseño:**

* **Las herramientas las ejecuta el código, no el modelo.** Con un modelo de 7B, elegir qué
  consultar falla más de lo que ayuda. Se junta la evidencia de forma determinista y se pide
  una sola respuesta con esquema JSON.
* **La evidencia nunca ve el futuro ni las etiquetas.** Se consulta una vista sin
  `is_fraud`/`fraud_scenario` y solo con datos anteriores al pago. Hay pruebas que lo verifican.
* **Barandas en código, no en el prompt:** cada hallazgo debe citar evidencia existente y cada
  número debe estar en la evidencia. Si no, el caso se marca para el analista.
* **Datos ≠ instrucciones:** la evidencia va delimitada y el modelo tiene orden de no obedecer
  nada que venga dentro (defensa contra *prompt injection*).
* **Se compara contra una heurística sin IA** (puntaje ≥ 0,5). Si el LLM no la supera, su
  aporte es ahorrar tiempo de lectura al analista, no decidir mejor.

### Resultado medido: el LLM explica, el modelo decide

Muestra de 40 casos (20 fraudes y 20 legítimos), `qwen2.5:7b` en una RTX 5060 de 8 GB:

| | Heurística sin IA (puntaje del modelo ≥ 0,5) | Agente `qwen2.5:7b` |
|---|---|---|
| Acierto de la recomendación | **90,0 %** | 75,0 % |
| Fraudes / legítimos reconocidos | — | 70 % / 80 % |
| Patrón correcto (card testing, toma de cuenta, clonación) | — | 62,5 % |
| Respuestas con números inventados o citas inválidas | — | **0 %** |
| Tiempo por caso | — | p50 14,7 s (≈ 4 s con la GPU libre) |

**Conclusión: el LLM decide peor que XGBoost.** El modelo aprendió de cientos de miles de pagos;
el LLM solo lee siete párrafos. Por eso Halcón **no le delega la decisión**:

* El **modelo y las reglas deciden**. El agente **redacta** el caso, para que el analista
  entienda en segundos lo que antes le tomaba minutos de leer tablas.
* Si el agente **contradice** al modelo, el caso se marca **⚠ DESACUERDO** para que el humano lo
  mire con más atención. Medido: hubo **10 desacuerdos y el agente tenía razón en 2**. Solo, el
  agente no es confiable. Pero el modelo cometió **4 errores** en la muestra, y la alerta de
  desacuerdo atrapó **la mitad** de ellos, a cambio de 8 falsas alarmas. Sirve como señal de
  "mira esto dos veces", no como juez.
* Las barandas funcionaron: **cero datos inventados en 40 casos**. La primera versión del prompt
  daba respuestas contradictorias ("incierto" con confianza "alta"). Se corrigió con dos
  cambios: el código calcula las comparaciones (E7) y una regla de consistencia obliga a
  reintentar.

```bash
uv run halcon cases ingest                 # casos de los últimos 30 días (sin Docker)
uv run halcon cases list
uv run halcon cases investigate --show-evidence
uv run halcon cases evaluate --limit 40    # acierto, patrón, barandas y latencia vs. la verdad
uv run halcon cases resolve --tx-id 945459 --as fraude --note "cliente confirmó"
```

## Crédito

El diseño del simulador está inspirado en el de *Reproducible Machine Learning for Credit Card
Fraud Detection — Practical Handbook* (Le Borgne, Siblini, Lebichot y Bontempi, ULB / Worldline,
2022). Halcón no contiene código de ese libro.

## Desarrollo

```bash
uv sync
uv run pytest
```
