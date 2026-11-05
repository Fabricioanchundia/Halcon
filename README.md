# Halcón

Detección de fraude en pagos en tiempo real: reglas + machine learning + agente de IA para analistas.

> 🚧 En construcción — Fase 3 (motor de decisión).

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

## Crédito

El diseño del simulador está inspirado en el de *Reproducible Machine Learning for Credit Card
Fraud Detection — Practical Handbook* (Le Borgne, Siblini, Lebichot y Bontempi, ULB / Worldline,
2022). Halcón no contiene código de ese libro.

## Desarrollo

```bash
uv sync
uv run pytest
```
