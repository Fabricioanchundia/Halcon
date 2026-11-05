# Halcón

Detección de fraude en pagos en tiempo real: reglas + machine learning + agente de IA para analistas.

> 🚧 En construcción — Fase 2 (modelo de machine learning).

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
la mayoría de sus etiquetas aún no llegan. El test son los 30 días siguientes.

**Métrica de negocio.** Un equipo de analistas revisa 50 alertas por día. Cada día se alertan
las 50 transacciones más riesgosas y se mide cuánto fraude cae ahí.

| Modelo (seed 42, 50 alertas/día) | PR-AUC | Precisión | Recall | $ atrapado |
|---|---|---|---|---|
| Regla ingenua: monto vs. promedio del cliente | 0.109 | 10.7 % | 33.3 % | 62.4 % |
| **XGBoost** | **0.899** | **30.3 %** | **94.8 %** | **97.2 %** |
| XGBoost "oráculo" (conoce etiquetas futuras) | 0.912 | 30.5 % | 95.4 % | 96.9 % |

Hay unos 16 fraudes por día, así que con 50 alertas la precisión máxima posible ronda 32 %.
Una prueba automática verifica que los features no miran al futuro: si se borran las
transacciones posteriores, los features anteriores no cambian.

> Los datos simulados son más limpios que los reales. Lo que vale aquí es la metodología:
> validación temporal, *feedback delay*, presupuesto de alertas y pruebas anti-fuga.

## Crédito

El diseño del simulador está inspirado en el de *Reproducible Machine Learning for Credit Card
Fraud Detection — Practical Handbook* (Le Borgne, Siblini, Lebichot y Bontempi, ULB / Worldline,
2022). Halcón no contiene código de ese libro.

## Desarrollo

```bash
uv sync
uv run pytest
```
