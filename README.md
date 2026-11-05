# Halcón

Detección de fraude en pagos en tiempo real: reglas + machine learning + agente de IA para analistas.

> 🚧 En construcción — Fase 1 (simulador de transacciones).

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

## Crédito

El diseño del simulador está inspirado en el de *Reproducible Machine Learning for Credit Card
Fraud Detection — Practical Handbook* (Le Borgne, Siblini, Lebichot y Bontempi, ULB / Worldline,
2022). Halcón no contiene código de ese libro.

## Desarrollo

```bash
uv sync
uv run pytest
```
