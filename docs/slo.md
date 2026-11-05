# SLO de Halcón

Un SLO (*Service Level Objective*) es una promesa medible sobre el servicio. Es lo que le
importa al negocio, no a la máquina: el comercio necesita una respuesta rápida para no perder
la venta.

## Indicador (SLI)

**Proporción de decisiones que salen en menos de 50 ms**, medida de extremo a extremo: desde
que el productor envía el pago hasta que la decisión sale hacia Redpanda.

```promql
sum(rate(halcon_end_to_end_seconds_bucket{le="0.05"}[30d]))
/ sum(rate(halcon_end_to_end_seconds_count[30d]))
```

El histograma tiene un borde exacto en 0,05 s para que este cálculo no interpole.

## Objetivo

| SLO | Ventana | Presupuesto de error |
|---|---|---|
| **99 % de las decisiones en < 50 ms** | 30 días | 1 % (≈ 26 000 pagos lentos de cada 2,6 M al mes, a 1 pago/s) |

¿Por qué 99 % y no 99,99 %? Porque en la laptop ya se midieron pausas de la máquina
(`docs/postmortems/2026-11-05-pausa-del-scorer.md`). Prometer lo que no se puede cumplir solo
genera alertas que todos terminan ignorando.

## Qué hacer con el presupuesto

* **Queda presupuesto:** se puede desplegar, experimentar y reentrenar el modelo.
* **Se agotó:** se congelan los cambios que no sean de confiabilidad hasta recuperarlo.

## Alertas (`ops/prometheus/alerts.yml`)

Se alerta por **velocidad de consumo del presupuesto** (*burn rate*), no por cada pico:

| Alerta | Condición | Por qué |
|---|---|---|
| `HalconLatenciaQuemaPresupuestoRapido` | > 14,4 % de decisiones lentas en 1 h **y** en 5 min | A ese ritmo, el presupuesto de 30 días se agota en ~2 días |
| `HalconFilaCreciendo` | > 5 000 pagos pendientes durante 1 min | La fila precede a la latencia: es el aviso temprano |
| `HalconSinScorers` | ningún scorer vivo durante 30 s | Nadie está decidiendo pagos |
| `HalconErrores` | > 0,1 errores/s durante 2 min | Problema sostenido, no un error suelto |

Las alertas tienen **pruebas unitarias** (`ops/prometheus/alerts_test.yml`, con
`promtool test rules`). Encontraron dos bugs antes de que llegaran a producción:

1. **Una alerta que nunca habría sonado.** "Sin scorers" usaba `count(up == 1) < 1`, pero
   `count()` de nada devuelve *vacío*, no 0. Ahora es `(count(...) or vector(0)) < 1`.
2. **Una alerta que sonaba sin motivo.** Con los scorers vivos pero sin pagos, la proporción de
   pagos lentos se calculaba como 100 %, y la alerta de latencia sonaba durante 13 minutos.
   Sin tráfico, el resultado debe ser "sin dato" (NaN), no "todo mal". Esta falsa alarma se
   vio en la laptop y ahora la cubre una prueba de regresión.

Una alerta que suena en falso es tan dañina como una que no suena: el equipo aprende a
ignorarla.
