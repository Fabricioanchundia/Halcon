# Postmortem: pausa del scorer y fila de 49 segundos

**Fecha:** 5 de noviembre de 2026 · **Duración del impacto:** ~90 s · **Severidad:** alta
(en producción habría incumplido el SLO) · **Formato:** sin culpables (*blameless*)

## Resumen

En la primera prueba en vivo (un scorer, 500 pagos/s, 30 000 pagos), el scorer procesó
~510 pagos/s durante 30 s. De pronto bajó a ~55 pagos/s durante unos 30 s. Como el productor
seguía enviando 500/s, se formó una fila: la latencia de extremo a extremo llegó a **49 s** en
el p99, y el scorer se recuperó solo hasta ~330 pagos/s. No se perdió ninguna decisión: las
30 000 se tomaron y fueron idénticas a las de las pruebas siguientes.

## Línea de tiempo (segundos desde el inicio)

| t | Evento |
|---|---|
| 0–30 | Estable: ~510 pagos/s, proceso p50 1,08 ms, e2e p50 ~40 ms |
| ~31 | El ritmo cae a ~55 pagos/s; el proceso p99 sube a ~20 ms |
| ~31–60 | La fila crece ~445 pagos por segundo |
| ~60 | El productor termina; el scorer sigue a ~330 pagos/s |
| ~118 | La fila se vacía. e2e p99 acumulado: 49 560 ms |

## Causas

1. **Utilización demasiado alta.** Con ~1,1–1,6 ms por pago, un scorer aguanta unos 600–900
   pagos/s. A 500/s estaba al 55–85 % de su capacidad. Por teoría de colas, la espera crece
   como ρ/(1−ρ): cerca del 100 %, cualquier pausa produce una fila que tarda mucho en vaciarse.
2. **Dos viajes de red por pago** a Valkey (leer y luego escribir), que duplicaban el costo de
   la red.
3. **Una pausa de la máquina** (causa exacta no confirmada: CPU, energía o Docker Desktop). En
   corridas posteriores, el mismo código tardó 0,7 ms o 1,6 ms por pago según el estado de la
   laptop.

## Qué ayudó a detectarlo

* El scorer imprimía métricas cada 5 s. Pero eran **acumuladas**, y eso escondió la pausa: el
  promedio de un minuto bueno la diluía.

## Qué dificultó el análisis

* No había métricas por ventana, ni histograma de latencia, ni medición de la fila.
* El "tx/s" final incluía los 10 s de espera sin mensajes. El número estaba mal.

## Acciones

| Acción | Estado |
|---|---|
| Lectura y escritura en Valkey en **un solo viaje** (pipeline) | ✅ PR #7 |
| Métricas por **ventana de 5 s**; tx/s final sin contar la espera | ✅ PR #7 |
| `init` **garantiza 3 particiones**; escalado a 3 scorers (2 014 pagos/s) | ✅ PR #7 |
| Métricas Prometheus (histogramas, fila por partición), tablero y **alertas con pruebas** | ✅ Fase 6 |
| SLO explícito (99 % < 50 ms) y runbook | ✅ Fase 6 |
| Confirmar la causa de la pausa de la máquina (perfilar CPU durante una prueba) | Pendiente |

## Lección

**No se arregló con "código más rápido": se arregló dejando margen.** Un sistema que trabaja
al 85 % de su capacidad está a una pausa de un incidente. Escalar horizontalmente (más
particiones y más scorers) bajó la utilización y la latencia típica a ~10 ms con el doble de
carga.
