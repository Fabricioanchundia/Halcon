# Runbook de Halcón

Qué hacer cuando suena cada alerta. Primero se mitiga (que los pagos vuelvan a decidirse) y
después se investiga.

Tablero: http://localhost:3000 · Prometheus: http://localhost:9090/alerts

## Latencia

**Alerta:** `HalconLatenciaQuemaPresupuestoRapido`

1. En Grafana, mira **"Pagos esperando decisión"**. Si sube, ve a [Fila](#fila): la latencia
   es consecuencia de la fila.
2. Si la fila está en 0 pero el **tiempo de proceso** subió: revisa Valkey
   (`docker stats halcon-valkey-1`) y la CPU de la máquina. Las pausas de la máquina ya
   pasaron antes (ver el postmortem del 5 de noviembre de 2026).
3. Si un solo scorer está lento (panel **"Pagos por scorer"**), reinícialo:
   `docker compose restart scorer`.

## Fila

**Alerta:** `HalconFilaCreciendo`

1. ¿Llegan más pagos de los que se procesan? Compara **"Pagos por segundo"** con la carga.
2. Mitiga escalando: `docker compose --profile app up -d --scale scorer=3`. Más de 3 no
   ayuda: el topic tiene 3 particiones y cada una la atiende un solo scorer a la vez.
3. Si la fila no baja con 3 scorers, el cuello está en Valkey o en la máquina: ve a
   [Latencia](#latencia), paso 2.

## Sin scorers

**Alerta:** `HalconSinScorers`

1. `docker compose --profile app ps`. ¿Los contenedores están arriba?
2. `docker compose logs --tail 50 scorer`. Errores comunes:
   * `Unknown topic` → falta `uv run halcon stream init`.
   * No encuentra `models/xgb.json` → falta `uv run halcon train`.
3. `docker compose --profile app up -d --scale scorer=3`

## Errores

**Alerta:** `HalconErrores`

1. Panel **"Errores por minuto"**: el tipo (`broker`, `lag`) dice dónde mirar.
2. `broker` → `docker compose logs redpanda`. `lag` → solo afecta a la métrica, no a las
   decisiones; no es urgente.
