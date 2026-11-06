# Drift: cuando el fraude cambia

**Pregunta de entrevista:** *"Tu modelo funciona hoy. ¿Qué pasa cuando los estafadores cambian?"*

`halcon drift` responde con un experimento reproducible.

## El escenario

1. Se simulan 150 días. Hasta el **día 80** el *card testing* es el de siempre (pruebas de
   $0,50–$2).
2. El día 80 los atacantes descubren la regla R001 ("4 o más compras de $2 o menos en una hora")
   y suben las pruebas a **$2,01–$3**. Como les funciona, **escalan el ataque ×4**.
3. El banco **todavía no tiene R005** (la regla de la Fase 7). Solo cuenta con el modelo y lo
   reentrena **cada semana** con los reclamos que ya le llegaron. Un reclamo tarda ~15 días en
   promedio.

Antes del día 80 el mundo es idéntico, pago por pago, al de la misma semilla sin drift. La ola
extra usa su propio generador aleatorio, y una prueba lo verifica. Así, todo cambio en los
resultados se debe al ataque.

## Cuatro formas de reaccionar

| Variante | Qué hace |
|---|---|
| **congelado** | Usa el modelo del día del ataque y no lo reentrena nunca. |
| **estándar** | Usa la receta de la Fase 2, reentrenando cada semana. Las etiquetas maduran durante un hueco de 14 días y los umbrales se calibran con 14 días de validación. |
| **hueco corto** | Igual que el estándar, pero con un hueco de 3 días. Ve antes el ataque nuevo, pero la mayoría de esos fraudes todavía no tiene reclamo y **entran como "legítimos"**. |
| **+ reclamos frescos** | Igual que el estándar, pero también suma **solo los fraudes ya confirmados** del hueco. No agrega legítimos inmaduros. |

## Resultados

Medidos en CPU con 3 semillas (42, 7 y 2026); cada una toma ~3 minutos. Se repitió la semilla 42
en la GPU de la laptop (RTX 5060, 91 s): mismas conclusiones, cifras en la última fila.

| | congelado | estándar | hueco corto | **+ reclamos frescos** |
|---|---|---|---|---|
| Semana en que bloquea ≥ 50 % de las tarjetas | nunca | 6 · 6 · 6 | 7 · 5 · nunca | **3 · 3 · 2** |
| $ perdido en card testing (8 semanas) | 2 081 · 1 285 · 4 212 | 1 767 · 1 681 · 2 349 | 11 350 · 9 387 · 10 260 | **1 220 · 846 · 1 066** |
| Clientes honestos rechazados por semana | 2,2 · 10,6 · 4,0 | 4,1 · 5,9 · 15,6 | 1,0 · 1,5 · 1,0 | 6,0 · 7,4 · 15,6 |
| **GPU, semilla 42**: semana en que aprende · $ perdido | nunca · 2 658 | 6 · 1 769 | 7 · 10 466 | **3 · 1 290** |

**Lo que muestran:**

* **El modelo congelado nunca aprende.** Bloquea el 0 % de las tarjetas las 8 semanas. Sin
  reentrenar, un ataque nuevo dura para siempre.
* **El estándar aprende tarde: siempre en la semana 6 (35 días).** No es mala suerte, sale de la
  propia receta. Entrena con datos que terminan 28 días antes de hoy (14 de hueco + 14 de
  validación), y esos fraudes además deben estar reportados. El primer ejemplo del ataque nuevo
  recién entra al entrenamiento ~5 semanas después.
* **Acortar el hueco es PEOR: pierde de 4 a 6 veces más que el reentreno estándar, y hasta 7
  veces más que no reentrenar nunca.** El modelo ve el ataque nuevo, pero como "legítimo",
  porque los reclamos aún no llegan. Aprende que las ráfagas de $2,50 son normales y deja pasar
  hasta los cobros grandes. Es el error más caro de
  este proyecto, y sale de una idea que suena razonable: "reentrenemos con datos más frescos".
* **Sumar solo los reclamos frescos aprende de 3 a 4 semanas antes y pierde menos dinero**, a
  cambio de algunos rechazos más a clientes honestos (similar al estándar). Los fraudes confirmados son
  información segura; los "legítimos" recientes no lo son.

## Monitoreo sin etiquetas: ¿quién avisa primero?

El modelo en producción no se entera hasta que llegan los reclamos. Estos monitores no
necesitan esperar:

| Monitor | Primer aviso (3 semillas en CPU; en GPU igual: +1 día) | Falsas alarmas en las 2 semanas previas |
|---|---|---|
| PSI de las 17 features (día contra referencia) | **nunca** | 0 |
| PSI del puntaje del modelo | **nunca** | 0 |
| Ráfagas en 7 días (fuera de media ± 3σ) | **+1 · +1 · +0 días** | 0 |
| Cobertura de R001 sobre las ráfagas (< 50 %) | **+1 · +1 · +2 días** | 0 |

El PSI es el monitor de drift más citado, pero **no ve este ataque**. Toca a ~1 de cada 1 000
clientes, y el promedio de la población no se mueve. Lo que sí cambia es la **cola**: aparecen
más ráfagas de compras chicas en línea, y la regla que antes las atrapaba deja de verlas. La
alarma "cobertura de R001" dice exactamente qué pasó: *las ráfagas siguen, pero R001 ya no
las ve*. Es la pista para escribir R005 **el día 81**, cinco semanas antes de que el
reentrenamiento estándar lo resuelva solo.

## Conclusión para producción

1. **Reglas para reaccionar en horas y reentrenamiento para la defensa de fondo.** Ninguno de
   los dos alcanza solo.
2. **Hay que vigilar la cola y las reglas, no solo el PSI global.**
3. **Al reentrenar, nunca hay que confundir "sin reclamo todavía" con "legítimo".** Los fraudes
   confirmados recientes se pueden sumar; los legítimos recientes deben esperar a madurar.

## Limitaciones

* Datos simulados con un solo tipo de drift. Un ataque que cambie la población entera, como un
  comercio nuevo masivo, sí movería el PSI.
* Son 7 a 15 tarjetas atacadas por semana, así que los porcentajes semanales saltan. Por eso se
  muestran 3 semillas.
* La pérdida cuenta lo aprobado y el 15 % de lo revisado hasta el primer rechazo, cuando se
  bloquea la tarjeta. El red team de la Fase 7 no descontaba lo evitado por el bloqueo; en eso
  las dos cifras no son comparables.
