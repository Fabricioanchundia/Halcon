# Modelo de amenazas de Halcón

Método **STRIDE** (suplantación, manipulación, repudio, fuga de información, denegación de
servicio, elevación de privilegios) sobre el flujo real del sistema. Cada amenaza dice qué la
mitiga **hoy**, con evidencia en el repo, y qué faltaría en producción.

```
productor ─► Redpanda "transactions" ─► scorer ─► Valkey (historial)
                                           │
                                           └─► Redpanda "decisions" ─► cola de casos ─► agente (Ollama)
                                                                                │
                                                                           analista humano
```

## Atacantes que importan

| Atacante | Qué quiere | Cómo |
|---|---|---|
| **Defraudador** | Que sus pagos se aprueben | Estudia qué se rechaza y se adapta (evasión) |
| **Cliente deshonesto** | Que le devuelvan una compra legítima | "Fraude amigo": reclama algo que sí compró (etiquetas falsas) |
| **Atacante de cadena de suministro** | Ejecutar código en el CI o en la imagen | Compromete una dependencia, una imagen o una GitHub Action |
| **Interno curioso** | Ver datos de clientes | Lee los logs, los topics o la cola de casos |

## Amenazas y mitigaciones

| # | STRIDE | Amenaza | Mitigación hoy (evidencia) | En producción, además |
|---|---|---|---|---|
| T1 | Manipulación | **Evasión de reglas**: el ladrón prueba tarjetas con $2,01 para esquivar R001 | `halcon redteam` lo simula: R001 pasó de 91,4 % a **0 %** de pruebas rechazadas. **R005** mira la velocidad sin depender del monto exacto y recupera el **100 %** de tarjetas bloqueadas antes del cobro, con 0 clientes honestos afectados | Red team periódico; reentrenar con fraudes nuevos (Fase 8); no publicar umbrales |
| T2 | Manipulación | **Envenenamiento de etiquetas**: reclamos falsos que enseñan al modelo que lo legítimo es fraude | Las etiquetas solo cuentan cuando se reportan (`fraud_reported_at`) y el entrenamiento respeta ese retraso | Validar contracargos antes de usarlos como etiqueta; vigilar cambios bruscos en la tasa de fraude por comercio |
| T3 | Manipulación | **Inyección de instrucciones** al agente (*prompt injection*) mediante texto en los datos | La evidencia va delimitada y se trata como datos; las barandas verifican citas y números en código; **el agente no decide** y no tiene herramientas de escritura | Nunca pasarle texto libre de terceros sin sanear; listas de herramientas mínimas |
| T4 | Fuga de información | **Secretos en el repo** | **Caso real:** GitGuardian detectó una contraseña de Grafana en el PR #9. Se eliminó (PR #10) y `test_no_credentials_in_compose` impide que vuelva a pasar. gitleaks en pre-commit | Secretos en un gestor (Vault, AWS Secrets Manager) y rotación |
| T5 | Fuga de información | **Datos de tarjeta** en mensajes o logs | No hay números de tarjeta: solo IDs de cliente y comercio. El agente y las métricas nunca ven etiquetas ni datos personales | PCI DSS: tokenizar el número de tarjeta en el borde y no guardarlo nunca; cifrado en tránsito (TLS) |
| T6 | Suplantación | Cualquiera en la red publica pagos falsos en Redpanda o lee Valkey | Solo en la laptop, sin exposición pública | TLS + SASL y ACL por topic en Redpanda; ACL y contraseña en Valkey; red privada |
| T7 | Elevación de privilegios | Código malicioso dentro del scorer | Contenedor **sin root**, **solo lectura**, sin capacidades del kernel y `no-new-privileges` (`test_scorer_container_is_hardened`). Trivy no encontró problemas en el Dockerfile | Políticas del orquestador (Kubernetes Pod Security "restricted") |
| T8 | Manipulación | **Cadena de suministro**: dependencia, imagen o Action comprometida | `uv.lock` fijo; imágenes con versión fija; **Actions fijadas por SHA** (`test_ci_actions_are_pinned_to_commit_sha`); `pip-audit` y **Trivy** en el CI; Dependabot. Trivy se descarga en una versión segura y **se verifica su checksum** antes de ejecutarlo | Firmas (cosign) y SBOM firmado; registro privado de imágenes |
| T9 | Denegación de servicio | Una avalancha de pagos llena la fila | Alertas `HalconFilaCreciendo` y de SLO; escalado a 3 scorers (runbook) | Autoescalado por lag; límites de tasa por comercio |
| T10 | Repudio | "Yo no tomé esa decisión" | Cada decisión sale con puntaje, reglas y razones SHAP al topic `decisions`; los casos guardan quién los cerró y por qué | Bitácora inmutable con retención regulatoria |

## El incidente que enseñó la T8

En marzo de 2026, atacantes publicaron un **Trivy v0.69.4 envenenado** y reescribieron los tags
de la GitHub Action `trivy-action`. Los pipelines que la usaban por tag ejecutaron un ladrón de
credenciales ([GHSA-69fq-xp46-6x23](https://osv.dev/vulnerability/GHSA-69fq-xp46-6x23)).
Por eso Halcón:

1. Fija cada Action a un **commit SHA**, que nadie puede mover.
2. No usa `trivy-action`: descarga el binario de una versión que el aviso declara segura y
   **verifica su SHA-256** contra el checksum oficial antes de ejecutarlo.

## Riesgo residual (lo que sigue abierto)

* Las 3 primeras pruebas de cada ataque adaptado todavía se aprueban: R005 necesita 4 para
  reconocer la ráfaga. Bajar el umbral a 3 afectaría a 14 clientes honestos por cada 364
  pruebas atrapadas.
* Un atacante que pruebe con más de $20, o despacio, esquiva R005. Le sale más caro y más lento,
  pero es posible. La defensa de fondo es reentrenar con los fraudes nuevos (**Fase 8**).
