# Plan de Desarrollo: Agente AlphaZero-style para Awalé (Oware)

Este documento define el orden de implementación y decisiones técnicas obligatorias antes de optimizar la arquitectura de la red neuronal. El objetivo es maximizar el rendimiento real (Elo) mediante experimentación controlada, minimizando suposiciones arquitectónicas.

## REGLA CENTRAL

> **NO optimizar la arquitectura de red antes de optimizar la representación del estado y el pipeline de entrenamiento.**

---

## PRIORIDAD GLOBAL (ORDEN OBLIGATORIO DE IMPLEMENTACIÓN)

### FASE 1 — Encoding del estado (CRÍTICO / BLOQUEANTE)

**Objetivo:** Reemplazar la representación plana por una representación estructurada para permitir el uso de convoluciones en fases posteriores.

* **Requisito de forma:** El estado debe transformarse de un vector plano a un tensor estructurado.
* **Shape recomendado:** `(C, 12)` (Canales $\times$ Pozos).
* **Canales mínimos sugeridos:**
    1. Semillas en cada pozo (Normalized).
    2. Indicador del jugador activo (One-hot o escala).
    3. Capturas del Jugador A.
    4. Capturas del Jugador B.
* **Restricción:** No introducir features complejas manualmente (ingeniería de características excesiva). Mantener la representación más directa posible del estado.

### FASE 2 — Baseline fuerte con MLP (CONTROL EXPERIMENTAL)

**Objetivo:** Establecer un estándar de rendimiento competitivo para validar si el problema es de capacidad de red o de datos.

* **Arquitectura:** MLP escalada.
  * `Dense(256) -> ReLU`
  * `Dense(256) -> ReLU`
  * `Dense(256) -> ReLU`
* **Policy Head:** 6 logits (correspondientes a los 6 pozos legales), con máscara de legalidad aplicada post-red.
* **Value Head:** Salida escalar con activación `tanh` (rango $[-1, 1]$).
* **Parámetros objetivo:** $\approx 150k$ – $300k$.

### FASE 3 — Sistema de Self-play + MCTS (ESCALA DE DATOS)

**Objetivo:** Asegurar que el rendimiento final sea impulsado por la cantidad de datos y no por la complejidad de la arquitectura.

* **Requisitos:**
  * Implementación de ciclo de Self-play continuo.
  * MCTS con configuración fija para todos los experimentos.
  * **Presupuesto de simulaciones constante:** No comparar arquitecturas con diferentes volúmenes de MCTS o Self-play.

### FASE 4 — Benchmark de Arquitecturas (COMPARACIÓN CONTROLADA)

**Solo ejecutable tras completar Fases 1, 2 y 3.**

#### Candidatos a evaluar

* **A) MLP Baseline:** Arquitectura de Fase 2.
* **B) MLP Escalada (Opcional):** Mismo diseño, mayor densidad de neuronas.
* **C) ResNet1D Pequeña:**
  * `Conv1D` + 4 bloques residuales.
  * Canales: $32$ – $64$.
  * Input: Tensor $(C, 12)$.
  * Circular Padding (opcional).

#### Métricas obligatorias

1. **Elo Relativo:** Contra un baseline fijo.
2. **Sample Efficiency:** Ganancia de Elo por cada $N$ partidas de Self-play.
3. **Policy Entropy:** Velocidad de convergencia de la política.
4. **Value Loss:** Estabilidad de la estimación de valor.
5. **Win-rate en Arena:** En condiciones controladas (mismo MCTS budget).

---

## FASE 5 — CRITERIO DE DECISIÓN

Solo se autoriza la migración a **ResNet1D** si se cumple:

1. Mejora consistente en *Sample Efficiency*.
2. Mejora clara en *Elo* bajo el mismo presupuesto de entrenamiento.
3. Mejora estadísticamente significativa vs. *MLP Escalada*.

Si no se cumple, se mantiene la **MLP Escalada** como solución definitiva.

---

## REGLAS CRÍTICAS

1. **No cambios arquitectónicos** hasta completar Fases 1–3.
2. **No introducir inductive bias complejo** (ResNet/CNN) antes de validar la MLP.
3. **Comparación controlada:** Mismo MCTS, mismo Self-play, mismo Pipeline.
4. **No usar intuición** como criterio de decisión.

---

## RESULTADO ESPERADO

Un sistema cuyo rendimiento esté validado experimentalmente, determinando si el cuello de botella reside en el **Encoding**, el **Self-play/MCTS** o la **Arquitectura de Red**.
