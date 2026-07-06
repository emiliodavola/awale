# Awale — Specification-driven Awale (Oware) RL (Julia)

## Resumen breve

Proyecto de investigación para desarrollar un sistema AlphaZero-like para Awale (Oware/Awari) en Julia. Enfoque: especificaciones formales → contratos → pruebas de propiedades → implementación. `spec/` documenta los contratos actuales del sistema y `src/` contiene la implementación.

## 🚀 Estado actual

El proyecto ya tiene una pipeline funcional de entrenamiento/evaluación, pero sigue siendo un repositorio de investigación. La regla principal sigue siendo: **no sacar conclusiones arquitectónicas antes de validar bien la calidad de la señal de entrenamiento**.

### Configuración del proyecto

El sistema utiliza `config.toml` como configuración principal de runtime para entrenamiento, evaluación y juego. La arquitectura del modelo vive en `src/Awale/config.toml` y se referencia desde `config.toml` mediante `model_config_path`.

### Scripts principales

- `train.jl` — continúa o ejecuta entrenamiento y actualiza checkpoints.
- `baseline_eval.jl` — evalúa un checkpoint contra `RandomAgent` y `HeuristicAgent`.
- `checkpoint_arena.jl` — compara checkpoints entre sí con `0`, `50` y `200` simulaciones.
- `play.jl` — corre partidas de exhibición con logs de tablero.
- `scripts/benchmarks.jl` — microbenchmarks de hot paths (`encode_state`, `select_puct`, `backup`).

## Hoja de ruta experimental

### Regla central

> **No optimizar la arquitectura de red antes de validar la representación de estado y el pipeline de entrenamiento.**

### Orden de trabajo recomendado

1. **State encoding**
   - usar una representación estructurada `(C, 12)`
   - evitar feature engineering excesivo
2. **Strong MLP baseline**
   - establecer una baseline fuerte antes de introducir inductive bias más complejo
3. **Self-play + MCTS scaling**
   - comparar siempre con el mismo presupuesto de búsqueda y self-play
4. **Architecture benchmarking**
   - recién después comparar MLP contra variantes más complejas como ResNet1D

### Métricas que importan

- **Sample efficiency**
- **Win rate / arena performance**
- **Policy entropy**
- **Value stability**
- **Elo relativo**

## Siguientes pasos recomendados

1. Seguir con benchmarks entre checkpoints de la pipeline nueva.
2. Medir cuánto aporta la red sola (`0 sims`) versus red + MCTS (`50/200 sims`).
3. Recién después decidir si hace falta cambiar arquitectura.

## Cómo usar el repositorio

Recomendado: usar siempre el entorno del proyecto de Julia.

### 1. Instanciar dependencias

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### 2. Correr la suite de tests

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
```

Alternativa local directa:

```powershell
julia --project=. test/runtests.jl
```

### 3. Entrenar

```powershell
julia --project=. .\train.jl
```

El entrenamiento usa `config.toml`, guarda `model_last.bin`, `model_best.bin`, `model_final.bin` y snapshots `model_iter_N.bin` solo en hitos automáticos: iteración 1, potencias de 2 y múltiplos de `checkpoint_every`.

### 4. Evaluación rápida contra baselines

```powershell
julia --project=. .\baseline_eval.jl
```

Usalo como sanity check. Si el modelo ya domina `RandomAgent` y `HeuristicAgent`, esa evaluación deja de ser discriminante.

### 5. Arena entre checkpoints

```powershell
julia --project=. .\checkpoint_arena.jl
```

Usalo para responder si hay progreso real entre checkpoints de la misma pipeline. Hoy es la evaluación más útil para distinguir señal de ruido porque compara checkpoints consecutivos sobre una opening suite reproducible.

### 6. Partidas de exhibición

```powershell
julia --project=. .\play.jl
```

### 7. Microbenchmarks

```powershell
julia --project=. .\scripts\benchmarks.jl
```

Usalo solo para medir hot paths y comparar impacto de cambios de performance.

## Estructura clave

- `spec/`: contratos actuales del sistema y notas de diseño
- `src/`: código fuente (módulos: `Awale/State`, `Awale/Env`, `Awale/MCTS`, `Awale/Model`, `Awale/Training`, `Awale/Utils`)
- `test/`: pruebas unitarias, invariantes y regresión
- `checkpoints/`: modelos y estado de entrenamiento
- `scripts/`: entrypoints auxiliares como microbenchmarks
- `.github/`: CI

## Flujo de trabajo (gitflow)

- Ramas permanentes: `main` (estable), `dev` (integración)
- Crear feature branches desde `dev`: `feature/<nombre>`
