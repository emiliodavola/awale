# Awale — Specification-driven Awale (Oware) RL (Julia)

## Resumen breve

Proyecto de investigación para desarrollar un sistema AlphaZero-like para Awale (Oware/Awari) en Julia. Enfoque: especificaciones formales → contratos → pruebas de propiedades → implementación. `spec/` documenta los contratos actuales del sistema y `src/` contiene la implementación.

## 🚀 Estado actual

El proyecto ya tiene una pipeline funcional de entrenamiento/evaluación, pero sigue siendo un repositorio de investigación. La regla principal sigue siendo: **no sacar conclusiones arquitectónicas antes de validar bien la calidad de la señal de entrenamiento**.

### Configuración del proyecto

El sistema usa `config.toml` como configuración local de runtime para entrenamiento, evaluación y juego. La plantilla versionada vive en `config.toml.example`; copiála a `config.toml` y editala localmente. Las semillas de inicialización/bootstrapping del entrenamiento y límites compartidos como `training.max_turns` se controlan desde ahí, así que la reproducibilidad y los topes de runtime no quedan hardcodeados. La arquitectura del modelo vive en `src/Awale/config.toml` de forma local/no versionada; la plantilla versionada vive en `src/Awale/config.toml.example`. Los checkpoints y logs de entrenamiento se agrupan por arquitectura bajo `checkpoints/<arquitectura>/` para separar corridas de MLP, CNN y futuras variantes.

### Scripts principales

- `train.jl` — continúa o ejecuta entrenamiento y actualiza checkpoints.
- `baseline_eval.jl` — evalúa un checkpoint contra `RandomAgent` y `HeuristicAgent`.
- `checkpoint_arena.jl` — compara checkpoints entre sí con `0`, `50` y `200` simulaciones.
- `play.jl` — corre una única partida de exhibición con logs de tablero y agentes configurables por CLI.
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

El entrenamiento usa tu `config.toml` local, guarda `model_last.bin`, `model_best.bin`, `model_final.bin` y snapshots `model_iter_N.bin` bajo `checkpoints/<arquitectura>/` solo en hitos automáticos: iteración 1, potencias de 2 y múltiplos de `checkpoint_every`. El log de entrenamiento también se guarda bajo `checkpoints/<arquitectura>/log/` e incluye la arquitectura activa en el nombre y en el contenido copiado.

Los checkpoints `.bin` se tratan como artefactos locales de confianza: el flujo actual usa `Serialization` para checkpoints generados por el propio repo, no como un formato para cargar archivos arbitrarios de terceros.

### 4. Evaluación rápida contra baselines

```powershell
julia --project=. .\baseline_eval.jl
```

Usalo como sanity check. Si el modelo ya domina `RandomAgent` y `HeuristicAgent`, esa evaluación deja de ser discriminante.

### 5. Arena entre checkpoints

```powershell
julia --project=. .\checkpoint_arena.jl
```

Usalo para responder si hay progreso real entre checkpoints de la misma pipeline. Hoy es la evaluación más útil para distinguir señal de ruido porque compara checkpoints consecutivos sobre una opening suite reproducible. El arena resuelve checkpoints del namespace de la arquitectura activa y conserva fallback legacy al root para artefactos viejos.

Si querés comparar arquitecturas distintas en la misma corrida, usá la API interna `run_duel` con selectores explícitos por arquitectura. Ejemplo en Julia:

```julia
include("checkpoint_arena.jl")
run_duel("best", "best"; architecture_a="mlp", architecture_b="cnn", sims=0, games=2)
```

Eso mantiene el flujo por defecto intacto, pero te deja chequear `mlp(best)` contra `cnn(best)` de forma explícita.

### 6. Partidas de exhibición

`play.jl` está pensado para una sola partida visible en terminal. Permite elegir ambos agentes por línea de comando: `human`, `best`, `last`, `final` o un path explícito a un checkpoint. Los alias `best/last/final` resuelven primero el namespace de la arquitectura activa (`checkpoints/<arquitectura>/...`) y después caen al path legacy del root si hace falta. También acepta `--sims` para controlar cuántas simulaciones usa cada agente IA, `--max-turns` para limitar la duración de la partida, `--seed` para reproducir una exhibición estocástica y `--deterministic` para desactivar esa variación.

Ejemplos:

```powershell
julia --project=. .\play.jl --agent1 best --agent2 human
julia --project=. .\play.jl --agent1 best --agent2 final
julia --project=. .\play.jl --agent1 checkpoints\<arquitectura>\model_best.bin --agent2 human
julia --project=. .\play.jl --agent1 best --agent2 final --sims 200 --max-turns 120
julia --project=. .\play.jl --agent1 best --agent2 final --seed 42
julia --project=. .\play.jl --agent1 best --agent2 final --deterministic
```

La interfaz muestra el humano abajo, la fila superior en orden inverso para respetar la siembra antihoraria, las capturadas de ambos jugadores y un banner claro por turno.

### 7. Microbenchmarks

```powershell
julia --project=. .\scripts\benchmarks.jl
```

Usalo solo para medir hot paths y comparar impacto de cambios de performance.

## Estructura clave

- `spec/`: contratos actuales del sistema y notas de diseño
- `src/`: código fuente (módulos: `Awale/State`, `Awale/Env`, `Awale/MCTS`, `Awale/Model`, `Awale/Training`, `Awale/Utils`)
- `test/`: pruebas unitarias, invariantes y regresión
- `checkpoints/`: modelos y estado de entrenamiento, agrupados por arquitectura
- `scripts/`: entrypoints auxiliares como microbenchmarks
- `.github/`: CI

## Flujo de trabajo (gitflow)

- Ramas permanentes: `main` (estable), `dev` (integración)
- Crear feature branches desde `dev`: `feature/<nombre>`
