# Awale

## Resumen breve

Proyecto de investigación para construir un sistema Awale estilo AlphaZero en Julia. El flujo es primero especificaciones: contratos formales → pruebas → implementación. Las reglas canónicas del juego están en `spec/01_game_rules/README.md` y `spec/03_environment_api/README.md`; `src/` contiene la implementación. Una versión espejo completa en español de `README.md` se mantiene en `README_ES.md`.

## Estado actual

El proyecto ya tiene una pipeline funcional de entrenamiento/evaluación, pero sigue siendo un repositorio de investigación. La regla principal sigue siendo: **no sacar conclusiones arquitectónicas antes de validar la calidad de la señal de entrenamiento**.

Nota: este archivo es una traducción/mirror en español de `README.md`; algunos términos técnicos pueden mantenerse en inglés cuando forman parte de la convención del repositorio.

### Configuración del proyecto

El sistema usa `config.toml` como configuración local de runtime para entrenamiento, evaluación y juego. La plantilla versionada vive en `config.toml.example`; cópiela en `config.toml` y edítela localmente. Las semillas de arranque del entrenamiento y límites compartidos como `training.max_turns` se controlan allí, así que la reproducibilidad y los límites de runtime no quedan hardcodeados.

La arquitectura del modelo vive en `src/Awale/config.toml` de forma local/no versionada; la plantilla versionada vive en `src/Awale/config.toml.example`. Los checkpoints y logs de entrenamiento se agrupan por arquitectura bajo `checkpoints/<architecture>/` para separar corridas de MLP, CNN y futuras variantes.

### Scripts principales

- `train.jl` — continúa o ejecuta entrenamiento y actualiza checkpoints.
- `publish_hf.jl` — arma un release bundle, genera una tarjeta de modelo en inglés para Hugging Face y, opcionalmente, lo publica.
- `checkpoint_arena.jl` — compara checkpoints entre sí con `0`, `50` y `200` simulaciones.
- `play.jl` — corre una única partida de exhibición con logs del tablero y agentes configurables por CLI.
- `scripts/benchmarks.jl` — microbenchmarks de hot paths (`encode_state`, `select_puct`, `backup`).

## Hoja de ruta experimental

### Regla central

> **No optimizar la arquitectura de red antes de validar la representación de estado y la pipeline de entrenamiento.**

### Orden de trabajo recomendado

1. **Codificación del estado**
   - usar una representación estructurada `(C, 12)`
   - evitar feature engineering excesivo
2. **Baseline fuerte de MLP**
   - establecer una baseline fuerte antes de introducir inductive bias más complejo
3. **Escalado de self-play + MCTS**
   - comparar siempre con el mismo presupuesto de búsqueda y self-play
4. **Benchmarking de arquitecturas**
   - recién después comparar MLP contra variantes más complejas como ResNet1D

### Métricas que importan

- **Eficiencia muestral**
- **Win rate / performance en arena**
- **Entropía de policy**
- **Estabilidad de value**
- **Elo relativo**

## Siguientes pasos recomendados

1. Seguir con benchmarks entre checkpoints de la nueva pipeline.
2. Medir cuánto aporta la red sola (`0 sims`) versus red + MCTS (`50/200 sims`).
3. Recién después decidir si hace falta cambiar la arquitectura.

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

El entrenamiento usa tu `config.toml` local, guarda `model_last.bin`, `model_best.bin`, `model_final.bin` y snapshots `model_iter_N.bin` bajo `checkpoints/<architecture>/` solo en hitos automáticos: iteración 1, potencias de 2 y múltiplos de `checkpoint_every`. El log de entrenamiento también se guarda bajo `checkpoints/<architecture>/log/` e incluye la arquitectura activa en el nombre y en el contenido copiado.

Los checkpoints `.bin` se tratan como artefactos locales de confianza: el flujo actual usa `Serialization` para checkpoints generados por el propio repo, no como un formato para cargar archivos arbitrarios de terceros. Las publicaciones públicas en Hugging Face usan exportaciones seguras en `Float32` en lugar de checkpoints crudos serializados por Julia.

### 4. Publicar una corrida terminada en Hugging Face

La publicación es un paso explícito posterior al entrenamiento. Se ejecuta localmente o en su VM después de que la corrida termina.

```powershell
$env:HF_TOKEN = "<tu-token>"
hf auth login --token $env:HF_TOKEN --add-to-git-credential
```

Después, arme o publique el bundle:

```powershell
julia --project=. .\publish_hf.jl --dry-run
julia --project=. .\publish_hf.jl --stage
julia --project=. .\publish_hf.jl --publish --repo-id your-namespace/awale-results
```

El bundle queda acotado por arquitectura bajo `checkpoints/<architecture>/release/<release_id>/`, y cada corrida conserva su propio `release_summary.toml` ahí para que las corridas viejas sigan siendo descubribles. `publish_hf.jl` toma por defecto el resumen más nuevo de esa arquitectura.

El flujo de staging también escribe un `README.md` en inglés en la raíz del bundle a partir de los metadatos de `release_summary.toml`, y el flujo de publicación sube ese archivo a la raíz del repositorio de Hugging Face para que se renderice directamente como tarjeta de modelo. La publicación pública contiene exportaciones seguras en `Float32`, no los checkpoints `.bin` crudos.

Incluye:

- `model_final.f32`, `model_best.f32`, `model_last.f32`
- `README.md` (tarjeta de modelo de Hugging Face en inglés)
- `training_state.toml`
- snapshots copiados de runtime/model config
- `release_summary.toml` y `manifest.toml`

### 5. Arena entre checkpoints

```powershell
julia --project=. .\checkpoint_arena.jl
```

Úselo para responder si hay progreso real entre checkpoints de la misma pipeline. Hoy es la evaluación más útil para distinguir señal de ruido porque compara checkpoints consecutivos sobre una opening suite reproducible. La arena resuelve checkpoints del namespace de la arquitectura activa y conserva un fallback legacy al root para artefactos viejos.

Si desea comparar arquitecturas distintas en la misma corrida, use la API interna `run_duel` con selectores explícitos por arquitectura. Ejemplo en Julia:

```julia
include("checkpoint_arena.jl")
run_duel("best", "best"; architecture_a="mlp", architecture_b="cnn", sims=0, games=2)
```

Eso mantiene intacto el flujo por defecto, pero le permite comprobar `mlp(best)` contra `cnn(best)` de forma explícita.

### 6. Partidas de exhibición

`play.jl` está pensado para una sola partida visible en terminal. Permite elegir ambos agentes por línea de comando: `human`, `best`, `last`, `final` o una ruta explícita a un checkpoint. Los alias `best/last/final` resuelven primero el namespace de la arquitectura activa (`checkpoints/<architecture>/...`) y después caen al path legacy del root si hace falta. También acepta `--sims` para controlar cuántas simulaciones usa cada agente IA, `--max-turns` para limitar la duración de la partida, `--seed` para reproducir una exhibición estocástica y `--deterministic` para desactivar esa variación.

Ejemplos:

```powershell
julia --project=. .\play.jl --agent1 best --agent2 human
julia --project=. .\play.jl --agent1 best --agent2 final
julia --project=. .\play.jl --agent1 checkpoints\<architecture>\model_best.bin --agent2 human
julia --project=. .\play.jl --agent1 best --agent2 final --sims 200 --max-turns 120
julia --project=. .\play.jl --agent1 best --agent2 final --seed 42
julia --project=. .\play.jl --agent1 best --agent2 final --deterministic
```

La interfaz muestra al humano abajo, la fila superior invertida para respetar la siembra antihoraria, las capturadas de ambos jugadores y un banner claro por turno.

### 7. Microbenchmarks

```powershell
julia --project=. .\scripts\benchmarks.jl
```

Úselo solo para medir hot paths y comparar el impacto de cambios de performance.

## Estructura clave

- `spec/` — contratos actuales del sistema y notas de diseño
- `src/` — código fuente (módulos: `Awale/State`, `Awale/Env`, `Awale/MCTS`, `Awale/Model`, `Awale/Training`, `Awale/Utils`)
- `test/` — pruebas unitarias, invariantes y cobertura de regresión
- `checkpoints/` — modelos y estado de entrenamiento, agrupados por arquitectura
- `scripts/` — entrypoints auxiliares como microbenchmarks
- `.github/` — CI

## Gitflow

- Ramas permanentes: `main` (estable), `dev` (integración)
- Crear feature branches desde `dev`: `feature/<nombre>`
