# Awale — Specification-driven Awale (Oware) RL (Julia)

## Resumen breve

Proyecto de investigación para desarrollar un sistema AlphaZero-like para Awale (Oware/Awari) en Julia. Enfoque: especificaciones formales → contratos → pruebas de propiedades → implementación. Este repositorio contiene la especificación completa en `spec/` y el código implementado en `src/`.

## 🚀 Estado actual

El proyecto ya tiene una pipeline funcional de entrenamiento/evaluación, pero sigue siendo un repositorio de investigación. La regla principal sigue siendo: **no sacar conclusiones arquitectónicas antes de validar bien la calidad de la señal de entrenamiento**.

### Configuración del Proyecto

El sistema utiliza `config.toml` como única configuración de runtime para entrenamiento, evaluación y juego. Ajustalo directamente según tu corrida.

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

## Cómo ejecutar pruebas (desarrollo)

Recomendado: usar el entorno del proyecto de Julia.

- Desde la raíz del repo, instanciar dependencias:

  ```powershell
  julia --project=. -e "using Pkg; Pkg.instantiate()"
  ```

- Ejecutar tests rápidos de desarrollo (incluyen los tests de especificación):

  ```powershell
  julia --project=. test/runtests.jl
  ```

## Estructura clave

- `spec/`: especificaciones formales y contratos (autoritativo)
- `src/`: código fuente (módulos: `Awale/State`, `Awale/Env`, `Awale/MCTS`, `Awale/Model`, `Awale/Training`, `Awale/Utils`)
- `test/`: pruebas unitarias, invariantes y regresión
- `.github/`: CI y copilot-instructions

## Flujo de trabajo (gitflow)

- Ramas permanentes: `main` (estable), `dev` (integración)
- Crear feature branches desde `dev`: `feature/<nombre>`
