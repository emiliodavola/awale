# Awale — Specification-driven Awale (Oware) RL (Julia)

Resumen breve
------------

Proyecto de investigación para desarrollar un sistema AlphaZero-like para Awale (Oware/Awari) en Julia. Enfoque: especificaciones formales → contratos → pruebas de propiedades → implementación. Este repositorio contiene la especificación completa en `spec/` y el código implementado en `src/`.

🚀 Estado Actual (v2.1)
------------------------

**Fases 1, 2 y 3 han sido completadas exitosamente.**

- ✅ **Core Logic & Termination:** Implementación de Grand Slam y detección de empates.
- ✅ **MCTS Architecture:** Integración de la Transposition Table (TT) para optimización de búsqueda.
- ✅ **Performance Optimization:** Reducción de asignaciones de memoria en los hot-paths.

Siguientes pasos recomendados
----------------------------

1. **Phase 4: Validation & Benchmarking**
   - Realizar pruebas de integridad de las nuevas reglas de terminación.
   - Comparar el rendimiento (Win Rate) de un agente con TT vs. Vanilla MCTS.
   - Ejecutar microbenchmarks para cuantificar el ahorro de memoria.

Cómo ejecutar pruebas (desarrollo)
--------------------------------

Recomendado: usar el entorno del proyecto de Julia.

- Desde la raíz del repo, instanciar dependencias:

  ```powershell
  julia --project=. -e "using Pkg; Pkg.instantiate()"
  ```

- Ejecutar tests rápidos de desarrollo (incluyen los tests de especificación):

  ```powershell
  julia --project=. test/runtests.jl
  ```

Estructura clave
----------------

- `spec/`: especificaciones formales y contratos (autoritativo)
- `src/`: código fuente (módulos: `Awale/State`, `Awale/Env`, `Awale/MCTS`, `Awale/Model`, `Awale/Training`, `Awale/Utils`)
- `test/`: pruebas unitarias y de invariantes
- `.github/`: CI y copilot-instructions

Flujo de trabajo (gitflow)
-------------------------

- Ramas permanentes: `main` (estable), `dev` (integración)
- Crear feature branches desde `dev`: `feature/<nombre>`

Contacto
-------

Mantén la especificación en `spec/` como fuente de verdad antes de cambiar la lógica.
