<<<<<<< ours
# Awale — Specification-driven Awale (Oware) RL (Julia)

Resumen breve
------------

Proyecto de investigación para desarrollar un sistema AlphaZero-like para Awale (Oware/Awari) en Julia. Enfoque: especificaciones formales → contratos → pruebas de propiedades → implementación. Este repositorio contiene la especificación completa en spec/ y un esqueleto de código en src/ con pruebas iniciales.

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

- spec/: especificaciones formales y contratos (autoritativo)
- src/: código fuente (módulos: Awale/State, Awale/Env, Awale/Utils)
- test/: pruebas unitarias y de invariantes
- .github/: CI y copilot-instructions

Flujo de trabajo (gitflow)
-------------------------

- Ramas permanentes: main (estable), dev (integración)
- Crear feature branches desde dev: feature/<nombre>

Siguientes pasos recomendados
----------------------------

1. Completar property-based tests y fixtures (Phase 1 finalizar)
2. Implementar transición de reglas y pruebas de integridad (Phase 2)
3. Reintroducir StaticArrays/Flux y preparar entorno reproducible (Project/Manifest)

Contacto
-------

Mantené la especificación en spec/ como fuente de verdad antes de cambiar la lógica.
||||||| base
=======
# Awale — Specification-driven Awale (Oware) RL (Julia)

Resumen breve
------------

Proyecto de investigación para desarrollar un sistema AlphaZero-like para Awale (Oware/Awari) en Julia. Enfoque: especificaciones formales → contratos → pruebas de propiedades → implementación. Este repositorio contiene la especificación completa en spec/ y un esqueleto de código en src/ con pruebas iniciales.

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

- spec/: especificaciones formales y contratos (autoritativo)
- src/: código fuente (módulos: Awale/State, Awale/Env, Awale/Utils)
- test/: pruebas unitarias y de invariantes
- .github/: CI y copilot-instructions

Flujo de trabajo (gitflow)
-------------------------

- Ramas permanentes: main (estable), dev (integración)
- Crear feature branches desde dev: feature/<nombre>

Siguientes pasos recomendados
----------------------------

1. Completar property-based tests y fixtures (Phase 1 finalizar)
2. Implementar transición de reglas y pruebas de integridad (Phase 2)
3. Reintroducir StaticArrays/Flux y preparar entorno reproducible (Project/Manifest)

Contacto
-------

Mantené la especificación en spec/ como fuente de verdad antes de cambiar la lógica.
>>>>>>> theirs
