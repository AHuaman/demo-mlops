# Primer 3: CI/CD

Un equipo que modifica código todos los días necesita una forma automática de saber si algo se rompió. CI (Continuous Integration) es esa forma: cada vez que subes cambios, un servidor ejecuta las pruebas por ti.

## Qué pasa cuando haces push

1. Subes tu rama a GitHub (`git push`) o abres un pull request.
2. GitHub lee los archivos de `.github/workflows/` y ejecuta el que corresponda.
3. Cada paso se marca como exitoso o fallido. El pull request muestra un check verde o rojo.

## Workflow, job, step

- Un **workflow** es un archivo YAML (el nuestro es `.github/workflows/ci.yml`).
- Un **job** es un grupo de pasos que corre en una máquina limpia.
- Un **step** es un comando o una acción reutilizable dentro del job.

Nuestro workflow tiene tres jobs:

| Job | Qué hace |
| --- | --- |
| `test` | Instala dependencias, ejecuta `ruff check`, `ruff format --check` y `pytest` |
| `quality-gate` | Ejecuta las pruebas lentas del flujo completo (entrenar, evaluar y promover un modelo) |
| `docker` | Construye la imagen, la ejecuta y comprueba que `/health` responde 200 y `/ready` responde 503 sin modelo |

`quality-gate` y `docker` declaran `needs: test`, así que solo empiezan si `test` pasó.

## Matrix

Con `matrix`, un mismo job se ejecuta varias veces con valores distintos. Nuestro job `test` corre en `ubuntu-latest` y en `windows-latest`. Así detectamos un fallo que solo ocurre en Windows, donde trabajan muchos estudiantes.

## Por qué un check rojo bloquea el merge

En GitHub puedes marcar ciertos checks como obligatorios para una rama. Si alguno está en rojo, el botón de merge queda deshabilitado. La regla protege `main`: nadie incorpora código que no pasa las pruebas, ni siquiera con prisa. (Esta regla se configura en el repositorio. Se muestra en clase, no se asume.)

## Pre-commit

`.pre-commit-config.yaml` ejecuta `ruff` en tu máquina antes de cada commit. Es el mismo chequeo que CI, pero con respuesta inmediata. CI sigue siendo la verificación definitiva, porque corre en una máquina limpia y nadie puede saltársela.

## Tiempos de una corrida real

La corrida verde de nuestro repositorio, por job (medidos en GitHub Actions, se repiten con variaciones):

| Job | Duración |
| --- | --- |
| `test` (ubuntu-latest) | 1 min 39 s |
| `test` (windows-latest) | 5 min 46 s |
| `quality-gate` | 1 min 3 s |
| `docker` | 1 min 45 s |

## CI y CD

- **CI** responde "¿el código sigue funcionando?". Lo hacemos en clase.
- **CD** (Continuous Delivery o Deployment) responde "¿cómo llega a producción?". Nuestro workflow no despliega nada.

El siguiente paso natural sería que el job `docker` publique la imagen en un registry (por ejemplo, GitHub Container Registry) y que otro job la despliegue en un servicio de nube. No lo hacemos en clase.

## Verifica que funciona

Ejecuta localmente los mismos chequeos que el job `test`:

```bash
uv run ruff check
uv run ruff format --check
```

Con el repositorio en orden, `ruff check` imprime `All checks passed!` y `ruff format --check` imprime el número de archivos con formato correcto (por ejemplo, `68 files already formatted`).

## Dónde lo usamos

Revisamos el workflow y el quality gate en [Calidad y CI](../guias/05-calidad-y-ci.md).
