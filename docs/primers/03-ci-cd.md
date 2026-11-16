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

## Cuándo un check rojo bloquea el merge

Un check rojo bloquea el botón de merge solo si una regla de protección de rama (branch protection o ruleset) exige ese check. Esa regla requiere un repositorio público o un plan de pago. En el repositorio privado de un plan gratuito, CI informa pero no bloquea. En un repositorio público, el dueño puede activar la regla y exigir el check. En clase tu repositorio es tuyo, así que la regla del equipo es: no se hace merge con CI en rojo.

## Pre-commit

`.pre-commit-config.yaml` ejecuta `ruff` en tu máquina antes de cada commit. Es el mismo chequeo que CI, pero con respuesta inmediata. CI sigue siendo la verificación definitiva, porque corre en una máquina limpia y no depende de tu máquina.

## Tiempos de una corrida real

Una corrida verde del instructor en GitHub Actions, por job. Son valores de una sola corrida, no una medición que puedas repetir; en la tuya variarán:

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

Revisamos el workflow y el quality gate en [Calidad y CI](../guias/05-ci.md).
