# Primer 1: API y HTTP

Una API web es un programa que espera mensajes de otros programas y responde. En la sesión, nuestro modelo de riesgo de crédito vive detrás de una API: una app envía los datos de una solicitud y recibe la probabilidad de incumplimiento.

## Request y response

Toda la comunicación es un par de mensajes HTTP.

- El **request** lleva un verbo, una dirección (URL), y a veces un cuerpo con datos.
- El **response** lleva un código de estado y, casi siempre, un cuerpo.

Los cuerpos usan JSON, un formato de texto con llaves y pares clave-valor:

```json
{"bureau_score": 780, "region": "lima"}
```

## Endpoints y verbos

Un endpoint es una dirección de la API con una función concreta. El verbo indica la intención:

- `GET` pide información y no cambia nada.
- `POST` envía datos para que la API los procese.

Los endpoints de nuestra API (`src/credit_mlops/api/app.py`):

| Verbo y ruta | Qué hace |
| --- | --- |
| `GET /health` | Responde si el proceso está vivo |
| `GET /ready` | Responde si el modelo champion ya cargó |
| `GET /model` | Nombre, versión y ventana de entrenamiento del modelo |
| `POST /predict` | Recibe solicitudes y devuelve probabilidad, decisión y explicación |
| `GET /docs` | Documentación interactiva (ver más abajo) |

## Códigos de estado

El código resume el resultado. Estos son los que usa nuestra API.

| Código | Significado | Cuándo ocurre aquí |
| --- | --- | --- |
| 200 | Todo bien | `/health` siempre; `/ready` y `/predict` con modelo cargado |
| 422 | Los datos enviados no cumplen el esquema | Falta un campo, o `bureau_score` no es un número |
| 503 | El servicio no puede atender ahora | No hay modelo champion disponible (por ejemplo, MLflow está apagado) |

Un 4xx significa que el error está en el request. Un 5xx significa que el error está del lado del servidor. Por eso `/health` responde 200 aunque el modelo no esté (el proceso vive) y `/ready` responde 503 (aún no puede predecir).

## curl

`curl` envía requests desde la terminal. En macOS se llama `curl`. En Windows PowerShell, `curl` es un alias de otro comando, así que escribe `curl.exe`.

macOS:

```bash
curl http://localhost:8000/health
```

Windows (PowerShell):

```powershell
curl.exe http://localhost:8000/health
```

Para un `POST` con el archivo de ejemplo `docs/guias/ejemplo_solicitud.json`:

macOS:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @docs/guias/ejemplo_solicitud.json
```

Windows (PowerShell):

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@docs/guias/ejemplo_solicitud.json"
```

Una respuesta real de nuestra API para las dos solicitudes del ejemplo (recortada). La versión del modelo y los números salen de la corrida del instructor y serán distintos en tu máquina:

```json
{"model_name": "credit_default_pd", "model_version": "1", "threshold": 0.15,
 "predictions": [
   {"probability_default": 0.0117, "decision": "approve", ...},
   {"probability_default": 0.7481, "decision": "review", ...}]}
```

Cada predicción incluye también una explicación con los tres factores que más pesaron.

## OpenAPI y /docs

FastAPI genera una descripción de la API en formato OpenAPI a partir del código. `/docs` la muestra como una página donde puedes leer cada endpoint, ver el esquema del cuerpo y enviar requests desde el navegador con el botón "Try it out". Es la forma más rápida de explorar una API que no conoces.

## Verifica que funciona

Con la API corriendo (la sesión te indica cómo iniciarla), ejecuta:

```bash
curl http://localhost:8000/health
```

En Windows usa `curl.exe`. Respuesta esperada:

```json
{"status":"ok"}
```

Abre también http://localhost:8000/docs en el navegador. Debes ver la lista de endpoints.

## Dónde lo usamos

Servimos el modelo con esta API en [Servir el modelo](../guias/04-sirviendo-el-modelo.md).
