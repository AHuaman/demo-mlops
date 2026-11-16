## Tiempos medidos

Medidos en la Mac del autor (Apple Silicon, `loans_per_month=2500`, Python 3.13). En otras maquinas variaran.

| Comando | Tiempo |
|---|---|
| Corrida completa del laboratorio (`uv run pytest -m slow tests/integration/test_lab.py`: lago 2019-2025, entrenamiento, 36 meses de produccion y los dos incidentes) | 29 s (32 s de reloj con el arranque de `uv`) |
| `uv run pytest -m slow` (laboratorio mas `test_flows.py`) | 49 s (52 s de reloj) |
