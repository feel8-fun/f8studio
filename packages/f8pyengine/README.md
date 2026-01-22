## f8pyengine

Python engine-side runtime service (`serviceClass=f8.pyengine`).

Entry wiring lives in `f8pyengine/pyengine_service.py` and is exposed via `f8pyengine/main.py`:
- `python -m f8pyengine.main --describe`
- `python -m f8pyengine.main --service-id engine1 --nats-url nats://127.0.0.1:4222`
