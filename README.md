# Smart Agriculture AI Layer

Production-ready, modular AI microservice for the Smart Agriculture System.
This repository covers **only the AI layer** — it receives sensor data and
crop images (from your IoT layer / database) over a REST API and returns
crop recommendations, growth-stage detection, disease detection, fertilizer
and irrigation guidance, weather advisories, and a combined advisory, ready
to be consumed by your frontend dashboard.

---

## Architecture

```
IoT Sensors + Camera → Database → [THIS AI LAYER] → Advisory → Dashboard
```

```
AI/
├── datasets/             # Put your training data here (see datasets/README.md)
│   ├── sensor_data/       # CSV for crop recommendation training
│   └── crop_images/       # Images for stage + disease model training
├── config/                # All dynamic rules (JSON) — edit, don't hardcode
│   ├── app_settings.json
│   ├── fertilizer_rules.json
│   ├── irrigation_rules.json
│   ├── weather_rules.json
│   └── disease_treatment.json
├── preprocessing/         # Sensor + image preprocessing pipelines
├── training/              # One script per model
├── models/                # Trained model artifacts land here (.pkl / .h5)
├── inference/             # Loads models from registry, runs predictions
├── services/              # Rule-based modules (fertilizer/irrigation/weather)
├── advisory/              # Combines everything into one final advisory
├── api/                   # FastAPI routes + Pydantic schemas
├── utils/                 # Logger, config loader, model registry, exceptions
├── tests/                 # Pytest suite
├── logs/                  # Rotating log files written at runtime
├── main.py                # FastAPI app entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

Every threshold, NPK target, treatment text, and irrigation rule lives in a
JSON file under `config/` — not in Python code — so you can tune behavior
without redeploying.

---

## Step-by-step: what to do next

### Step 1 — Create a virtual environment
(You said you'll do this yourself — just make sure it's Python 3.11+.)

### Step 2 — Install requirements
```bash
cd AI
pip install -r requirements.txt
```

### Step 3 — Add your datasets
See `datasets/README.md` for the exact folder layout. Short version:

- **Sensor CSV** → `datasets/sensor_data/crop_recommendation.csv`
  (a synthetic sample file is already there so you can test the pipeline
  immediately — replace it with your real data before going to production)
- **Stage images** → `datasets/crop_images/<rice|wheat|maize>/stage/<seedling|tillering|vegetative|flowering|maturity>/*.jpg`
- **Disease images** → `datasets/crop_images/disease/<rice|wheat|maize>/<disease_class>/*.jpg`
  (example class folders are already created; add real photos into them, and
  register any new disease class name in `config/disease_treatment.json`)

### Step 4 — Train the models (run in this order)

```bash
# 1. Crop recommendation (tabular, fast — seconds)
python training/train_crop_model.py
# optional: --algorithm xgboost, --data path/to/your.csv

# 2. Growth stage classification (images, needs a GPU for speed but works on CPU)
python training/train_stage_model.py
# optional: --epochs 20 --batch-size 32

# 3. Disease detection (images)
python training/train_disease_model.py
# optional: --epochs 20 --batch-size 32
```

Each script saves its model + labels into `models/` automatically — paths
are read from `config/app_settings.json`, so you never need to edit code to
relocate them.

You can start using `/predict/crop`, `/predict/fertilizer`,
`/predict/irrigation`, and `/predict/weather` as soon as Step 4.1 finishes —
they don't depend on the image models. `/predict/stage` and
`/predict/disease` (and the image portions of `/predict/advisory`) become
available once 4.2 / 4.3 finish.

### Step 5 — Start the FastAPI server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000/docs` for interactive Swagger UI to try every
endpoint.

### Step 6 — Test the APIs
```bash
pytest tests/ -v
```
Rule-based and health/fertilizer/irrigation/weather tests run immediately.
The crop-prediction test auto-skips until Step 4.1 is done; add your own
image-based tests once you've trained those models.

Quick manual test:
```bash
curl -X POST http://localhost:8000/predict/fertilizer \
  -H "Content-Type: application/json" \
  -d '{"crop": "rice", "N": 60, "P": 30, "K": 55}'
```

### Step 7 — Docker deployment
```bash
cp .env.example .env
docker compose up --build
```
This builds the image, installs dependencies, and starts the API on port
8000. The `models/`, `datasets/`, `config/`, and `logs/` folders are bind-
mounted, so retraining on the host updates the running container without a
rebuild — just restart the container after retraining:
```bash
docker compose restart
```

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service + model load status |
| POST | `/predict/crop` | Crop recommendation from N/P/K/temperature/humidity/moisture |
| POST | `/predict/fertilizer` | Urea/DAP/MOP/organic recommendation from N/P/K |
| POST | `/predict/irrigation` | Irrigation need + amount from moisture/crop/stage |
| POST | `/predict/weather` | Weather-based advisory from temperature/rainfall forecast |
| POST | `/predict/stage` | Growth stage from a crop image (multipart upload) |
| POST | `/predict/disease` | Disease + treatment from a crop image (multipart upload) |
| POST | `/predict/advisory` | Combined advisory — all of the above in one call |

`/predict/stage`, `/predict/disease`, and `/predict/advisory` accept
`multipart/form-data` (images), the rest accept JSON.

---

## Connecting to your IoT layer and dashboard

- Your IoT device/database layer should call these endpoints with the
  latest sensor reading + image whenever new data arrives (or your dashboard
  backend can poll the database and call this AI layer on a schedule).
- The dashboard should call `/predict/advisory` for the all-in-one farmer
  recommendation, or call individual endpoints if it wants to show each
  module's result separately (e.g. a dedicated "Disease Reports" panel).
- This AI layer is stateless — it does **not** read from or write to your
  database itself. Pass it the readings/images directly in each request,
  and store its JSON response in your database/dashboard layer if you want
  history (the "historical trends" feature lives in your frontend/database
  layer, not here).

---

## Retraining / extending

- New crop for recommendation: add rows to your sensor CSV with that crop's
  label, retrain `train_crop_model.py` — no code changes.
- New growth stage or disease class: add the image folder, retrain the
  relevant script. For disease, also add a matching entry to
  `config/disease_treatment.json`.
- New fertilizer/irrigation/weather rule: edit the corresponding JSON file
  in `config/` — no restart needed for rule services (configs are read at
  call time and cached; call `ConfigLoader.reload_all()` or just restart the
  process to pick up changes immediately).
