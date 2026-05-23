# CareBridge Behavior Analysis AI

Sensor time-series data is transformed into fixed-length behavior windows and classified with an LSTM model. The project is organized so that preprocessing, training, inference, API, models, and shared utilities have separate responsibilities.

## Project Structure

```text
src/
  preprocessing/
    features.py              Feature cleaning, encoding, and mapping logic
    windowing.py             Sequence window generation logic
    run_preprocessing.py     Preprocessing CLI entry point
    make_windows.py          Window creation CLI entry point
    restructure_data.py      Placeholder for raw-data restructuring
  train/
    dataset.py               NPZ dataset loader
    engine.py                Training and validation loops
    train_lstm.py            LSTM training CLI entry point
  inference/
    predictor.py             Model checkpoint loading and label loading
  api/
    app_api.py               FastAPI application
    app_gradio.py            Gradio demo application
  models/
    lstm.py                  LSTM baseline model
  utils/
    config.py                YAML config loader and path resolver
    io.py                    Shared file I/O helpers

data/
notebooks/
configs/
output/
requirements.txt
README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Pipeline

Run commands from the `behavior_analysis` directory.

```powershell
.\.venv\Scripts\python.exe src\preprocessing\run_preprocessing.py
.\.venv\Scripts\python.exe src\preprocessing\make_windows.py
.\.venv\Scripts\python.exe src\train\train_lstm.py
```

The original training data may be absent locally. In that case, preprocessing and training will fail at the missing input path, which is expected.

## API

```powershell
.\.venv\Scripts\python.exe -m uvicorn --app-dir src api.app_api:app --host 127.0.0.1 --port 8000
```

Docs:

```text
http://127.0.0.1:8000/docs
```

## Gradio Demo

```powershell
.\.venv\Scripts\python.exe src\api\app_gradio.py
```
