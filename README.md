# CareBridge Behavior Analysis AI

CareBridge Behavior Analysis AI is the behavior prediction module for the CareBridge senior care service. It analyzes time-windowed sensor features and predicts a senior user's likely behavior state with an LSTM-based classifier.

This repository contains only the source code, notebooks, and dependency manifest for the behavior analysis module. Training data, model weights, generated windows, and runtime outputs are intentionally excluded.

## What It Does

The module converts sensor records into fixed-length behavior windows and trains an LSTM baseline model to classify behavior states. The trained model can then be served through either a FastAPI endpoint or a Gradio demo.

Expected input signals include environmental and activity-related features such as temperature, humidity, illuminance, activity IR, CO2, TVOC, time features, and user metadata fields.

## Mechanism

1. Raw sensor records are restructured into a consistent tabular format.
2. Preprocessing scripts clean and transform the records.
3. Window generation creates fixed-length 1-hour sequence samples.
4. The LSTM model learns sequence patterns from the generated windows.
5. Inference receives a recent sensor sequence and returns a predicted behavior class with probabilities.

## Project Structure

```text
app/
  app_api.py                     FastAPI inference server
  app_gradio.py                  Gradio demo app
notebooks/
  EDA.ipynb                      Exploratory data analysis
  Preprocessed_data_EDA.py.ipynb Preprocessed data analysis
src/
  restruct_data.py               Raw data restructuring
  preprocessing.py               Sensor data preprocessing
  make_behavior_window_1h.py     1-hour behavior window generation
  train_lstm_window_1h_baseline.py
requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Training

Place local training data under `data/`, then run the preprocessing and window generation scripts as needed:

```powershell
python src\preprocessing.py
python src\make_behavior_window_1h.py
python src\train_lstm_window_1h_baseline.py
```

Generated files and trained weights are written under ignored local directories such as `output/` or `model/`.

## Inference

FastAPI:

```powershell
uvicorn app.app_api:app --reload
```

Gradio:

```powershell
python app\app_gradio.py
```

## Repository Policy

The following artifacts are intentionally not tracked:

- training and validation data
- generated `.npz` window files
- trained model weights such as `.pt` and `.pth`
- runtime outputs and experiment logs
- local virtual environments

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.
