from __future__ import annotations

import random
import sys
import time
from collections import deque
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import numpy as np
import torch

from inference import get_device, load_labels, load_model
from utils.config import model_artifact_path


DEVICE = get_device()
ID_TO_LABEL = load_labels()
MODEL_PATH = model_artifact_path()
model, ckpt = load_model(path=MODEL_PATH, device=DEVICE)
WINDOW_SIZE = ckpt["window_size"]
INPUT_SIZE = ckpt["input_size"]


def make_sensor_row(step: int):
    temperature = round(random.uniform(20.0, 28.0), 2)
    humidity = round(random.uniform(35.0, 70.0), 2)
    illuminance = round(random.uniform(0.0, 700.0), 2)
    activity_ir = round(random.choice([0, 0, 0, 1, 2, 3, 5, 10]), 2)
    co2 = round(random.uniform(400.0, 1200.0), 2)
    tvoc = round(random.uniform(0.0, 0.8), 4)
    hour = step % 24
    weekday = (step // 24) % 7

    return [
        temperature,
        humidity,
        illuminance,
        activity_ir,
        co2,
        tvoc,
        np.log1p(max(activity_ir, 0)),
        np.log1p(max(co2, 0)),
        np.log1p(max(tvoc, 0)),
        np.log1p(max(illuminance, 0)),
        hour,
        weekday,
        78,
        0,
        1,
    ], {
        "step": step,
        "hour": hour,
        "weekday": weekday,
        "temperature": temperature,
        "humidity": humidity,
        "illuminance": illuminance,
        "activity_ir": activity_ir,
        "co2": co2,
        "tvoc": tvoc,
    }


@torch.no_grad()
def predict_from_window(window):
    x = np.array(window, dtype=np.float32)
    if x.shape != (WINDOW_SIZE, INPUT_SIZE):
        return "waiting", {}, 0.0
    probs = torch.softmax(model(torch.from_numpy(x).unsqueeze(0).to(DEVICE)), dim=1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return (
        ID_TO_LABEL.get(pred_id, str(pred_id)),
        {ID_TO_LABEL.get(i, str(i)): round(float(probs[i]), 4) for i in range(len(probs))},
        float(probs[pred_id]),
    )


def start_simulation():
    window = deque(maxlen=WINDOW_SIZE)
    for step in range(1, 10_000):
        row, sensor_view = make_sensor_row(step)
        window.append(row)
        if len(window) < WINDOW_SIZE:
            pred_label, prob_dict, summary = "waiting", {}, f"Collecting sensor rows: {len(window)} / {WINDOW_SIZE}"
        else:
            pred_label, prob_dict, pred_prob = predict_from_window(list(window))
            summary = f"Prediction: {pred_label}\n\nConfidence: {pred_prob * 100:.2f}%\n\nModel: {MODEL_PATH}"
        yield [[key, value] for key, value in sensor_view.items()], pred_label, prob_dict, summary
        time.sleep(5)


with gr.Blocks(title="Behavior Prediction Demo") as demo:
    gr.Markdown("# Behavior Prediction Demo")
    start_btn = gr.Button("Start realtime prediction", variant="primary")
    with gr.Row():
        sensor_output = gr.Dataframe(headers=["item", "value"], label="Current sensor data", interactive=False)
        with gr.Column():
            predicted_label = gr.Textbox(label="Predicted behavior", interactive=False)
            probabilities = gr.Label(label="Class probabilities")
    summary = gr.Markdown()
    start_btn.click(fn=start_simulation, inputs=[], outputs=[sensor_output, predicted_label, probabilities, summary])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
