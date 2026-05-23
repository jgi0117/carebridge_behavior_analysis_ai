# =========================================================
# app_gradio.py
# 5초마다 랜덤 센서 데이터 생성 + LSTM 행동 예측 데모
# =========================================================

from pathlib import Path
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import gradio as gr


# =========================================================
# 1. Config
# =========================================================
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "output" / "lstm_window_1h_baseline" / "best_lstm_window_1h.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID_TO_LABEL = {
    0: "기타",
    1: "수면",
    2: "외출",
    3: "식사",
}


# =========================================================
# 2. Model
# =========================================================
class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.classifier(last_hidden)


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    model = LSTMBaseline(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_classes=ckpt["num_classes"],
        num_layers=ckpt.get("num_layers", 1),
        dropout=ckpt.get("dropout", 0.0),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, ckpt


model, ckpt = load_model()

WINDOW_SIZE = ckpt["window_size"]
INPUT_SIZE = ckpt["input_size"]


# =========================================================
# 3. Random Sensor Generator
# =========================================================
def make_sensor_row(step):
    temperature = round(random.uniform(20.0, 28.0), 2)
    humidity = round(random.uniform(35.0, 70.0), 2)
    illuminance = round(random.uniform(0.0, 700.0), 2)
    activity_ir = round(random.choice([0, 0, 0, 1, 2, 3, 5, 10]), 2)
    co2 = round(random.uniform(400.0, 1200.0), 2)
    tvoc = round(random.uniform(0.0, 0.8), 4)

    hour = step % 24
    weekday = (step // 24) % 7
    age = 78
    gender_id = 0
    environment_id = 1

    activity_ir_log1p = np.log1p(max(activity_ir, 0))
    co2_log1p = np.log1p(max(co2, 0))
    tvoc_log1p = np.log1p(max(tvoc, 0))
    illuminance_log1p = np.log1p(max(illuminance, 0))

    row = [
        temperature,
        humidity,
        illuminance,
        activity_ir,
        co2,
        tvoc,
        activity_ir_log1p,
        co2_log1p,
        tvoc_log1p,
        illuminance_log1p,
        hour,
        weekday,
        age,
        gender_id,
        environment_id,
    ]

    sensor_view = {
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

    return row, sensor_view


@torch.no_grad()
def predict_from_window(window):
    x = np.array(window, dtype=np.float32)

    if x.shape != (WINDOW_SIZE, INPUT_SIZE):
        return "대기 중", {}, 0.0

    x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = ID_TO_LABEL.get(pred_id, str(pred_id))
    pred_prob = float(probs[pred_id])

    prob_dict = {
        ID_TO_LABEL.get(i, str(i)): round(float(probs[i]), 4)
        for i in range(len(probs))
    }

    return pred_label, prob_dict, pred_prob


# =========================================================
# 4. Streaming Function
# =========================================================
def start_simulation():
    window = deque(maxlen=WINDOW_SIZE)

    for step in range(1, 10_000):
        row, sensor_view = make_sensor_row(step)
        window.append(row)

        if len(window) < WINDOW_SIZE:
            pred_label = "대기 중"
            prob_dict = {}
            summary = f"""
## 데이터 수집 중

최근 시점 수: **{len(window)} / {WINDOW_SIZE}**

LSTM 예측을 위해 최근 1시간 데이터가 필요합니다.
"""
        else:
            pred_label, prob_dict, pred_prob = predict_from_window(list(window))
            summary = f"""
## 현재 예측 결과: **{pred_label}**

- 예측 확률: **{pred_prob * 100:.2f}%**
- 최근 window: **{WINDOW_SIZE}개 시점**
- 갱신 주기: **5초**
"""

        sensor_table = [
            ["Step", sensor_view["step"]],
            ["Hour", sensor_view["hour"]],
            ["Weekday", sensor_view["weekday"]],
            ["Temperature", sensor_view["temperature"]],
            ["Humidity", sensor_view["humidity"]],
            ["Illuminance", sensor_view["illuminance"]],
            ["Activity IR", sensor_view["activity_ir"]],
            ["CO2", sensor_view["co2"]],
            ["TVOC", sensor_view["tvoc"]],
        ]

        yield sensor_table, pred_label, prob_dict, summary

        time.sleep(5)


# =========================================================
# 5. UI
# =========================================================
with gr.Blocks(title="실시간 행동 예측 AI 데모") as demo:
    gr.Markdown(
        """
# 실시간 독거노인 행동 예측 AI 데모

시작 버튼을 누르면 **5초마다 랜덤 센서 데이터**가 생성되고,  
최근 1시간 window를 기반으로 행동 상태를 예측합니다.

예측 클래스: **기타 / 수면 / 외출 / 식사**
"""
    )

    start_btn = gr.Button("실시간 예측 시작", variant="primary")

    with gr.Row():
        sensor_output = gr.Dataframe(
            headers=["항목", "값"],
            label="현재 센서 데이터",
            interactive=False,
        )

        with gr.Column():
            predicted_label = gr.Textbox(
                label="현재 예측 행동",
                interactive=False,
            )

            probabilities = gr.Label(
                label="클래스별 확률",
            )

    summary = gr.Markdown()

    start_btn.click(
        fn=start_simulation,
        inputs=[],
        outputs=[
            sensor_output,
            predicted_label,
            probabilities,
            summary,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )