# CareBridge Behavior Analysis AI

CareBridge Behavior Analysis AI는 독거노인 또는 돌봄 대상자의 환경 센서 시계열 데이터를 바탕으로 현재 행동 상태를 예측하는 행동 분석 모듈입니다. 온도, 습도, 조도, 활동 감지 IR, CO2, TVOC 같은 센서값과 시간 정보, 사용자 메타 정보를 조합해 일정 길이의 window 데이터를 만들고, LSTM 모델로 행동 클래스를 분류합니다.

이 프로젝트의 목표는 단순히 모델을 학습하는 것이 아니라, 원본 데이터 정리부터 전처리, window 생성, 학습, 추론 API 제공까지 이어지는 작은 ML 파이프라인을 한 폴더 안에서 일관되게 관리하는 것입니다.

## 예측 대상

현재 모델은 학습 당시 저장된 클래스 매핑을 기준으로 다음 행동 상태를 예측합니다.

- 기타
- 수면
- 외출
- 식사

클래스 라벨은 [configs/paths.yaml](configs/paths.yaml)에서 관리합니다.

## 동작 흐름

1. 원본 또는 중간 산출 CSV 데이터를 읽습니다.
2. 센서값을 정리하고, 결측치와 이상 범위를 처리합니다.
3. 행동 라벨과 범주형 변수를 숫자 ID로 변환합니다.
4. 10분 단위 센서 데이터를 1시간 window로 묶습니다.
5. window 데이터를 `.npz` 파일로 저장합니다.
6. LSTM 모델을 학습하고 가장 좋은 checkpoint를 저장합니다.
7. FastAPI 또는 Gradio에서 checkpoint를 로드해 예측 결과를 제공합니다.

## 프로젝트 구조

```text
src/
  preprocessing/
    features.py              센서 feature 정리, 인코딩, 매핑 생성
    windowing.py             시계열 데이터를 고정 길이 window로 변환
    run_preprocessing.py     전처리 실행 진입점
    make_windows.py          window 생성 실행 진입점
    restructure_data.py      원본 데이터 재구성용 자리
  train/
    dataset.py               npz window 데이터셋 로더
    engine.py                학습/검증 루프
    train_lstm.py            LSTM 학습 실행 진입점
  inference/
    predictor.py             모델 checkpoint 로딩과 라벨 로딩
  api/
    app_api.py               FastAPI 예측 서버
    app_gradio.py            Gradio 데모
  models/
    lstm.py                  LSTM baseline 모델
  utils/
    config.py                YAML 설정 로더와 경로 resolver
    io.py                    공통 파일 입출력 함수

data/                         원본 데이터 위치
notebooks/                    EDA 및 실험 노트북
configs/                      경로, 전처리, 학습 설정
output/                       전처리 결과, window 데이터, 모델 checkpoint
requirements.txt
README.md
```

## 설정 파일

- [configs/paths.yaml](configs/paths.yaml): 데이터, output, 모델 checkpoint 경로와 API 라벨
- [configs/preprocessing.yaml](configs/preprocessing.yaml): 전처리 컬럼, window 크기, stride, feature 목록
- [configs/training.yaml](configs/training.yaml): seed, batch size, epoch, learning rate, LSTM hidden size

경로는 프로젝트 루트인 `behavior_analysis` 기준의 상대경로로 적습니다.

## 환경 설정

```powershell
cd C:\Users\codeit44\Desktop\side_project\behavior_analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 파이프라인 실행

원본 학습 데이터가 삭제된 상태라면 전처리와 학습 단계는 입력 파일 없음 오류가 나는 것이 정상입니다. 이미 학습된 모델 checkpoint가 있으면 API는 바로 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe src\preprocessing\run_preprocessing.py
.\.venv\Scripts\python.exe src\preprocessing\make_windows.py
.\.venv\Scripts\python.exe src\train\train_lstm.py
```

## FastAPI 실행

```powershell
.\.venv\Scripts\python.exe -m uvicorn --app-dir src api.app_api:app --host 127.0.0.1 --port 8000
```

API 문서:

```text
http://127.0.0.1:8000/docs
```

상태 확인:

```text
http://127.0.0.1:8000/health
```

## Gradio 데모 실행

```powershell
.\.venv\Scripts\python.exe src\api\app_gradio.py
```

## 산출물

- `output/behavior_preprocessed/`: 전처리된 CSV
- `output/behavior_window_1h/`: 학습용 window `.npz`
- `output/lstm_window_1h_baseline/`: 학습 history, config, best model checkpoint

대용량 데이터와 모델 산출물은 로컬에서 관리하고, 필요할 때만 공유하는 것을 권장합니다.
