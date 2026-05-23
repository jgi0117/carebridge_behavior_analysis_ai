# CareBridge Behavior Analysis AI

CareBridge Behavior Analysis AI는 시니어 케어 서비스 CareBridge의 행동 분석 모듈입니다. 센서 데이터를 일정한 시간 구간의 시계열 특징으로 변환하고, LSTM 기반 분류 모델을 통해 사용자의 행동 상태를 예측합니다.

이 저장소에는 행동 분석 모듈의 소스 코드, 노트북, 의존성 파일만 포함합니다. 학습 데이터, 모델 가중치, 생성된 window 데이터, 실행 결과물은 저장소에 포함하지 않습니다.

## 주요 기능

이 모듈은 센서 기록을 고정 길이의 행동 window로 변환한 뒤, LSTM baseline 모델을 학습해 행동 상태를 분류합니다. 학습된 모델은 FastAPI 서버 또는 Gradio 데모를 통해 추론에 사용할 수 있습니다.

입력 특징은 온도, 습도, 조도, 활동 IR, CO2, TVOC, 시간 정보, 사용자 메타데이터 등 환경 및 활동 관련 센서 값으로 구성됩니다.

## 동작 방식

1. 원본 센서 기록을 일관된 표 형식으로 재구성합니다.
2. 전처리 스크립트가 데이터를 정제하고 필요한 특징을 변환합니다.
3. window 생성 스크립트가 1시간 단위의 고정 길이 시퀀스 샘플을 만듭니다.
4. LSTM 모델이 생성된 window의 시계열 패턴을 학습합니다.
5. 추론 시 최근 센서 시퀀스를 입력받아 행동 클래스와 확률을 반환합니다.

## 프로젝트 구조

```text
app/
  app_api.py                     FastAPI 추론 서버
  app_gradio.py                  Gradio 데모 앱
notebooks/
  EDA.ipynb                      탐색적 데이터 분석 노트북
  Preprocessed_data_EDA.py.ipynb 전처리 데이터 분석 노트북
src/
  restruct_data.py               원본 데이터 재구성
  preprocessing.py               센서 데이터 전처리
  make_behavior_window_1h.py     1시간 행동 window 생성
  train_lstm_window_1h_baseline.py
requirements.txt
```

## 환경 설정

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 학습

로컬 학습 데이터를 `data/` 아래에 둔 뒤, 필요한 전처리 및 window 생성 스크립트를 실행합니다.

```powershell
python src\preprocessing.py
python src\make_behavior_window_1h.py
python src\train_lstm_window_1h_baseline.py
```

생성 파일과 학습된 가중치는 `output/`, `model/` 등 Git에서 제외되는 로컬 디렉터리에 저장됩니다.

## 추론

FastAPI 실행:

```powershell
uvicorn app.app_api:app --reload
```

Gradio 실행:

```powershell
python app\app_gradio.py
```

## 저장소 정책

다음 항목은 의도적으로 Git에 포함하지 않습니다.

- 학습 및 검증 데이터
- 생성된 `.npz` window 파일
- `.pt`, `.pth` 등 학습된 모델 가중치
- 실행 결과 및 실험 로그
- 로컬 가상환경

## 라이선스

이 프로젝트는 Apache License 2.0을 따릅니다. 자세한 내용은 `LICENSE`와 `NOTICE`를 참고하세요.
