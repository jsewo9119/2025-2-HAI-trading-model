# 2025-2 HAI DQN Trading Model Study

### 환경설정 (Required)

```sh
pip install uv
uv add -r requirements.txt
```

### 실행

```sh
uv run main.py
```

### 학습
- training.ipynb

##### 모듈 테스트 (Optional)

```sh
uv run -m modules.[module_name]
```


## TODO

#### modules/dataset.py
- [line 30] 데이터 전처리(preprocess) / Reward 산출방식
```py
def preprocess(self, df):
    ...
```

#### modules/agent.py
- [line 6] 모델 구조
```py
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        ...
```

#### training.ipynb
- [cell 3] hyperparameter tuning
```py
CONFIGS = edict({
    "GET_DATASET": True,
    "TICKERS": [
        "KS11",
        "KQ11",
    ],
    "DATAFRAME_PATH": "./data",
    "WINDOW_SIZE": 10,
    "OUTPUT_SIZE": 2,
    "EPOCHS": 50,
    "GAMMA": 0.95,
    "BATCH_SIZE": 2,
    "LEARNING_RATE": 1e-5,
})
```