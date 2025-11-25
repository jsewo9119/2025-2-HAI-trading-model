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


## 프로젝트 변경사항 요약 (Change Summary)

### 🔹 신규 추가 파일

#### modules/env.py
	•	OpenAI Gym 스타일의 TradingEnv 환경 구현
	•	매수/매도/홀드 액션 처리
	•	비율 기반 포지션(%) 적용
	•	수수료 반영
	•	step(), reset() 메서드 구현
	•	백테스트용 run_with_agent() 제공

### 🔹 주요 수정 파일

#### modules/agent.py
	•	QNetwork 모델 구조 수정
	•	epsilon-greedy 정책 추가 (act())
	•	모델 저장 기능 추가 (save())

#### modules/trainer.py
	•	기존 dataset 기반 학습 제거
	•	train_with_env()로 환경과 상호작용하며 학습되도록 변경
	•	episode reward 기록 기능 추가

#### training.ipynb
	•	학습 환경 변경 반영
	•	epsilon schedule 추가
	•	최근 365일 성능 비교 추가

## 시스템 구조 변화

### Before
	•	dataset 기반 offline 학습
	•	미래 수익률 기반 reward (lookahead leakage 위험)
	•	거의 action=0(홀드) 편향 발생

### After
	•	환경 기반 RL 구조로 전환
	•	episode 단위 학습
	•	실시간 reward feedback
	•	행동 다양성 크게 증가
	•	최근 1년/전체 기간 백테스트 가능