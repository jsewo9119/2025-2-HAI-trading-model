import torch


class TradingEnv:
    """
    간단한 1종목 트레이딩 환경.
    - prices: 1차원 가격 시계열 (list/ndarray/torch.Tensor)
    - window_size: state에 포함할 과거 일 수
    - action: 0 = 현금 0% (All Cash), 1 = 주식 50% 보유, 2 = 주식 100% 보유 (풀 포지션)
    - 포지션: 여러 주까지 보유 가능 (num_shares ≥ 0, 리밸런싱 방식)
    - reward: 
      * 주식을 보유 중이면 (V_{t+1} - V_t) / V_t (수익률)
      * 주식을 보유하지 않으면 작은 패널티(-0.001)
    """

    def __init__(self, prices, window_size: int, fee_rate: float = 0.001, initial_cash: float = 1_000_000):
        # 가격을 torch 텐서로 통일
        self.prices = torch.as_tensor(prices, dtype=torch.float32)
        self.window_size = int(window_size)
        self.fee_rate = float(fee_rate)
        self.initial_cash = float(initial_cash)

        self.n_steps = len(self.prices)

        # 에피소드 상태 변수들
        self.current_step: int | None = None
        self.cash: float | None = None
        self.num_shares: int | None = None  # 현재 보유 주식 수 (0 또는 1)
        self.position: int | None = None  # 0 = 현금, 1 = 롱
        self.done: bool = False

    def reset(self):
        """
        에피소드를 처음 상태로 초기화하고 state를 반환.
        state: 현재 시점 기준 과거 window_size일 종가 (shape: [window_size])
        """
        if self.n_steps < self.window_size:
            raise ValueError("prices 길이가 window_size보다 짧습니다.")

        # window의 마지막 인덱스를 window_size-1에서 시작
        self.current_step = self.window_size - 1
        self.cash = self.initial_cash
        self.num_shares = 0
        self.position = 0  # 0 = no position
        self.done = False

        state = self._get_state()
        return state

    def _get_state(self) -> torch.Tensor:
        """
        현재 current_step를 기준으로 과거 window_size 만큼의 가격을 잘라서 state로 사용.
        shape: [window_size]
        """
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1  # slice 끝은 포함 X라 +1
        window = self.prices[start:end]
        # 안전을 위해 복사본 반환 (외부에서 inplace 수정 방지)
        return window.clone()

    def _portfolio_value(self, price: float) -> float:
        """
        현재 가격에서의 포트폴리오 총 가치 = 현금 + 주식 평가액
        """
        return float(self.cash) + float(self.num_shares) * float(price)

    def step(self, action: int):
        """
        환경 한 스텝 진행.
        action: 0 = Hold, 1 = Buy, 2 = Sell

        반환:
        - next_state: 다음 state (torch.Tensor, shape [window_size])
        - reward: float (V_{t+1} - V_t)
        - done: bool (에피소드 종료 여부)
        - info: dict (포트폴리오 가치, 현금, 포지션 등 부가 정보)
        """
        if self.done:
            raise RuntimeError("에피소드가 끝난 상태에서 step()을 호출했습니다. reset()을 먼저 호출하세요.")

        # 현재 가격과 현재 포트폴리오 가치
        current_price = float(self.prices[self.current_step])
        prev_value = self._portfolio_value(current_price)

        # ----- 액션 → 목표 포지션 비율 (리밸런싱 방식) -----
        # 0 = 0% 투자, 1 = 50% 투자, 2 = 100% 투자
        if action == 0:
            target_frac = 0.0
        elif action == 1:
            target_frac = 0.5
        elif action == 2:
            target_frac = 1.0
        else:
            target_frac = 0.0  # 안전하게 처리

        # 현재 포트폴리오 가치와 주식 비중
        total_value = prev_value
        current_stock_value = float(self.num_shares) * current_price
        current_frac = current_stock_value / total_value if total_value > 0 else 0.0

        # 목표 주식 가치
        target_stock_value = target_frac * total_value
        diff_value = target_stock_value - current_stock_value  # >0: 매수, <0: 매도

        # ----- 리밸런싱: diff_value를 기준으로 매수/매도 수량 결정 -----
        if diff_value > 0:
            # 매수: 수수료를 고려한 1주당 비용
            unit_cost = current_price * (1.0 + self.fee_rate)
            max_shares = int(diff_value / unit_cost)
            if max_shares > 0:
                cost = max_shares * unit_cost
                # 혹시 diff_value 계산 등으로 인해 cost가 total_value를 초과할 수 있으므로
                # cash 한도 내에서만 매수
                if self.cash >= cost:
                    self.cash -= cost
                    self.num_shares += max_shares
        elif diff_value < 0:
            # 매도: 수수료를 고려한 1주당 수령액
            unit_proceed = current_price * (1.0 - self.fee_rate)
            # 줄이고 싶은 주식 가치에 해당하는 주식 수 (양수)
            shares_to_sell = int(-diff_value / unit_proceed)
            # 보유 수보다 많이 팔 수는 없으므로 제한
            shares_to_sell = min(shares_to_sell, self.num_shares)
            if shares_to_sell > 0:
                proceeds = shares_to_sell * unit_proceed
                self.cash += proceeds
                self.num_shares -= shares_to_sell

        # position 플래그 업데이트 (보유 여부)
        self.position = 1 if self.num_shares > 0 else 0

        # ----- 시간 한 칸 전진 -----
        if self.current_step < self.n_steps - 1:
            self.current_step += 1
        else:
            # 이미 마지막 가격이면 그대로 done 처리
            self.done = True

        new_price = float(self.prices[self.current_step])
        new_value = self._portfolio_value(new_price)

        # ----- reward 설계 -----
        # 주식을 보유 중이면: 수익률 (V_{t+1} - V_t) / V_t
        # 주가 수익률 (기회비용 계산용)
        if current_price > 0:
            price_return = (new_price - current_price) / current_price
        else:
            price_return = 0.0

        if prev_value > 0:
            if self.num_shares > 0:
                # 주식을 들고 있을 때: 포트폴리오 수익률
                reward = (new_value - prev_value) / prev_value
            else:
                # 주식을 안 들고 있을 때:
                #   - 주가가 오르면 -> 음수 (기회비용)
                #   - 주가가 떨어지면 -> 양수 (잘 피했다)
                reward = -price_return
        else:
            reward = 0.0

        next_state = self._get_state()
        # 마지막 인덱스에 도달하면 done = True
        if self.current_step >= self.n_steps - 1:
            self.done = True

        info = {
            "portfolio_value": new_value,
            "cash": self.cash,
            "position": self.position,
            "num_shares": self.num_shares,
            "price": new_price,
            "step": self.current_step,
        }

        return next_state, reward, self.done, info
    
    def run_with_agent(self, agent, epsilon: float = 0.0):
        """
        학습된 agent를 사용하여 환경을 처음부터 끝까지 1회 실행.
        epsilon=0 → 완전 greedy 정책.

        반환:
        - states: state 리스트
        - actions: action 리스트
        - infos: info 리스트 (각 스텝의 포트폴리오 정보)
        """
        states = []
        actions = []
        infos = []

        state = self.reset()
        done = False

        while not done:
            # 학습이 끝난 agent → epsilon=0 (탐험 없이 greedy 행동)
            action = agent.act(state, epsilon)

            next_state, reward, done, info = self.step(action)

            states.append(state)
            actions.append(action)
            infos.append(info)

            state = next_state

        return states, actions, infos