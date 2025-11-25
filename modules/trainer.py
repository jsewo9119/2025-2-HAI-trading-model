import random
import torch


def train_memory(agent, gamma, batch_size, optimizer, loss_fn):
    if len(agent.memory) < batch_size:
        return

    batch = random.sample(agent.memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = [s.unsqueeze(0) for s in states] 
    next_states = [ns.unsqueeze(0) for ns in next_states]

    states = torch.cat(states, dim=0)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.cat(next_states, dim=0)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    current_qs = agent.nn(states).gather(1, actions)
    next_qs = agent.nn(next_states).max(1)[0].unsqueeze(1)
    target_qs = rewards + (gamma * next_qs * (1 - dones))

    loss = loss_fn(current_qs, target_qs.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train(agent, dataset, CONFIGS, optimizer, loss_fn):
    epochs = CONFIGS.EPOCHS
    gamma = CONFIGS.GAMMA
    batch_size = CONFIGS.BATCH_SIZE
    
    epsilon      = CONFIGS.EPSILON_START
    epsilon_min  = CONFIGS.EPSILON_MIN
    epsilon_decay = CONFIGS.EPSILON_DECAY
    
    loss_history = []

    for i in range(len(dataset) - 1):
        state, reward = dataset[i]
        next_state, _ = dataset[i + 1]

        action = agent.act(state, epsilon)
        agent.remember(state, action.item(), reward.item(), next_state, 0)

    state, reward = dataset[-1]
    action = agent.act(state, epsilon)
    agent.remember(state, action.item(), reward.item(), state, 1)

    for epoch in range(1, epochs+1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")

        loss = train_memory(agent, gamma, batch_size, optimizer, loss_fn)
        if loss is not None:
            loss_history.append(loss)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return loss_history


# TradingEnv 기반 DQN 학습용 함수
def train_with_env(agent, env, CONFIGS, optimizer, loss_fn):
    """
    TradingEnv 기반 DQN 학습용 함수.

    - agent: DQNAgent 인스턴스 (agent.act, agent.remember, agent.nn 사용)
    - env: TradingEnv 인스턴스
    - CONFIGS: 설정 (EPOCHS, GAMMA, BATCH_SIZE, EPSILON_* 등)
    - optimizer, loss_fn: PyTorch 옵티마이저 / 손실 함수

    반환:
    - loss_history: 학습 중 발생한 loss들의 리스트
    - episode_rewards: 각 에피소드 총 reward 리스트
    """
    epochs = CONFIGS.EPOCHS
    gamma = CONFIGS.GAMMA
    batch_size = CONFIGS.BATCH_SIZE

    epsilon = CONFIGS.EPSILON_START
    epsilon_min = CONFIGS.EPSILON_MIN
    epsilon_decay = CONFIGS.EPSILON_DECAY

    # 너무 자주 업데이트되는 것을 막기 위한 설정
    update_every = 4                 # 4 스텝마다 한 번만 학습
    min_memory = batch_size * 2      # 최소한 이만큼 모이면 학습 시작

    loss_history = []        # 에피소드 평균 loss 기록
    episode_rewards = []     # 에피소드 총 reward 기록

    global_step = 0


    for episode in range(1, epochs + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        # 에피소드 내부 loss를 평균 내기 위한 변수
        episode_loss_sum = 0.0
        episode_loss_count = 0

        while not done:
            # 1) epsilon-greedy 행동 선택
            action = agent.act(state, epsilon)

            # 2) 환경 한 스텝 진행
            next_state, reward, done, info = env.step(action)

            # 3) 리플레이 메모리에 transition 저장
            agent.remember(state, action, reward, next_state, done)

            global_step += 1

            state = next_state
            total_reward += reward

            # 4) 일정 스텝마다, 그리고 메모리가 충분히 쌓였을 때만 DQN 업데이트
            if len(agent.memory) >= min_memory and global_step % update_every == 0:
                loss = train_memory(agent, gamma, batch_size, optimizer, loss_fn)
                if loss is not None:
                    episode_loss_sum += loss
                    episode_loss_count += 1

        # 에피소드 종료 후 epsilon 감소
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

         # 에피소드 평균 loss 기록
        if episode_loss_count > 0:
            mean_loss = episode_loss_sum / episode_loss_count
            loss_history.append(mean_loss)
        else:
            mean_loss = None


        if mean_loss is not None:
            print(
                f"Episode {episode}/{epochs} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Mean Loss: {mean_loss:.4f}"
            )
        else:
            print(
                f"Episode {episode}/{epochs} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Mean Loss: N/A"
            )

    return loss_history, episode_rewards