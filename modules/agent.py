import torch
import torch.nn as nn
from collections import deque
import random


class QNetwork(nn.Module):
    input_size = 10
    output_size = 3

    def __init__(self, input_size, output_size):
        # TODO: [edit here]

        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc4(x))  
        x = self.fc3(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNAgent:
    nn = None
    memory = None

    def __init__(self, CONFIGS):
        input_size = CONFIGS.WINDOW_SIZE
        output_size = CONFIGS.OUTPUT_SIZE
        self.nn = QNetwork(input_size, output_size)
        self.memory = deque(maxlen=10000)

        self.output_size = CONFIGS.OUTPUT_SIZE

    def act(self, state, epsilon: float):
        # epsilon 확률로 랜덤 행동 (탐험)
        if random.random() < epsilon:
            return random.randrange(self.output_size)

        # 나머지는 Q-network 기반 argmax (이용)
        with torch.no_grad():
            q_values = self.nn(state)
        return torch.argmax(q_values).item()
    
    def parameters(self):
        return self.nn.parameters()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def actions(self, dataset):
        actions = []
        for idx, (x, _) in enumerate(dataset):
            with torch.no_grad():
                q = self(x)
                actions.append([idx, q.item()])
    def save(self, path):
        torch.save(self.state_dict(), path)

        return actions
