import torch
import torch.utils.data as data


class StockDataset(data.Dataset):
    """
    prices: Close prices
    rewards: buy = 1, sell = 0
    """

    prices = None
    rewards = None
    window_size = 10
    term = 5

    def __init__(self, df, window_size, term):
        self.window_size = window_size
        self.term = term
        self.prices, self.rewards = self.preprocess(df)

    def __len__(self):
        return len(self.rewards) - self.window_size - self.term + 1

    def __getitem__(self, idx):
        return (
            self.prices[idx : idx + self.window_size],
            self.rewards[idx],
        )

    def preprocess(self, df) -> tuple[torch.FloatTensor, torch.LongTensor]:
        # TODO: [edit here]
        
        prices = torch.FloatTensor(df["Close"].values)
        rewards = []

        for i in range(len(prices) - self.window_size - self.term + 1):
            rewards.append(
                max(prices[i + self.window_size : i + self.window_size + self.term]) - prices[i + self.window_size - 1]
            )

        rewards = torch.FloatTensor(rewards)

        return prices, rewards


def gen_dataset(dfs, CONFIGS) -> data.ConcatDataset:
    window_size = CONFIGS.WINDOW_SIZE
    term = CONFIGS.OUTPUT_SIZE
    datasets = [StockDataset(df, window_size, term) for df in dfs]
    return data.ConcatDataset(datasets)